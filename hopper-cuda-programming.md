# GPU Kernel 编程准则 — Hopper (H200 / H800)

## 目标平台

本文档针对 NVIDIA Hopper 架构，适用于：

- **GPU 型号：** H200（GH100，132 SM，141 GB HBM3e @ 4.8 TB/s）  
  H800（GH100，132 SM，80 GB HBM3/HBM2e @ ~2.0 TB/s，NVLink 受限）
- **Compute Capability：** 9.0（`sm_90` / `sm_90a`）
- **CUDA Toolkit：** ≥ 12.0（`sm_90a` PTX 需要 CUDA 12.0+）
- **编译目标：** `-arch=sm_90`（或 `-gencode arch=compute_90,code=sm_90`）

编译命令示例：
```bash
nvcc -arch=sm_90 -lineinfo -O3 -o my_kernel my_kernel.cu
```

> **H800 vs H200 注意事项：**
> - 相同的 GH100 die，相同的 SM 架构，编程模型完全一致。
> - 区别在于 NVLink 带宽（H800: 400 GB/s vs H100/H200: 900 GB/s）和 HBM 带宽（H800: ~2.0 TB/s vs H200: 4.8 TB/s）。
> - NCU metric 名称和 SASS 指令集完全相同，chip 名均为 `gh100`。
> - 对比 B200/B300（Blackwell）：Hopper 没有 TMEM、没有 FP4、没有 tcgen05 指令。

---

## Hopper H200/H800 架构关键参数速查

| 参数 | H200 数值 | H800 数值 | 对比 B200 |
|------|----------|----------|----------|
| SM 数量 | 132 | 132 | 148 |
| 每 SM 最大 warp 数 | 64 | 64 | 64 |
| 每 SM 寄存器文件 | 64K × 32-bit | 64K × 32-bit | 64K × 32-bit |
| 每线程最大寄存器 | 255 | 255 | 255 |
| 每 SM 最大 thread block 数 | 32 | 32 | 32 |
| Shared Memory 每 SM（可配置）| 最高 228 KB | 最高 228 KB | 最高 228 KB |
| **Tensor Memory (TMEM)** | **无** | **无** | **256 KB/SM** |
| L2 Cache | 50 MB | 50 MB | 126 MB |
| HBM 容量 | 141 GB HBM3e | 80 GB HBM3/HBM2e | 192 GB HBM3e |
| HBM 带宽 | 4.8 TB/s | ~2.0 TB/s | 8 TB/s |
| FP4 Tensor | 不支持 | 不支持 | 9 PFLOPS |
| FP8 Tensor（dense） | 1.98 PFLOPS | 1.98 PFLOPS | 4.5 PFLOPS |
| FP16/BF16 Tensor（dense）| 0.99 PFLOPS | 0.99 PFLOPS | 2.25 PFLOPS |
| NVLink 带宽 | 900 GB/s（NVLink 4） | **400 GB/s（受限）** | 1.8 TB/s（NVLink 5） |
| 最大 Cluster size（portable） | 8 | 8 | 8 |

---

## Hopper 核心特性与编程要点

### 1. 第 4 代 Tensor Core 与 wgmma 指令

Hopper 的 Tensor Core 以 **warp-group（4 个 warp，128 线程）** 为单位进行 MMA 运算，通过 `wgmma.mma_async` PTX 指令发起。

**与 Blackwell tcgen05 的关键区别：**
- **累加器在寄存器中**（不在 TMEM）：wgmma 的结果直接写入 warpgroup 内线程的寄存器。这意味着 epilogue 处理（bias、activation 等）直接在寄存器上操作，不需要 `tcgen05.ld` 这样的显式加载。
- **4 warp 协作**：wgmma 指令必须由 warpgroup 内所有 4 个 warp 同步发起（不能是单线程）。
- **异步执行**：`wgmma.mma_async` 是异步的，需要 `wgmma.wait_group` 来等待结果就绪。

**操作数来源：**
- A 矩阵：来自 Shared Memory（通过 SWIZZLE 布局）或 RF（寄存器）
- B 矩阵：必须来自 Shared Memory（smem descriptor 寻址）
- D（累加器）：在寄存器中

**编程模式（通过 CUTLASS 或 inline PTX）：**
```cuda
// 通过 CUTLASS 3.x 使用 wgmma（推荐）
#include <cutlass/gemm/collective/collective_builder.hpp>
// CUTLASS 会自动处理 wgmma + TMA + mbarrier pipeline

// 直接使用 PTX（仅用于理解底层，不建议生产使用）
// 需要 A 和 B 的 shared memory descriptor (smem_desc)
asm volatile(
    "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
    "{%0,%1,...}, %desc_a, %desc_b, 1, 1, 1, 0;"
    : "+f"(acc[0]), "+f"(acc[1]), ...
    : "l"(desc_a), "l"(desc_b)
);
// 等待 wgmma 完成
asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(0));
```

**实践建议：**
- **优先使用 CUTLASS 3.x 的 Hopper 路径**，它内置了 wgmma + TMA + multi-stage pipeline 的最优组合。
- 手写 wgmma 时，必须遵循 smem 的 SWIZZLE 布局要求（LDMATRIX 友好格式），否则性能损失严重。
- Warpgroup 内的 4 个 warp 必须全部到达 wgmma 发起点，不能有 divergence。

---

### 2. TMA（Tensor Memory Accelerator）— 硬件异步数据加载

TMA 是 Hopper 新增的专用数据搬运引擎，能够直接将 global memory 数据以 tile 为单位加载到 shared memory，完全绕过 SM 的 LSU 管线（不消耗 LSU 带宽）。

**TMA 的核心优势：**
- 单线程发起，无需全 warp 参与
- 支持多维 tensor 的跨步（stride）和 box 加载
- 与计算流水线异步，隐藏 global memory 延迟
- 配合 `mbarrier` 实现轻量级完成通知

**编程模式：**
```cuda
// 1. Host 端创建 TMA descriptor
CUtensorMap tma_desc;
cuTensorMapEncodeTiled(&tma_desc,
    CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
    2,                          // 维度数
    global_ptr,                 // 基地址
    {(uint64_t)K, (uint64_t)M}, // global tensor 尺寸
    {(uint64_t)K, 1},           // stride（bytes）
    {TILE_K, TILE_M},           // tile 尺寸
    {1, 1},                     // element stride
    CU_TENSOR_MAP_INTERLEAVE_NONE,
    CU_TENSOR_MAP_SWIZZLE_128B, // 与 wgmma 的 smem 格式对齐
    CU_TENSOR_MAP_L2_PROMOTION_NONE,
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

// 2. Device 端发起 TMA 加载（单线程，通常是 threadIdx.x == 0）
if (threadIdx.x == 0) {
    uint64_t* mbar = &shared_mbarrier;   // mbarrier 在 shared memory 中
    // 初始化 mbarrier，等待 1 个 transaction
    asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"(mbar));
    // 发起 TMA 加载
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(smem_ptr), "l"(&tma_desc), "r"(tile_k), "r"(tile_m), "r"(mbar)
    );
}
__syncthreads();
// 等待 TMA 完成
asm volatile(
    "mbarrier.try_wait.parity.shared.b64 %%pred, [%0], %1;"
    :: "r"(mbar), "r"(phase_bit)
);
```

**实践建议：**
- TMA 的 SWIZZLE 模式必须与 wgmma 消费 smem 的格式匹配（通常 `SWIZZLE_128B`）。
- 每个 TMA transaction 触发 `complete_tx` 事件让 mbarrier 减一；pipeline 中的多个 tile 需要多个 mbarrier。
- CUTLASS 3.x 已封装这些细节，直接用 `Sm90TmaGmmaRmemAAccum` 等 collective 即可。

---

### 3. mbarrier（轻量级同步屏障）

相比 `__syncthreads()`（等待 block 内所有线程），mbarrier 支持更细粒度的同步：

- 指定等待到达的线程/transaction 数（`arrive_count`）
- 区分 "计算 arrive" 和 "memory transaction arrive"（配合 TMA 的 `complete_tx`）
- 支持 cluster 级别的 arrive（跨 SM 同步）

**典型用法（double-buffer pipeline）：**
```cuda
// Shared memory 中的 mbarrier 数组（每个 stage 一个）
__shared__ uint64_t mbarriers[STAGES];

// 初始化
if (threadIdx.x < STAGES)
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
                 :: "r"(&mbarriers[threadIdx.x]), "r"(1));
__syncthreads();

// 生产者（TMA 线程）：发起加载后 arrive
// 消费者（wgmma 线程）：等待 mbarrier
uint32_t phase = 0;
// wait:
asm volatile("mbarrier.try_wait.parity.shared.b64 %%pred, [%0], %1;"
             :: "r"(&mbarriers[stage]), "r"(phase));
```

---

### 4. 低精度数据类型：FP8 / BF16 / FP16

Hopper 不支持 FP4，最低精度是 FP8（两种格式）：

| 格式 | 说明 | Dense 吞吐量 | 适用场景 |
|------|------|-------------|---------|
| FP8 E4M3 | 较高精度（4-bit 指数） | 1.98 PFLOPS | 前向推理、激活值 |
| FP8 E5M2 | 较大范围（5-bit 指数） | 1.98 PFLOPS | 梯度（反向传播） |
| BF16 | 标准 16-bit（训练主格式） | 0.99 PFLOPS | 权重、训练 |
| TF32 | 19-bit（10-bit mantissa） | 0.49 PFLOPS | 精度敏感训练 |
| FP64 | 双精度 | 67 TFLOPS | 科学计算 |

**FP8 量化注意：**
- Hopper 硬件支持 FP8 wgmma，但没有硬件 block scaling（不像 Blackwell 的 MXFP8/MXFP4）。
- 需要软件维护 per-tensor 或 per-channel scale factor。
- CUTLASS 3.x 和 TransformerEngine 库已封装 FP8 推理和训练流程。

---

### 5. Thread Block Cluster 与 Distributed Shared Memory (DSMEM)

Hopper 引入的 Cluster 特性（Blackwell 继承）：

- 同一个 GPC（Graphics Processing Cluster）内的多个 SM 可以组成一个 cluster
- Cluster 内的 block 可以直接读写其他 block 的 shared memory（DSMEM）
- 最大 cluster size：portable = 8，non-portable = 8（H100/H200/H800 相同）

**使用方式：**
```cuda
// 编译时设置 cluster shape
__cluster_dims__(2, 1, 1)
__global__ void my_kernel(...) {
    // cluster 内的 block 间同步
    cluster.sync();
    // 访问相邻 block 的 shared memory
    float* remote_smem = cluster.map_shared_rank(local_smem_ptr, neighbor_rank);
}
```

**适用场景：** 
- 需要在相邻 block 间传递小量数据（如 flash attention 中的 row-max reduction）
- 多个 block 共同处理同一个 KV tile 的场景

---

### 6. 关键性能准则（Hopper 特有权重）

与之前 Ampere (A100) 相比，Hopper 上以下准则的权重发生了变化：

**TMA 是新的 bottleneck 观察点（准则 15 升级）：**
- 如果 PM Sampling 显示高 `long_scoreboard` 但 DRAM 利用率不高，检查 TMA 是否正确发起异步加载。
- 没有 TMA 的 Hopper kernel 相当于浪费了大量的架构带宽优势。

**wgmma 需要更大的 tile 才能达到高效率：**
- wgmma 的最小高效 tile 是 `m64n64k16`（BF16）。小于这个 tile 的 GEMM 需要 split 或 fuse 处理。
- 与 Blackwell 的 2CTA 模式不同，Hopper 没有 CTA pair — 大 M 维度完全依赖更大的 block size。

**L2 Cache 较小（50 MB vs B200 的 126 MB）：**
- 对于 KV cache 很大的 attention kernel，Hopper 上 L2 hit rate 可能明显低于 Blackwell。
- 考虑更激进的 smem tiling 策略来减少对 L2 的依赖。

**H800 专项注意：**
- NVLink 带宽只有 H100/H200 的 44%（400 GB/s vs 900 GB/s）。在多卡通信密集的工作负载（AllReduce、Tensor Parallel）中，通信成为 bottleneck 的概率显著提高。
- 优化方向：减少通信频率（增大 micro-batch）、使用 communication-computation overlap（`nccl_allreduce` + `cudaGraph`）。

---

## 准则总览

| # | 准则 | 核心 NCU 指标 | Hopper 特有考量 |
|---|------|-------------|--------------|
| 1 | 保证足够并行度 | Occupancy, Wave 数 | 132 SM，每 SM 64 warp 上限 |
| 2 | 合并内存访问 | sectors/request, L1 hit | 同 Blackwell |
| 3 | 利用 Shared Memory | L1/L2 hit, DRAM throughput | TMA 自动填 smem，减少手写 cp.async |
| 4 | 避免 Bank Conflict | smem wavefronts | wgmma 的 smem SWIZZLE 布局已消除冲突 |
| 5 | 避免 Warp Divergence | branch efficiency | wgmma 要求 warpgroup 4 warp 同步，divergence 直接阻断 MMA |
| 6 | 合理控制寄存器压力 | register spill, occupancy | wgmma 累加器占用寄存器（不像 Blackwell 在 TMEM）— 大 GEMM tile 的寄存器压力较高 |
| 7 | 隐藏内存延迟 | long_scoreboard stalls | **用 TMA + mbarrier 替代 cp.async 链** |
| 8 | 使用合适精度 | FP64 pipe util | FP8 最低精度；无 FP4 |
| 9 | 最小化 Host-Device 同步 | 启动频率 | 同 Blackwell |
| 10 | 合理使用 Tensor Core | tensor pipe utilization | 用 wgmma；FP8 是最高吞吐选项 |
| 11 | 优化 Grid/Block 配置 | tail effect, occupancy | 同 Blackwell |
| 12 | 减少 Atomic 竞争 | long_scoreboard on atomic | 同 Blackwell |
| 13 | 向量化访存 | sectors/request, LSU 利用率 | 使用 TMA 时自动向量化；手写路径同 Blackwell |
| 14 | 利用只读路径（`__ldg`） | L1 hit rate | 同 Blackwell |
| 15 | **Pipeline: TMA + wgmma** | SM util timeline | **核心区别：TMA→smem→wgmma 三段式 pipeline** |
| 16 | 减少同步开销 | barrier stall | **用 mbarrier 替代 `__syncthreads__`；cluster.sync() 用于跨 SM** |

---

## NCU 诊断快速对照（Hopper 视角）

| NCU 信号 | 典型含义 | 优先检查 |
|---|---|---|
| `long_scoreboard > 40%`，DRAM 不高 | TMA 未异步，或 pipeline 深度不足 | 是否用了 TMA？mbarrier 是否正确 wait？|
| `sm__pipe_tensor_cycles_active ≈ 0%` | wgmma 没有被使用 | 是否在用 CUTLASS Hopper path？是否用了 scalar FMA？|
| `smsp__average_warps_issue_stalled_gmma_per_issue_active.ratio` 高 | wgmma 流水线是瓶颈（等待 `WARPGROUP.ARRIVES`） | 增大 tile 尺寸（更少 wgmma 次数）；增加 pipeline 深度；检查 wgmma 等待时机 |
| `barrier > 20%` | `__syncthreads` 过多或 mbarrier 等待时间长 | 减少同步；检查 wgmma 等待是否提前 |
| `l1tex__t_sector_hit_rate < 50%`，DRAM 高 | 数据未复用，L2 被穿透 | smem tiling 策略；Hopper L2 仅 50 MB |
| H800: `sm__throughput` 高但总体吞吐低 | 多卡通信是瓶颈（NVLink 400 GB/s） | 检查 AllReduce 等待时间；考虑通信-计算 overlap |

---

## 相关资料

- [`blackwell-cuda-programming.md`](blackwell-cuda-programming.md) — Blackwell B200/B300 编程参考（tcgen05、TMEM、FP4）
- [`reference/08-gpu-metric-names.md`](reference/08-gpu-metric-names.md) — Hopper vs Blackwell metric 名称对照表
- CUTLASS 3.x：[github.com/NVIDIA/cutlass](https://github.com/NVIDIA/cutlass)（`examples/57_hopper_grouped_gemm` 等）
- NVIDIA Hopper Architecture whitepaper：搜索 "NVIDIA Hopper GH100 whitepaper"
