# GPU Kernel 编程准则 Skill

## 目标平台

本文档针对 NVIDIA **Blackwell 系列**（B200 和 B300），两者共用相同的 tcgen05 指令集和 TMEM 编程模型，主要区别在于规格参数：

| 参数 | B200 | B300（Blackwell Ultra） |
|------|------|------------------------|
| Compute Capability | 10.0（sm_100） | 10.3（sm_103） |
| SM 数量 | 148（2 die × 74） | 160 |
| HBM 容量 | 192 GB HBM3e | 288 GB HBM3e |
| HBM 带宽 | 8 TB/s | 8 TB/s |
| FP4 Tensor（dense） | 9 PFLOPS | 15 PFLOPS |
| FP16/BF16 Tensor（dense） | 2.25 PFLOPS | 3.7 PFLOPS |
| TDP | 1,000 W | 1,400 W |
| ncu chip 名 | `gb202` | `gb300` |

- **CUDA Toolkit：** ≥ 13.0（sm_103 需要 CUDA 12.9+）
- **编译目标：**
  - B200 专用：`-arch=sm_100a` 或 `-gencode arch=compute_100a,code=sm_100a`
  - B300 专用：`-gencode arch=compute_103,code=sm_103`
  - **B200 + B300 通用**（推荐）：`-gencode arch=compute_100,code=compute_100`（`compute_100f` 家族，sm_100 和 sm_103 均可运行）

编译命令示例：
```bash
# B200 专用
nvcc -arch=sm_100a -lineinfo -O3 -o my_kernel my_kernel.cu

# B300 专用
nvcc -gencode arch=compute_103,code=sm_103 -lineinfo -O3 -o my_kernel my_kernel.cu

# B200 + B300 通用（优先使用这个）
nvcc -gencode arch=compute_100,code=compute_100 -lineinfo -O3 -o my_kernel my_kernel.cu
```

---

## Blackwell 架构关键参数速查

| 参数 | B200 | B300（Blackwell Ultra） | 对比 H200（Hopper） |
|------|------|------------------------|-------------------|
| SM 数量 | 148（2 die × 74） | 160 | 132 |
| 每 SM 最大 warp 数 | 64 | 64 | 64 |
| 每 SM 寄存器文件 | 64K × 32-bit | 64K × 32-bit | 64K × 32-bit |
| 每线程最大寄存器 | 255 | 255 | 255 |
| 每 SM 最大 thread block 数 | 32 | 32 | 32 |
| Shared Memory 每 SM（可配置） | 最高 228 KB | 最高 228 KB | 最高 228 KB |
| **Tensor Memory (TMEM) 每 SM** | **256 KB** | **256 KB** | **无** |
| L2 Cache | 126 MB | 192 MB | 50 MB |
| HBM 容量 | 192 GB HBM3e | 288 GB HBM3e | 141 GB HBM3e |
| HBM 带宽 | 8 TB/s | 8 TB/s | 4.8 TB/s |
| FP4 Tensor（dense/sparse） | 9 / 18 PFLOPS | 15 / 30 PFLOPS | 不支持 |
| FP8 Tensor（dense/sparse） | 4.5 / 9 PFLOPS | 7.5 / 15 PFLOPS | 1.98 / 3.96 PFLOPS |
| FP16/BF16 Tensor（dense/sparse） | 2.25 / 4.5 PFLOPS | 3.7 / 7.4 PFLOPS | 0.99 / 1.98 PFLOPS |
| TF32 Tensor（dense/sparse） | 1.13 / 2.25 PFLOPS | 1.85 / 3.7 PFLOPS | 0.49 / 0.99 PFLOPS |
| FP64 Tensor | 45 TFLOPS | 75 TFLOPS | 67 TFLOPS |
| NVLink 带宽 | 1.8 TB/s（NVLink 5） | 1.8 TB/s（NVLink 5） | 900 GB/s（NVLink 4） |
| 最大 Cluster size（portable / non-portable） | 8 / 16 | 8 / 16 | 8 / 8 |
| Compute Capability | sm_100 | sm_103 | sm_90 |

---

## Blackwell 独有特性与编程要点

### 1. 第 5 代 Tensor Core 与 tcgen05 指令

Blackwell 的 Tensor Core 与前代有根本性不同。核心变化：

**从 warpgroup MMA 到 warp-level 单线程 MMA：** Hopper 的 `wgmma` 以 128 线程（4 个 warp 的 warpgroup）为单位发起 MMA，而 Blackwell 的 `tcgen05.mma` 由**单个线程**代表整个 CTA 发起。这大幅降低了单条指令延迟（比 Hopper 低 3-11 倍），但要求新的编程模型。

**累加器存放在 TMEM 而非寄存器：** 线程不再"拥有" MMA 结果。`tcgen05.mma` 的输出写入 TMEM，需要通过 `tcgen05.ld` 显式搬回寄存器做后处理（epilogue）。

**操作数来源：** A 矩阵可以来自 Shared Memory 或 TMEM，B 矩阵来自 Shared Memory。D（累加器）始终在 TMEM。

**编程模式（通过 inline PTX）：**
```cuda
// TMEM 分配（由单个 warp 执行）
asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
             : : "r"(smem_addr), "r"(num_columns));

// MMA 指令（由单个线程，即 elected leader 发射）
asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, %4;"
             : : "r"(tmem_addr), "l"(a_desc), "l"(b_desc), "r"(idesc), "b"(pred));

// 从 TMEM 加载结果到寄存器
asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 %0, [%1];"
             : "=r"(reg) : "r"(tmem_addr));

// TMEM 释放
asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
             : : "r"(tmem_addr), "r"(num_columns));
```

**实践建议：**
- 对于 GEMM 类 kernel，优先使用 CUTLASS 4.x 的 Blackwell 原生路径，而非手写 tcgen05。手写 tcgen05 需要管理 leader election、mbarrier 同步、TMEM 分配/释放等大量细节。
- 手写 tcgen05 kernel 时，tile 操作数建议按 64×64 元素对齐以最大化 TMEM 带宽。
- `tcgen05.mma` 与 `tcgen05.cp`（smem→tmem）形成隐式 pipeline——按发射顺序执行。利用这一特性可以省去部分同步操作。

### 2. Tensor Memory (TMEM)

TMEM 是 Blackwell 新增的片上内存空间，每个 SM 256 KB，专用于 Tensor Core 数据通路。

**关键特性：**
- 读带宽 ~16 TB/s/SM，写带宽 ~8 TB/s/SM（远超 shared memory）
- 组织为 512 列 × 128 lane × 32-bit
- 只能通过专用指令访问：`tcgen05.ld`、`tcgen05.st`、`tcgen05.cp`（smem→tmem）
- 不能被传统 load/store 指令（`ld.shared`、`ld.global` 等）访问
- 需要显式分配和释放（类似 malloc/free）

**TMEM vs 寄存器 vs Shared Memory：**

| 特性 | 寄存器 | Shared Memory | TMEM |
|------|--------|--------------|------|
| 容量/SM | 64 KB | 最高 228 KB | 256 KB |
| 延迟 | 1 cycle | ~20-30 cycles | 专用通路 |
| 谁能访问 | 单个线程 | block 内所有线程 | Tensor Core + 显式搬运 |
| 用途 | 通用计算 | 线程间共享数据 | MMA 累加器 + operand staging |

**实践建议：**
- TMEM 主要用于 GEMM/MMA 的累加器，不要尝试将其用作通用暂存。
- 中间累加结果应尽量保持在 TMEM 中，减少 TMEM↔寄存器搬运。只在 epilogue（如 bias add、activation）时才将结果搬到寄存器。
- TMEM 分配必须由单个 warp 执行，且同一个 warp 负责分配和释放。分配列数必须是 2 的幂且≥32。
- TMEM 分配存在竞争的可能（多个 block 同时分配），高性能 kernel 应实现带重试和 nanosleep 退避的分配逻辑。

### 3. CTA Pair（2CTA）— 双 SM 协作

Blackwell 中，同一 TPC（Texture Processing Cluster）内的两个 SM 可以组成 **CTA pair** 协作执行 MMA。这通过 `cta_group::2` 修饰符启用。

**好处：**
- 两个 CTA 共享输入操作数，将每个 CTA 的 shared memory 带宽需求减半
- 有效的 MMA M 维度翻倍（如单 CTA M=128，2CTA 则 M=256）
- 这是利用满 Tensor Core 吞吐量的必要条件——单 SM 的 MMA（M=64）只能达到 ~50% 理论峰值

**实践建议：**
- 对于大型 GEMM，尽量启用 2CTA 模式以充分利用 Tensor Core。
- 使用 Thread Block Cluster（cluster_size=2）来确保 CTA pair 映射到同一 TPC。
- 2CTA 模式下 TMA 数据加载也可以跨 CTA 分摊。

### 4. 低精度数据类型：FP4 / FP6 / FP8 与 Block Scaling

Blackwell Tensor Core 原生支持的精度格式（吞吐量关系）：

| 格式 | 说明 | Dense 吞吐量 | 相对 FP16 |
|------|------|-------------|----------|
| NVFP4 | E2M1 + block-16 E4M3 scale + per-tensor FP32 scale | 9 PFLOPS | 4x |
| MXFP4 | E2M1 + block-32 E8M0 scale | 9 PFLOPS | 4x |
| MXFP6 (E3M2 / E2M3) | 6-bit float + block scaling | 4.5 PFLOPS | 2x |
| MXFP8 / FP8 (E4M3 / E5M2) | 8-bit float | 4.5 PFLOPS | 2x |
| FP16 / BF16 | 标准 16-bit | 2.25 PFLOPS | 1x |
| TF32 | 19-bit（10-bit mantissa） | 1.13 PFLOPS | 0.5x |
| FP64 | 双精度 | 45 TFLOPS | 0.02x |

**FP4 与 FP6 吞吐量相同的原因：** FP4 和 FP8 共享相同物理 Tensor Core 电路，FP4 实现 2x 吞吐是因为每个 cycle 能处理双倍数量的元素。FP6 和 FP8 共享电路，因此 FP6 吞吐量等于 FP8（不是 FP4 和 FP8 的中间值）。

**NVFP4 vs MXFP4：**
- NVFP4 使用更小的 block size（16 vs 32）和更精确的 scale format（E4M3 vs E8M0），通常精度更好
- MXFP4 是 OCP 社区标准，跨平台兼容性更好
- cuBLAS 13.2 对 B200 上的 NVFP4 已有优化支持

**实践建议：**
- **LLM 推理优先考虑 NVFP4（W4A4）或 FP8（W8A8）：** NVFP4 将模型显存占用降低 ~1.8x（相比 FP8），同时保持接近 FP8 的精度。
- **训练推荐 BF16 + FP8 混合：** BF16/FP16 用于累加保证稳定性，FP8 用于前向/反向的矩阵乘法。
- **不要盲目用最低精度：** FP4 在某些层可能导致显著精度退化（实测平均 ~8.2% perplexity 退化），应该逐层选择精度。
- **编译注意：** 使用 NVFP4/MXFP4 需要 CUTLASS 4.x 或 cuBLAS 的相应 API。CUDA 13.2 的 cuBLAS 已支持 Grouped GEMM 的 MXFP8。

### 5. 硬件解压引擎（Decompression Engine）

B200 内置硬件解压引擎，支持 LZ4、Snappy、Zstandard、GZIP、Bitcomp、ANS 等格式，吞吐量可达 ~539 GB/s（100 MB 块，亚毫秒延迟）。

**使用场景：**
- 大型 LLM 推理：将量化权重以压缩格式存储在 HBM 中，解压引擎在加载时自动解压
- 稀疏矩阵运算：将稀疏数据以 RLE/Bitcomp 压缩存储，流式解压到 TMEM 做运算
- 通过 NVIDIA nvCOMP 库可移植编程

**实践建议：**
- 当模型权重的 HBM 读取是瓶颈时（memory-bound 推理），使用硬件解压可以有效节省带宽。
- 解压引擎与 SM 并行工作，不占用 SM 计算资源。

### 6. Thread Block Cluster 与 Distributed Shared Memory (DSMEM)

继承自 Hopper 的特性，在 Blackwell 上得到增强：

- B200 支持 portable cluster size 最大 8，non-portable 最大 16（需要 `cudaFuncAttributeNonPortableClusterSizeAllowed`）
- Cluster 内的 block 可以直接读写其他 block 的 shared memory（DSMEM）
- DSMEM 与 L2 cache 带宽可以同时使用（叠加带宽）

**DSMEM 访问最佳实践：**
- 访问模式应与 global memory 类似：合并、对齐到 32 字节段
- 避免非 unit stride 访问——如需要，先拷到 local shared memory 再随机访问

### 7. CUDA 13.2 特定功能

- **cuTile（CUDA Tile）：** 新的 tile-level 编程模型，现已支持 Ampere/Ada/Blackwell（sm_80+）。提供 Python 和 C++ 两种接口，可自动生成 tcgen05 + TMA + pipeline 等底层代码。对于新 kernel 开发，cuTile 是一个比手写 PTX 更高效的起点。
- **cuBLAS Grouped GEMM：** 13.2 新增 MXFP8 Grouped GEMM 实验性支持，结合 CUDA Graph 可实现无 host 同步的 device-side shape 处理（MoE 场景 4x 加速）。
- **cuSOLVER FP64 仿真：** 利用 INT8 Tensor Core 仿真 FP64 运算，对 QR/LU/Cholesky 等分解有显著加速。适合 B200 上 INT8 吞吐量远超 FP64 的场景。
- **Nsight Compute 新功能：** 报告聚类（Clustering）、寄存器依赖可视化、PM Sampling 改进。

### 8. Blackwell 上的关键性能准则调整

与 Hopper 相比，以下准则的权重需要调整：

**寄存器压力（准则 6）更宽松：** TMEM 承担了 MMA 累加器存储，原本占大量寄存器的 accumulator 现在不再需要寄存器。对 GEMM 类 kernel，有效可用寄存器增加了。

**Shared Memory 带宽压力降低：** 2CTA 模式下两个 SM 共享 operand，每个 SM 的 shared memory 带宽需求减半。因此对 tiling 策略的 shared memory 带宽约束放宽。

**Pipeline 设计更关键（准则 15）：** Blackwell Tensor Core 吞吐翻倍，但 shared memory 容量没变。需要更精细的 pipeline（3-4 stage）才能喂饱 Tensor Core。如果 PM Sampling 显示 Tensor pipe 利用率有波动，优先检查 pipeline 深度和 prefetch 策略。

**L2 Cache 大幅增加（126 MB）：** 更多 working set 可以留在 L2 内。对于需要跨 block 共享数据的场景（如 multi-head attention 的 KV cache），可以利用 L2 persistence 策略显式管理热数据驻留。

---

## 写在前面：为什么需要这份文档

作为 AI 助手，我在辅助 GPU kernel 开发时存在一个系统性弱点：**我掌握大量 GPU 编程原则，但在实际写代码时经常"想不起来"应用它们。** 我可能写出一个功能正确但性能糟糕的 kernel，然后在用户指出问题后才意识到违反了某条我本来知道的准则。

这份文档的目的是：

1. **在写 kernel 之前系统性地过一遍检查清单**，而不是写完再补救。
2. **将每条准则与 NCU profiling 指标关联**——如果用户反馈了 profiling 数据，我能快速定位是哪条准则被违反了。
3. **标注例外情况**——某些 kernel（如 LLM decode、reduction、通信 kernel）天然违反某些准则，不应盲目"优化"。

**使用方式：**
- **写 kernel 前：** 先阅读"Blackwell 独有特性"确认是否能利用新硬件，再过"核心准则"检查清单。
- **优化 kernel 时：** 根据 NCU 报告中的异常指标，在本文档中查找对应准则和修复方向。
- **Review kernel 时：** 逐条对照检查清单，判断是真正的性能问题还是 kernel 特性决定的合理妥协。

---

## 准则总览

| # | 准则 | 核心 NCU 指标 | 常见违规 kernel |
|---|------|-------------|---------------|
| 1 | 保证足够的并行度 | Occupancy, Wave 数 | decode attention（batch=1, seq=1） |
| 2 | 合并内存访问（Coalescing） | sectors/request, L1 hit rate | 稀疏矩阵, AoS 布局 |
| 3 | 利用 Shared Memory 减少 Global 访问 | L1/L2 hit rate, DRAM throughput | element-wise kernel（无数据复用） |
| 4 | 避免 Bank Conflict | shared memory wavefronts | 无 shared memory 的 kernel |
| 5 | 避免 Warp Divergence | branch efficiency, predicated inst | tree-based reduction |
| 6 | 合理控制寄存器压力 | register spill, occupancy | 大型 fused kernel |
| 7 | 隐藏内存延迟（提高指令级并行度） | warp stall long_scoreboard | 严格顺序依赖的算法 |
| 8 | 使用合适的数学精度和内置函数 | FP32/FP64 pipe util, 吞吐量 | 科学计算要求双精度 |
| 9 | 最小化 Host-Device 同步 | kernel 启动频率（非 NCU 指标） | 控制流密集的应用 |
| 10 | 合理使用 Tensor Core | tensor pipe utilization | 非矩阵运算 kernel |
| 11 | 优化 Grid/Block 配置 | tail effect, occupancy | 输入尺寸不可控的 dynamic shape |
| 12 | 减少 Atomic 竞争 | long_scoreboard on atomic, L2 压力 | histogram, allreduce |
| 13 | 向量化访存 | sectors/request, LSU 利用率 | 不对齐的小粒度访问 |
| 14 | 利用只读路径 | L1 hit rate, cache throughput | 频繁写入的数据结构 |
| 15 | Pipeline 化：计算与访存重叠 | SM utilization timeline 波动 | 单 stage 的 naive kernel |
| 16 | 减少同步开销 | barrier stall, SM idle time | 必须全局同步的算法 |

---

## 准则详解

### 准则 1：保证足够的并行度（Occupancy 与 Wave 数）

**原则：** GPU 靠大量线程并行来隐藏延迟。kernel 必须启动足够多的 thread block 来填满所有 SM，且每个 SM 上要有足够多的 active warp 来在一个 warp stall 时切换到另一个 warp 执行。

**量化标准：**
- 目标 Occupancy ≥ 50%（对于 memory-bound kernel 更高更好）
- Wave 数 ≥ 2（至少能填满所有 SM 两轮，避免 tail effect）
- Active warps per SM ≥ 16（经验值，具体取决于架构）

**NCU 诊断：**
- `sm__warps_active.avg.pct_of_peak_sustained_active` 低于 50% → 并行度不足
- `launch__occupancy_limit_*` 系列指标 → 定位是寄存器、shared memory 还是 block size 限制了 occupancy
- PM Sampling 时间线末尾出现漫长的利用率下降 → wave 数不足，最后一波 block 无法填满所有 SM

**修复方向：**
- 增大 grid size（更多 block）
- 减小 block size 以降低每个 block 的资源占用（但不低于 128 threads）
- 减少每线程寄存器用量（`__launch_bounds__`）
- 减少每 block shared memory 用量

**例外情况：**
- **LLM decode attention（batch=1）：** 在自回归解码中，query 只有 1 个 token，总计算量极小。无论怎么配置 grid，都无法填满所有 SM。这是算法特性决定的，正确的优化方向不是盲目增加并行度，而是：使用 split-K 将 KV sequence 维度拆分到多个 block；或者使用 persistent kernel 在一个 kernel 中处理多个 attention head。
- **Reduction kernel 最后几级：** 每级 reduce 并行度减半，最后几级天然低 occupancy。正确做法是将多级 reduce fuse 到一个 warp 内完成，避免多次 kernel launch。
- **尾部处理 kernel：** 处理 tensor 尾部不对齐元素的 kernel 天然 grid 小。可考虑合并到主 kernel 中做 masking 处理。

---

### 准则 2：合并内存访问（Memory Coalescing）

**原则：** 同一个 warp 中 32 个线程的内存访问应当落在连续的内存地址上。当 warp 访问 global memory 时，硬件将请求合并为若干 128 字节的 cache line 事务。如果 32 个线程访问 32 个分散的地址，可能需要 32 次事务而非理想的 1 次。

**量化标准：**
- `l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio` ≈ 4（理想值，128B / 32B-sector = 4 sectors for a fully coalesced 128B request）
- 如果远大于 4（如 16 或 32），说明严重不合并

**NCU 诊断：**
- sectors/request 远大于 4 → 非合并访存
- `l1tex__t_sector_hit_rate.pct` 极低 → cache 无法有效缓存分散访问
- Memory Workload Analysis 中 Rules 会直接报告 "uncoalesced" 警告
- `dram__throughput` 高但有效计算吞吐低 → 大量带宽浪费在无效数据搬运上

**修复方向：**
- **AoS → SoA 转换：** 将结构体数组转为数组结构体，使同一字段的数据在内存中连续
- **调整线程到数据的映射：** 确保 threadIdx.x 对应最内层（连续）维度
- **使用 shared memory 做转置：** 先合并地读到 shared memory，再以任意模式访问
- **padding 对齐：** 确保每行数据起始地址对齐到 128 字节

**例外情况：**
- **稀疏矩阵运算（SpMV, SpMM）：** 稀疏数据的非零元素天然不连续。这不是代码写错了，而是数据特性决定的。优化方向是选择更好的稀疏格式（如 ELL、blocked CSR）或使用 texture cache 路径。
- **Tree/Graph 遍历：** 指针追踪式的访存模式天然不合并。优化方向是 BFS 式遍历 + 排序以提高局部性。
- **Gather/Scatter 操作：** 根据索引数组间接访存。如果索引随机分布，无法合并。考虑对索引排序后再访问。

---

### 准则 3：利用 Shared Memory 减少 Global Memory 访问

**原则：** Shared memory 是 SM 上的片上高速存储（带宽约 global memory 的 10-20 倍，延迟约 20-30 cycles vs global 的 200-800 cycles）。当多个线程需要访问相同或相邻的数据时，应先将数据从 global memory 加载到 shared memory，然后从 shared memory 重复读取。

**量化标准：**
- 数据复用率 > 1（同一数据被多个线程或多次迭代使用）时应使用 shared memory
- `dram__throughput.avg.pct_of_peak_sustained_elapsed` 接近峰值且 SM throughput 不高 → memory-bound，需要减少 DRAM 访问

**NCU 诊断：**
- `dram__throughput` 高 + `sm__throughput` 低 → kernel 被 memory bandwidth 限制
- `l1tex__t_sector_hit_rate.pct` 低 → 数据没有被有效复用/缓存
- `smsp__inst_executed_op_shared_ld.sum` = 0 且 `smsp__inst_executed_op_global_ld.sum` 很大 → 完全没有使用 shared memory
- Warp stall reason 以 `long_scoreboard` 为主 → 大量时间在等待 global memory 返回数据

**修复方向：**
- **Tiling：** 将输入矩阵分块加载到 shared memory（经典的 GEMM tiling）
- **数据预取：** 在计算当前 tile 时异步加载下一个 tile（double buffering）
- **减少冗余 load：** 将多个线程共用的 scalar 值加载到 shared memory（或 `__constant__` memory）

**例外情况：**
- **Element-wise kernel（如 ReLU, Add, Scale）：** 每个元素只被访问一次，没有数据复用。使用 shared memory 反而增加延迟和代码复杂度。这类 kernel 天然是 memory-bound，正确优化方向是 kernel fusion（把多个 element-wise 操作合并到一个 kernel 中减少 global memory 读写次数）。
- **超大 working set：** 如果每个 block 需要的数据量超过 shared memory 容量（B200 上通常是 228 KB configurable），则无法完全放入。需要 multi-stage tiling。
- **Occupancy 受限时：** 增大 shared memory 用量会降低每 SM 可容纳的 block 数。需要在 shared memory 利用与 occupancy 之间权衡。

---

### 准则 4：避免 Shared Memory Bank Conflict

**原则：** Shared memory 由 32 个 bank 组成，连续 4 字节分别映射到连续 bank。同一 warp 中如果多个线程访问同一 bank 的不同地址，则这些访问会被串行化（bank conflict）。N-way bank conflict 导致该访问耗时变为 N 倍。

**量化标准：**
- `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum` / `l1tex__data_pipe_lsu_cycles_active_mem_shared_op_ld.sum` → 每 cycle 完成的 shared memory wavefronts，低则说明有 conflict
- 理想情况下每次 shared memory 请求 = 1 wavefront（无 conflict）

**NCU 诊断：**
- Shared memory wavefronts per request > 1 → 存在 bank conflict
- Memory Workload Analysis 中 shared memory 部分显示高 wavefront count
- 如果 shared memory 访问量不大，但 `stall_barrier` 或 `stall_short_scoreboard` 高 → 可能是 bank conflict 导致延迟增大

**修复方向：**
- **Padding：** 在 shared memory 数组每行末尾加 1 个 padding 元素，打破规律性 bank mapping。例如 `__shared__ float tile[32][33];`（33 而非 32）
- **Swizzle 访问模式：** 对索引进行异或变换来打散 bank 映射
- **调整数据布局：** 确保同一 warp 中的线程访问不同 bank

**例外情况：**
- **Broadcast 访问：** 如果一个 warp 中所有线程访问 shared memory 的**同一地址**，硬件会自动 broadcast，不产生 conflict。这是合法的。
- **不使用 shared memory 的 kernel：** 此准则不适用。
- **低 shared memory 访问频率：** 如果 shared memory 只在 kernel 开头/结尾少量访问，即使有 conflict，对整体性能影响也很小。优先关注其他瓶颈。

---

### 准则 5：避免 Warp Divergence

**原则：** GPU 以 warp（32 线程）为单位执行指令。当一个 warp 内的线程走入不同的分支时，硬件必须依次执行每个分支路径（当前架构使用 predication），使非活跃线程等待。严重 divergence 会导致 SIMT 效率降低到 1/32。

**量化标准：**
- Branch efficiency（活跃线程比例）≥ 90%
- `smsp__thread_inst_executed_per_inst_executed.ratio` 接近 32 → 几乎没有 predicated-off 线程

**NCU 诊断：**
- `smsp__thread_inst_executed_per_inst_executed.ratio` 远低于 32 → warp 内活跃线程少
- Source page 中某些分支指令的 `predicated-off thread %` 很高
- `sm__inst_executed` vs `sm__inst_issued`：如果 issued 远大于 executed → 大量 predicated 指令

**修复方向：**
- **数据重排：** 让需要执行相同分支的元素被相邻线程处理（如排序后再处理）
- **条件改为 mask 计算：** 用 `cond * valueA + (1-cond) * valueB` 代替 if-else
- **将 divergent 逻辑移到 warp 粒度：** 让整个 warp 走同一分支（每 32 个元素做一次判断）

**例外情况：**
- **Tree-based reduction 最后几级：** 在 warp 内做 reduce 时，每一级活跃线程数减半。这是算法固有的，且 warp-level reduction 总共只有 5 级（32→16→8→4→2→1），开销很小。可以使用 `__shfl_down_sync()` 优化。
- **Boundary/mask 处理：** kernel 处理 tensor 边缘时，部分线程可能超出范围需要 mask 掉。这通常只影响最后几个 warp，不必过度优化。
- **Dynamic shape kernel：** 不同 batch element 长度不同（如 NLP 中变长序列），divergence 难以完全消除。Padding + packing 是常用缓解策略。

---

### 准则 6：合理控制寄存器压力

**原则：** 每个 SM 的寄存器文件容量有限（B200 上每 SM 65536 个 32-bit 寄存器）。每线程使用的寄存器越多，能同时驻留的 warp 就越少（降低 occupancy）。当寄存器不够时，编译器会将变量溢出到 local memory（register spill），而 local memory 实际在 DRAM 上，延迟极高。

**量化标准：**
- `launch__registers_per_thread` ≤ 128（B200 上 128 寄存器 × 512 线程 = 65536，刚好用完一个 SM 的 RF）
- `smsp__inst_executed_op_local_ld.sum` = 0 且 `smsp__inst_executed_op_local_st.sum` = 0 → 无 spill
- Register spill 不是二元的：少量 spill 如果被 L1 缓存住，影响可能很小

**NCU 诊断：**
- `local_ld/local_st` > 0 → 发生了 register spill
- `launch__registers_per_thread` 很高 + `sm__warps_active` 低 → 寄存器限制了 occupancy
- `launch__occupancy_limit_registers` 是 occupancy 的最大限制因素 → 需要减少寄存器
- `l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum` 大 → 大量 local memory load

**修复方向：**
- **使用 `__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)`：** 告诉编译器目标 occupancy，让编译器更积极地限制寄存器分配
- **拆分大 kernel：** 一个过于复杂的 fused kernel 可能因寄存器爆炸而适得其反，拆成两个 kernel 可能总体更快
- **减少中间变量：** 重新计算代替存储某些中间结果
- **使用 shared memory 替代部分寄存器存储：** 对于线程私有但数组化的数据

**例外情况：**
- **大型 fused kernel（如 FlashAttention）：** 这类 kernel 通过 fusion 减少了大量 global memory 读写，但代价是寄存器压力高。fusion 带来的 memory 节省通常远超 occupancy 降低的损失。只要没有严重的 register spill，可以接受较低的 occupancy。
- **Compute-bound kernel：** 如果 kernel 是纯计算密集型（SM throughput 接近 100%），低 occupancy 未必是问题——已经不需要更多 warp 来隐藏延迟了。
- **显式优化的 kernel：** 手写汇编（CUTLASS level）的 kernel 往往精确控制寄存器分配，occupancy 是有意设计的，不应盲目调整。

---

### 准则 7：隐藏内存延迟（提高 ILP 和 TLP）

**原则：** Global memory 延迟高达 200-800 cycles。GPU 通过两种方式隐藏延迟：线程级并行（TLP，多个 warp 交替执行）和指令级并行（ILP，同一线程内多条独立指令同时 in-flight）。如果每次都 load 一个数据 → 等待 → 计算 → 再 load 下一个，pipeline 就是空的。

**量化标准：**
- Warp stall reason 中 `long_scoreboard` 不应占绝对主导
- `sm__inst_executed.avg.per_cycle_active`（IPC）应接近理论值
- Scheduler 的 eligible warp 数不应长期为 0

**NCU 诊断：**
- `smsp__pcsamp_warps_issue_stalled_long_scoreboard` 占比 > 40% → 大量等待 memory 返回
- `smsp__warps_eligible.avg.per_cycle_active` 接近 0 → 没有可发射的 warp
- SchedulerStats 中 "no eligible" 比例高 → 所有 warp 都在等待
- Source page 中 load 指令后的第一条依赖指令 stall 最严重

**修复方向：**
- **Software pipelining / double buffering：** 在计算当前 tile 时预取下一个 tile 的数据
- **增加每线程处理的数据量：** 让每个线程同时发起多个 load，增加 in-flight 请求
- **使用 `cp.async`（CUDA 11+）：** 异步将 global memory 数据拷贝到 shared memory，不占用寄存器和 ALU
- **提高 occupancy：** 更多活跃 warp = 更多 TLP（但受准则 6 制约）

**例外情况：**
- **严格顺序依赖的算法：** 如串行扫描（prefix sum 的串行部分），下一步计算依赖上一步结果，无法重叠。优化方向是算法层面改用并行版本（如 Blelloch scan）。
- **Pointer-chasing 访问：** 链表遍历、树查询等，下一次 load 的地址取决于上一次 load 的结果。GPU 天然不擅长这类工作。

---

### 准则 8：使用合适的数学精度和内置函数

**原则：** FP64 吞吐量通常是 FP32 的 1/2 到 1/64（取决于架构）。FP32 的超越函数（sin, cos, exp）比 native 近似版本（__sinf, __cosf, __expf）慢 10 倍以上。在精度允许的情况下，应使用低精度和快速函数。

**量化标准：**
- 确认使用了正确的精度：FP16/BF16 用于训练/推理、FP32 用于累加、FP64 仅在必要时使用
- `sm__pipe_fp64_cycles_active` 接近 0（除非确实需要双精度）

**NCU 诊断：**
- InstructionStats 中 FP64 指令占比高 → 检查是否有意外的双精度运算（常见原因：字面量 `1.0` 默认是 double，应写 `1.0f`）
- SFU（Special Function Unit）利用率高 → 大量超越函数调用，考虑用近似版本
- ComputeWorkloadAnalysis 中各 pipeline 利用率不均 → 某个 pipeline 成为瓶颈

**修复方向：**
- **所有浮点字面量加 `f` 后缀：** `1.0f` 不是 `1.0`，`0.5f` 不是 `0.5`
- **使用 `__fdividef(x, y)` 代替 `x / y`**
- **使用 `__expf()`, `__logf()`, `__sinf()` 代替 `expf()`, `logf()`, `sinf()`**
- **混合精度：** FP16 计算 + FP32 累加（Tensor Core 原生支持这种模式）
- **编译选项 `--use_fast_math`：** 自动替换为快速函数（但会牺牲精度和特殊值处理）

**例外情况：**
- **科学计算、金融计算：** 某些领域对精度有严格要求，必须使用 FP64。此时优化方向是算法层面减少运算量，而非降低精度。
- **数值稳定性关键路径：** 如 softmax 中的 log-sum-exp，累加器必须用 FP32 即使输入输出是 FP16。

---

### 准则 9：最小化 Host-Device 同步

**原则：** 每次 `cudaDeviceSynchronize()` 或隐式同步（如 `cudaMemcpy` D2H）都会导致 CPU 等待 GPU 完成所有工作，GPU 也可能等待 CPU 提交新工作，形成 pipeline bubble。GPU 计算能力在同步等待期间被浪费。

**量化标准：**
- 这不是单 kernel 级别的问题，NCU 不直接检测。应使用 Nsight Systems（nsys）查看 timeline。
- 如果多个短 kernel 之间有大段 CPU 空闲 → 同步过多

**NCU 相关诊断：**
- 如果 profiling 的 kernel 极短（< 10μs） → 可能是同步导致无法合并多个 kernel
- `gpu__time_duration.sum` 很小但用户报告端到端延迟高 → kernel 间有 CPU 同步开销

**修复方向：**
- **使用 CUDA Stream 实现异步执行：** kernel launch、memcpy 提交到 stream 后立即返回
- **使用 `cudaMemcpyAsync` 代替 `cudaMemcpy`**
- **使用 CUDA Events 代替 `cudaDeviceSynchronize()`：** 只同步特定 stream
- **使用 CUDA Graph：** 将一系列 kernel 和 memcpy 录制为 graph，一次提交减少 launch overhead
- **Host 端用 pinned memory（`cudaMallocHost`）：** 否则 `cudaMemcpyAsync` 仍然会同步

**例外情况：**
- **调试阶段：** 同步便于定位错误，正式版再移除。
- **必须等待 GPU 结果的控制流：** 如果 CPU 需要 GPU 计算的结果来决定下一步逻辑（如动态 shape），同步不可避免。可以用 device-side launch（CUDA Dynamic Parallelism）或 conditional graph node 来规避。

---

### 准则 10：合理使用 Tensor Core

**原则：** 现代 GPU 的 Tensor Core 吞吐量是 CUDA Core 的数倍到数十倍。矩阵乘法相关的 kernel 应尽量走 Tensor Core 路径（WMMA/MMA 指令）。但 Tensor Core 对数据布局和尺寸有严格要求。

**量化标准：**
- 对于 GEMM 类 kernel：`sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` 应 > 50%
- 如果 FP16/BF16 GEMM 但 tensor pipe = 0% → 没有走 Tensor Core

**NCU 诊断：**
- `sm__inst_executed_pipe_tensor.sum` = 0 → 完全没有 Tensor Core 指令
- Tensor pipe 利用率低 + DRAM throughput 高 → Tensor Core 在等待数据
- InstructionStats 中有大量 FFMA（FP32 FMA）而非 HMMA/DMMA → 落回 CUDA Core 路径

**修复方向：**
- **使用 cuBLAS / CUTLASS / cuDNN：** 这些库已针对 Tensor Core 优化
- **确保矩阵维度对齐：** M, N, K 对齐到 16（FP16）或 8（TF32）的倍数
- **确保数据布局正确：** Tensor Core 偏好特定 layout（如 column-major 或 interleaved）
- **使用 WMMA API 或 PTX mma 指令：** 手写 kernel 时显式调用

**例外情况：**
- **非矩阵运算：** element-wise、reduction、sort 等 kernel 无法使用 Tensor Core，这是正常的。
- **小矩阵：** 当 M, N, K 很小时，Tensor Core tile 大小无法充分利用，可能不比 CUDA Core 快。
- **精度不匹配：** Tensor Core 支持特定精度组合（FP16, BF16, TF32, INT8, FP8 等）。如果需要 FP64 精度，多数 GPU 的 FP64 Tensor Core 吞吐量很有限。

---

### 准则 11：优化 Grid/Block 配置

**原则：** Block size 影响 occupancy、warp 数、shared memory 分配粒度。Grid size 决定总并行度和 SM 填充率。配置不当会导致 SM 利用不均（partial wave）或 occupancy 下降。

**量化标准：**
- Block size 应为 32 的倍数（warp 对齐），通常 128 或 256 是好的起点
- Grid size 应为 SM 数量 × 每 SM block 数 的倍数（避免 partial wave）
- 总 thread 数 ≥ GPU 全部 SM 能承载的最大 thread 数 × 2

**NCU 诊断：**
- PM Sampling 时间线末尾的 tail 占总时间比例 > 20% → grid size 不够或不是 wave size 的倍数
- `sm__warps_active.avg.pct_of_peak_sustained_active` 低但 `launch__occupancy_limit_*` 无明显限制 → grid 太小
- LaunchStats 中 block size 不是 32 的倍数 → 最后一个 warp 有浪费

**修复方向：**
- **Block size 选择 128 或 256：** 在大多数架构上能平衡 occupancy 和资源利用
- **Grid size 向上取整到 wave size 的倍数：** `wave_size = blocks_per_sm × num_sms`
- **如果 grid 太小，考虑 persistent kernel：** 每个 block 用循环处理多个工作单元
- **使用 cudaOccupancyMaxPotentialBlockSize()：** 自动计算最优 block size

**例外情况：**
- **Dynamic shape 场景：** 输入尺寸在推理时才确定，无法预先优化 grid 配置。解决方案是使用 persistent kernel + work-stealing 模式。
- **固定功能 kernel（如 softmax over 词表）：** grid 大小 = batch_size，如果 batch 小则 grid 天然小。考虑将 batch 维度和序列维度同时并行化。

---

### 准则 12：减少 Atomic 操作竞争

**原则：** Atomic 操作（atomicAdd, atomicCAS 等）在多个线程同时访问同一地址时会被串行化。如果大量线程 atomic 到少量地址，竞争会严重降低性能。

**量化标准：**
- Atomic 操作不应成为主要 stall 原因
- 如果必须使用 atomic，尽量分散到多个地址（如分 bucket 再最终 reduce）

**NCU 诊断：**
- `long_scoreboard` stall 在 atomic 指令上集中 → atomic 竞争
- `lts__t_sectors_op_atom.sum` 或 `lts__t_sectors_op_red.sum` 很大 → 大量 atomic 操作到 L2
- L2 throughput 高但有效带宽低 → atomic 重试浪费带宽

**修复方向：**
- **Hierarchical reduction：** 先 warp 内 reduce（`__shfl_down_sync`），再 block 内 reduce（shared memory），最后 block 间 atomic → atomic 次数从 N 降到 gridDim.x
- **分 bucket：** 先写入 per-thread 或 per-block 的 local buffer，最后合并
- **使用 CAS loop 代替 atomicAdd（对 FP16 等不支持原生 atomic 的类型）**

**例外情况：**
- **Histogram：** 天然需要大量 atomic。优化方向是 shared memory histogram + 最终 global atomic merge，以及尽量让不同线程落到不同 bin。
- **AllReduce 通信 kernel：** 分布式训练中的 NCCL kernel 大量使用 atomic 做跨 GPU 同步，这是算法本质需要。

---

### 准则 13：向量化访存

**原则：** GPU 的 load/store 单元单次可搬运 1/2/4/8/16 字节。使用向量类型（float2, float4, int4 等）可以单条指令加载更多数据，减少总指令数和 LSU 压力。

**量化标准：**
- 对于 memory-bound kernel，使用向量化 load 可以减少 load 指令数 2-4 倍
- `sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed` 高 → LSU 是瓶颈，向量化可缓解

**NCU 诊断：**
- LSU pipe 利用率高接近峰值 → load/store 指令太多
- InstructionStats 中 LDG/STG 指令占比大且数量多
- Source page 中 load 指令密集

**修复方向：**
- **使用 `float4` / `int4` 类型读写：** `float4 val = *reinterpret_cast<float4*>(ptr + idx);`
- **确保地址对齐：** `float4` load 需要 16 字节对齐
- **处理尾部不对齐元素：** 最后不足 vector width 的元素单独处理

**例外情况：**
- **非对齐访问：** 如果数据地址不能保证对齐到 vector size，强制向量化 load 会导致未定义行为或性能退化。
- **小 tensor：** 如果 tensor 元素数量很少，向量化可能导致越界。需要做边界检查。
- **已经 compute-bound 的 kernel：** 如果 kernel 瓶颈在计算而非访存，向量化收益有限。

---

### 准则 14：利用只读缓存路径

**原则：** GPU 有专门的只读数据路径（texture cache / `__ldg()` intrinsic / `const __restrict__` 修饰）。只读数据可以走更高效的缓存路径，避免与读写数据竞争 L1 cache 容量。

**量化标准：**
- 对于只读输入数据，确保使用 `const __restrict__` 指针修饰
- CUDA 编译器通常能自动检测只读访问并使用 LDG（global load through texture path）

**NCU 诊断：**
- InstructionStats 中 LDG（只读 load）vs LD（普通 load）的比例 → LDG 越多越好
- `l1tex__t_sector_hit_rate.pct` 低但数据应该有局部性 → 可能只读数据未走 texture cache

**修复方向：**
- **指针声明加 `const __restrict__`：** `__global__ void kernel(const float* __restrict__ input, float* __restrict__ output)`
- **显式使用 `__ldg(&ptr[idx])`：** 强制走只读路径
- **使用 `__constant__` memory：** 对于小量广播数据（< 64KB）

**例外情况：**
- 现代编译器（CUDA 11+）在大多数情况下能自动优化，显式 `__ldg` 通常不必要。但在复杂的指针别名场景下，编译器可能保守处理。
- 如果数据既读又写，不能使用只读路径。

---

### 准则 15：Pipeline 化——计算与访存重叠

**原则：** 高性能 kernel 应当将执行流程拆分为多个 stage（如 load → compute → store），并使用 double buffering 或 multi-stage pipeline 使不同 stage 的不同数据块重叠执行。这样 Tensor Core 在计算当前块时，LSU 同时加载下一块数据。

**量化标准：**
- PM Sampling 时间线应该平坦（无周期性波动）
- SM throughput 和 DRAM throughput 应该同时保持高位

**NCU 诊断：**
- PM Sampling 时间线呈锯齿状（SM throughput 和 DRAM throughput 交替高低） → 计算和访存没有重叠
- `smsp__pcsamp_warps_issue_stalled_long_scoreboard` 高 → 计算在等待数据
- SM throughput 时间线有周期性谷底 → pipeline bubble

**修复方向：**
- **Double buffering：** 两份 shared memory buffer 交替使用
  ```cuda
  // 伪代码
  load(smem_buf[0], global_ptr + 0);
  for (int i = 0; i < num_tiles - 1; i++) {
      load(smem_buf[(i+1)%2], global_ptr + (i+1)*tile_size);  // prefetch next
      __syncthreads();
      compute(smem_buf[i%2]);  // compute current
      __syncthreads();
  }
  compute(smem_buf[(num_tiles-1)%2]);  // last tile
  ```
- **使用 `cp.async` + `cp.async.commit_group` + `cp.async.wait_group`：** 异步拷贝不需要寄存器中转
- **Multi-stage pipeline（如 CUTLASS 的 3-stage/4-stage）：** 更深的 pipeline 能更好地隐藏延迟，但需要更多 shared memory

**例外情况：**
- **Compute-bound kernel：** 如果计算时间远大于访存时间，单 buffer 就够了——访存总是能在计算完成前返回。
- **极短 kernel：** tile 数量 < 3 时，pipeline 带来的代码复杂度不值得。
- **大量 shared memory 占用：** double buffering 使 shared memory 需求翻倍，可能严重降低 occupancy。需要权衡。

---

### 准则 16：减少同步开销

**原则：** `__syncthreads()` 强制 block 内所有线程到达屏障后才继续。如果某些 warp 运行较快，它们必须等待最慢的 warp。过多的同步点会产生 bubble。同样，grid-level 同步（cooperative groups 或 multi-kernel）开销更大。

**量化标准：**
- `smsp__pcsamp_warps_issue_stalled_barrier` 不应占主导
- 同步点之间的计算量应足够大以分摊同步开销

**NCU 诊断：**
- `barrier` stall 占比 > 30% → 同步等待时间过长
- Source page 中 `BAR.SYNC` 指令 stall 采样很高
- 如果 `barrier` stall 高且 `warp_divergence` 也高 → divergence 导致同步时等待时间变长

**修复方向：**
- **减少 `__syncthreads()` 次数：** 合并多个需要同步的阶段
- **使用 warp-level primitives 代替 block-level 同步：** `__shfl_sync`, `__ballot_sync` 等在 warp 内不需要 barrier
- **Warp-specialized execution：** 不同 warp 做不同工作（如 CUTLASS 中 producer/consumer warp），减少不必要的全 block 同步
- **使用 `__syncwarp()` 代替 `__syncthreads()`：** 当只需要 warp 内同步时

**例外情况：**
- **需要全局 reduce 或 scan 的算法：** block 间同步不可避免。可以用 cooperative groups 的 grid-wide sync 或拆分为多个 kernel。
- **正确性需要：** 如果不同步会导致 race condition，那同步是必须的。绝不能为了性能牺牲正确性。

---

## 特殊 Kernel 类型速查

下表总结了特定类型 kernel 天然会违背哪些准则，以及针对性的优化方向。在分析这些 kernel 时，相关准则的违反应当降低优先级。

### LLM Decode Attention（单 token 生成）

| 天然违反 | 原因 | 替代优化方向 |
|---------|------|------------|
| 准则 1（低并行度） | query 只有 1 个 token，计算量极小 | Split-K 沿 KV seq_len 维度拆分；multi-query/GQA 减少 head 数 |
| 准则 11（grid 太小） | batch_size × num_heads 可能不足以填满 SM | Persistent kernel；合并多个 head 到一个 block |
| 准则 15（无法 pipeline） | 数据量太小，没有足够 tile 做 pipeline | 专注减少 kernel launch overhead；CUDA Graph |

### Element-wise Kernel（ReLU, Add, Scale, Cast）

| 天然违反 | 原因 | 替代优化方向 |
|---------|------|------------|
| 准则 3（不用 shared memory） | 每个元素只访问一次，无数据复用 | 正常——不需要 shared memory |
| 准则 10（不用 Tensor Core） | 非矩阵运算 | 正常——不适用 |

优化重点应在：kernel fusion（减少 launch 次数和 global memory 读写次数）、向量化 load（float4）、确保 coalescing。

### Reduction（Sum, Max, Softmax）

| 天然违反 | 原因 | 替代优化方向 |
|---------|------|------------|
| 准则 5（warp divergence） | Tree reduction 每级活跃线程减半 | Warp shuffle 替代 shared memory reduction |
| 准则 1（后期低并行度） | 最后几级只剩少量线程 | 两阶段：parallel reduce → single block final reduce |
| 准则 12（atomic 竞争） | Block 间 reduce 需要 atomic | 尽量在 block 内完成 reduce，减少 atomic 次数 |

### GEMM / Matmul

优化重点：应当是所有准则中表现最好的 kernel 类型。主要关注：Tensor Core 利用率（准则 10）、shared memory tiling（准则 3）、pipeline（准则 15）、避免 bank conflict（准则 4）。

如果 GEMM kernel 的 Tensor Core 利用率低于 60%，几乎一定有优化空间。

### Scatter / Gather / Embedding Lookup

| 天然违反 | 原因 | 替代优化方向 |
|---------|------|------------|
| 准则 2（非合并访存） | 索引驱动的随机访问 | 对索引排序后再访问；使用 texture cache |
| 准则 7（无法隐藏延迟） | 地址依赖前一次 load 结果 | 增大每线程处理的 element 数 |

### 通信 Kernel（NCCL AllReduce, P2P）

| 天然违反 | 原因 | 替代优化方向 |
|---------|------|------------|
| 准则 12（atomic 竞争） | 跨 GPU 同步本质需要 atomic | NCCL 已高度优化，不建议手动改 |
| 准则 16（同步开销） | 多 GPU 同步不可避免 | 计算与通信重叠（overlap） |

---

## 写 Kernel 前的检查清单

在开始编写或优化一个 CUDA kernel 之前，逐项过一遍：

### Blackwell 专项（B200 / sm_100a）
0. **编译目标正确吗？** 是否使用了 `-arch=sm_100a` 而非旧架构？
0. **能用 TMEM 吗？** 如果是 GEMM/MMA 类 kernel，累加器应在 TMEM 中而非寄存器。
0. **能用 2CTA 吗？** 大型 GEMM 应启用 CTA pair（`cta_group::2`）以达到 100% Tensor Core 利用率。
0. **数据类型选对了吗？** LLM 推理考虑 NVFP4/FP8，训练考虑 BF16+FP8 混合。FP4 吞吐量是 FP16 的 4 倍。
0. **能用 cuTile 吗？** CUDA 13.2 的 cuTile 支持 Blackwell，可自动生成 tcgen05+TMA pipeline 代码。
0. **L2 persistence 有用吗？** B200 有 126 MB L2，对于热数据（如 KV cache）可以用 L2 persistence 策略驻留。

### 通用准则
1. **我需要多少线程？** Grid 能否填满所有 148 SM ≥ 2 个 wave？（准则 1, 11）
2. **内存访问模式是什么？** warp 内线程是否访问连续地址？（准则 2）
3. **有数据复用吗？** 如果有，是否规划了 shared memory tiling？（准则 3）
4. **Shared memory 访问会有 bank conflict 吗？** 是否需要 padding？（准则 4）
5. **有分支吗？** 分支条件是否以 warp 为粒度一致？（准则 5）
6. **寄存器够用吗？** 是否设置了 `__launch_bounds__`？TMEM 是否分担了累加器？（准则 6）
7. **计算和访存能重叠吗？** 是否设计了 double buffering 或更深 pipeline？B200 需要 3-4 stage 才能喂饱 Tensor Core。（准则 7, 15）
8. **精度选对了吗？** 字面量是否都加了 `f` 后缀？能否用 NVFP4/FP8 提升吞吐？（准则 8）
9. **有不必要的同步吗？** 能否用 warp-level 操作替代 block-level 同步？（准则 16）
10. **能用 Tensor Core 吗？** 维度是否对齐？数据类型是否匹配？B200 的 tcgen05 是否适用？（准则 10）
11. **Load 能向量化吗？** 地址是否对齐到 8/16 字节？（准则 13）
12. **指针标注了 `const __restrict__` 吗？**（准则 14）
