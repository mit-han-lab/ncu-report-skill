# Six Analysis Dimensions

Every kernel profile report is ambiguous until you look at it through specific lenses. These six dimensions are the ones that consistently matter. Walk through all six; don't stop at the first finding.

For each dimension this doc describes:

- **What you're answering**
- **Which metrics to read** (B200 / sm_100 names)
- **How to read them** (what's "normal", what's "bad")
- **Which `helpers/` to run**

---

## Dimension 1 — SM occupancy & launch geometry

**What:** is the grid large enough to fill the GPU? Is occupancy being limited by registers, shared memory, or block-size constraints?

**Metrics:**
```
launch__grid_size
launch__block_size
launch__grid_dim_x / _y / _z
launch__waves_per_multiprocessor
launch__registers_per_thread
launch__shared_mem_per_block
launch__occupancy_limit_blocks
launch__occupancy_limit_registers
launch__occupancy_limit_shared_mem
launch__occupancy_limit_warps
device__attribute_multiprocessor_count    # 132 on H100/H200/H800 · 148 on B200 · 160 on B300
sm__maximum_warps_per_active_cycle_pct    # theoretical occupancy %
sm__warps_active.avg.pct_of_peak_sustained_active  # achieved occupancy %
```

**Reading:**

- **Waves / SM < 1**: grid is too small to fill the chip. With fewer blocks than `SM_count × blocks_per_SM`, some SMs sit idle the entire time. `Est. Speedup` from NCU often hits 50-90% here. SM counts: 132 (H200/H800), 148 (B200), 160 (B300).
- **Waves / SM in [1, 2)**: you have a tail wave (partial last wave). Tail effect magnitude is roughly `(last_wave_blocks / wave_size) × (block_exec_time / total_kernel_time)`.
- **Waves / SM > 4**: grid is plenty big, scheduling averages out.
- **Theoretical occupancy 100% but achieved << 100%**: stalls are the bottleneck, not launch config. Move to Dimension 3.

- **Theoretical occupancy < 100% and `launch__occupancy_limit_registers` is the tightest**: reduce register usage or add `__launch_bounds__`.
- **`launch__occupancy_limit_shared_mem`** the tightest: shared mem / block is too large, reduce tile size.

**Derived: wave math**

```python
blocks_per_sm = min(occ_limit_blocks, occ_limit_registers, occ_limit_shared_mem, occ_limit_warps)
wave_size = blocks_per_sm * num_sms
num_waves = (total_blocks + wave_size - 1) // wave_size
last_wave_blocks = total_blocks - (num_waves - 1) * wave_size
last_wave_utilization_pct = last_wave_blocks / wave_size * 100
```

**Helper:** `analyze_reports.py` prints all the key launch metrics under "Launch geometry" in the output txt.

---

## Dimension 2 — Thread-block balance (tail effect)

**What:** are blocks finishing at roughly the same time, or do a few outliers drag out the kernel?

**Metrics:**

There is no single "imbalance" metric — use these signals together:

```
# Per-SM active-cycle distribution (from MemoryWorkloadDistribution section)
# These show as "max XX% above average, min YY% below average" in details page
sm__cycles_active.{avg,max,min,sum}       # via action.source_info or the details page

# PM sampling (time series) — the shape matters, not just the mean
pmsampling:smsp__warps_issue_stalled_long_scoreboard.avg
pmsampling:smsp__warps_issue_stalled_short_scoreboard.avg
pmsampling:smsp__warps_issue_stalled_wait.avg
```

**Reading:**

- NCU's `details` page already says it: `"One or more SMs have a much lower number of active cycles than the average. Maximum instance value is X% above, while the minimum is Y% below."` X=51% / Y=95% (seen in practice) means severe imbalance.
- Render the PM-sampling time series (use `plot_timeline.py`). Possible shapes:
  - **Flat high → clean drop**: ideal. Well-balanced, good SM fill.
  - **Flat high → gradual tail**: tail effect. The tail's length is how much time a few slow blocks waste. Usually caused by variable-length inputs (e.g. seq_len varies per batch element).
  - **Flat low**: grid is too small (Dimension 1).
  - **Periodic waves / sawtooth**: pipeline bubbles — compute and memory alternate, nothing overlaps.

**Where imbalance typically comes from:**

1. **Variable-length per-CTA work**: when each CTA's iteration count depends on an input axis (e.g., per-element lengths driven by a prefix-sum / cumulative-length array), CTAs take very different times.
2. **Branch-and-early-exit inside the kernel**: some blocks bail early via `return`, others don't.
3. **Work-stealing without proper load balancing**: custom scheduling logic that happens to assign heavy work to a few blocks.

**Fix direction:** chunk the variable-length work (e.g., time-chunking for sequence-style workloads), or oversubscribe with work-stealing.

**Helper:** `plot_timeline.py` produces ASCII timeline plots. If you see a gradual slope on the right side, that's your tail effect.

Additionally, **always inspect the input distribution**. If per-CTA work is driven by an array like `per_element_lengths`:
```python
# Example: given a per-CTA work-count array, compute imbalance ratios
work_per_cta = [...]  # derive this from whatever drives the inner loop count
avg = sum(work_per_cta) / len(work_per_cta)
print(f"max/avg = {max(work_per_cta)/avg:.2f}x, max/min = {max(work_per_cta)/min(work_per_cta):.2f}x")
```

Ratios > 5x indicate significant potential for tail effect.

---

## Dimension 3 — Stall reason breakdown + per-line hotspots

**What:** when warps aren't issuing, what are they waiting for? Which source lines generate the most stalls?

**Aggregate stall metrics (SOL-adjacent, aggregated over the kernel):**

> **Metric name note:** on 🔵 Hopper (sm_90) the suffix is `.pct` and there is no `average_` prefix: `smsp__warps_issue_stalled_long_scoreboard_per_issue_active.pct`. On 🟠 Blackwell (sm_100/sm_103) the suffix is `.ratio` with `average_` prefix as shown below. See [`08-gpu-metric-names.md`](08-gpu-metric-names.md).

```
# Ratio per issued warp — how many of 16 active warps are in each stall state
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio
smsp__average_warps_issue_stalled_wait_per_issue_active.ratio
smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio
smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio
smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio
smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio
smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio
smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio
smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio
```

**Per-line stall metrics (from `--set source`):**
```
smsp__pcsamp_sample_count                                  # total samples
smsp__pcsamp_warps_issue_stalled_long_scoreboard           # per-PC counts
smsp__pcsamp_warps_issue_stalled_short_scoreboard
smsp__pcsamp_warps_issue_stalled_wait
smsp__pcsamp_warps_issue_stalled_selected                  # productive cycles
...
```

**Stall reasons you need to know:**

| Reason | Meaning | Typical cause | Fix direction |
|---|---|---|---|
| `long_scoreboard` | waiting on long-latency dep | global memory load hasn't returned | coalesce, reuse, add ILP |
| `short_scoreboard` | waiting on short-latency dep | shared/local memory, or compute chain | add ILP, shorten dep chains |
| `wait` | waiting on fixed-latency pipe | SFU / tensor-core output | more independent ops in flight |
| `barrier` | `__syncthreads` / mbarrier wait | other threads haven't arrived | reduce syncs, fix divergence |
| `membar` | memory fence | `__threadfence` | avoid if possible |
| `math_pipe_throttle` | FMA pipe saturated | legit compute-bound | you're doing well, find other wins |
| `mio_throttle` / `lg_throttle` / `tex_throttle` | LSU/LD-ST/TEX pipe saturated | too many load/store insns | vectorize, use shared mem |
| `not_selected` | eligible but scheduler picked another | **good sign** — plenty of parallelism | ignore |
| `selected` | actually issuing this cycle | **productive** | ignore |
| `dispatch_stall` | dispatch unit busy | rare | usually minor |
| `no_instruction` | warp has nothing to issue | kernel prologue/epilogue | usually minor |
| `drain` | warp finishing last few instructions | end of kernel | ignore |
| `branch_resolving` | branch target calc in progress | tight branches | usually minor |

**Reading the ratio metric:** a value of e.g. `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio = R` means: for every cycle in which a warp issues, R other warps (on that SM sub-partition, per cycle-with-an-issue) were stalled on `long_scoreboard`. Higher values = more stalled time. On a kernel with 16 active warps per scheduler, R is bounded above by ~15 (all other warps stalled on this reason).

**Reading the `pcsamp` percentages:** normalize by `smsp__pcsamp_sample_count` to get "% of samples stalled on X". Rules of thumb:

- **`long_scoreboard` > 40% of samples**: kernel is memory-latency-bound. Check Dimension 6 (access patterns) next.
- **`short_scoreboard` > 30%**: check for long dep chains or heavy shared-memory use.
- **`barrier` > 20%**: too much synchronization, or warp divergence before a barrier.
- **`selected` < 10%**: very little actual issue — the whole kernel is stall-bound.

**Helper:** `extract_stall_hotspots.py` produces `stall_hotspots_<tag>.txt` which ranks source lines by total stall samples. This directly points at the offending `LDG`, `BAR.SYNC`, or compute op in source.

---

## Dimension 4 — Tensor Core utilization

**What:** is the kernel using tensor cores at all? If yes, how well?

**Metrics:**
```
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed      # overall TC activity
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active       # per active SM cycle
sm__pipe_tensor_subpipe_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed  # BF16/FP16 MMA
sm__pipe_tensor_subpipe_imma_cycles_active.avg.pct_of_peak_sustained_elapsed  # INT MMA
sm__pipe_tensor_subpipe_dmma_cycles_active.avg.pct_of_peak_sustained_elapsed  # FP64 MMA
sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32_sparsity_off.avg       # BF16×BF16→FP32 ops
```

**Reading:**

- **`sm__pipe_tensor_cycles_active = 0%`**: no tensor core usage at all. For matmul-ish kernels (attention, GEMM, conv), this is almost always a missed optimization.
- **`... = X%` but X << 50%**: tensor cores are being used but underutilized. Usually means data isn't arriving fast enough (Dimension 6) or tile sizes are wrong.
- **`... > 50%`** on B200: kernel is doing well on the Tensor-Core front. Focus elsewhere.

🟠 **Blackwell (B200/B300):** 5th-gen tensor cores with `tcgen05.mma` + TMEM accumulators. Hand-rolled kernels need `tcgen05.alloc`, `tcgen05.mma`, `tcgen05.ld`, `tcgen05.dealloc` PTX. Most projects should use CUTLASS 4.x / cuBLAS instead. B300 has higher FP4 peak (15 PFLOPS vs B200's 9 PFLOPS). See [`../blackwell-cuda-programming.md`](../blackwell-cuda-programming.md).

🔵 **Hopper (H200/H800):** 4th-gen tensor cores with `wgmma` (warp-group MMA, 128 threads). No TMEM — accumulators live in registers. FP8 is the lowest supported precision (no FP4). See [`../hopper-cuda-programming.md`](../hopper-cuda-programming.md).

**Fix direction:** if you see 0% and the workload is matrix-multiplication-shaped, redesign around MMA. This is usually a major refactor but gives 2-10× on compute-bound paths.

---

## Dimension 5 — SM utilization timeline

**What:** how does SM utilization vary over the kernel's lifetime?

**Metrics (PM sampling, time-series):**
```
pmsampling:sm__throughput.avg.pct_of_peak_sustained_elapsed
pmsampling:sm__warps_active.avg.pct_of_peak_sustained_active
pmsampling:dram__throughput.avg.pct_of_peak_sustained_elapsed
pmsampling:smsp__warps_issue_stalled_long_scoreboard.avg
pmsampling:smsp__warps_issue_stalled_short_scoreboard.avg
```

**Reading (timeline shapes):**

- **Flat high, clean drop**: ideal.
- **Flat high, long tail**: tail effect (Dimension 2).
- **Flat low**: grid too small (Dimension 1) or severely stall-bound (Dimension 3).
- **Periodic sawtooth (compute ↕ memory)**: no compute-memory overlap — missing pipeline/double-buffering.
- **Slow ramp up, flat middle, clean drop**: kernel has warmup work (prologue), then steady state. Usually fine.

**Helper:** `plot_timeline.py` — renders ASCII plots. Look at multiple series side-by-side (SM throughput + DRAM throughput + long_scoreboard stalls) to distinguish the shapes.

**Note:** PM sampling interval is ~2 µs on Blackwell and Hopper. Very short kernels (< 20 µs) produce few samples — interpret with care.

---

## Dimension 6 — Memory access pattern & cache efficiency

**What:** are global loads coalesced? Are caches hit? Is DRAM actually busy?

**Metrics:**
```
# DRAM
dram__bytes_read.sum
dram__bytes_read.sum.pct_of_peak_sustained_elapsed
dram__bytes_write.sum.pct_of_peak_sustained_elapsed
dram__bytes_read.sum.per_second                    # achieved BW

# L1 / L2 hit rates
l1tex__t_sector_hit_rate.pct
lts__t_sector_hit_rate.pct
l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct
l1tex__t_sector_pipe_lsu_mem_global_op_st_hit_rate.pct

# Sectors per request (coalescing quality)
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum              # total sectors
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum             # total requests
# Compute sectors/request yourself — ideal is 4 (128B aligned load)

# Store efficiency
smsp__sass_average_data_bytes_per_sector_mem_global_op_st.ratio  # useful bytes/sector, max 32

# Register spill (local memory)
smsp__sass_inst_executed_op_local_ld.sum
smsp__sass_inst_executed_op_local_st.sum

# Global instruction counts
smsp__sass_inst_executed_op_global_ld.sum
smsp__sass_inst_executed_op_global_st.sum
smsp__sass_inst_executed_op_shared.sum                     # 0 if no shared memory
```

**Reading:**

- **`dram__bytes_read.sum.pct_of_peak_sustained_elapsed` ≈ 80-100%**: genuinely DRAM-bandwidth-bound. Reduce bytes / amortize reads with shared memory.
- **`... << 10%` but kernel is slow**: *not* bandwidth-bound. It's latency-bound (Dimension 3) or compute-bound (check `sm__throughput`).
- **`l1tex__t_sector_hit_rate.pct > 90%`**: good data locality, L1 is absorbing the reuse.
- **`lts__t_sector_hit_rate.pct < 50%`**: L2 is being blown through, reads fall to DRAM.
- **Sectors/request = 4.0 (ideal 128B fully-coalesced) to 5.0**: small coalescing issue but acceptable.
- **Sectors/request > 8.0**: serious non-coalesced access — big optimization opportunity.
- **`smsp__sass_average_data_bytes_per_sector_mem_global_op_st.ratio < 16`**: stores are using less than half of each 32B sector. Usually means only a subset of warp lanes write (e.g. `if (lane_id < 4)` patterns).
- **`smsp__sass_inst_executed_op_local_ld.sum > 0`**: **register spill** — very bad, local memory is DRAM-backed. Reduce register pressure with `__launch_bounds__` or kernel splitting.
- **`smsp__sass_inst_executed_op_shared.sum == 0`**: kernel uses no shared memory. Fine for element-wise kernels; often a missed optimization for data-reuse-heavy kernels.

NCU's rule engine often reports the coalescing issue directly:
```
OPT   Est. Speedup: 10.8%
      The memory access pattern for global loads from L1TEX might not be optimal.
      On average, only 7.6 of the 32 bytes transmitted per sector are utilized...
```

**Fix directions (by pattern):**

- Strided access (e.g. `x[lane * stride + i]`): change the per-thread layout so lanes access contiguous elements.
- AoS → SoA: restructure the data.
- Sparse writes: pack writes into a single coalesced store at the end of the warp.
- Register spill: add `__launch_bounds__`, reduce intermediate variables, or split kernel.

---

## Cross-dimension synthesis

After walking through all six, write a one-line diagnosis combining them. Structure: name the top 3–4 signals, each tied to a specific dimension. For example:

> "The kernel runs at X% of peak SM throughput and Y% of peak DRAM (Dim 1, 6). Stall time is dominated by `<stall_reason>` (Z% of samples, Dim 3), concentrated on <N> source lines whose access pattern is <coalesced/uncoalesced/...> (Dim 6). The PM timeline shows <flat / tail / sawtooth> shape (Dim 2/5). Tensor cores <used / unused at W%> (Dim 4)."

Fill in the X/Y/Z/W/N values and <classifications> from your own report. That sentence is the deliverable. Everything else in the report is evidence backing it.
