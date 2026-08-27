# GPU Metric Name Reference (Multi-Architecture)

Nsight Compute metric names **are not uniform across architectures**. This doc covers three generations:

| Architecture | GPUs | Compute Capability | ncu chip |
|---|---|---|---|
| Hopper | H100, H200, H800 | sm_90 / CC 9.0 | `gh100` |
| Blackwell | B200 | sm_100 / CC 10.0 | `gb202` |
| Blackwell Ultra | B300 | sm_103 / CC 10.3 | `gb300` |

> **Note on B300 chip name:** run `ncu --list-chips` on your target machine to confirm the exact name. `gb300` is the expected value based on the naming pattern; NVIDIA may use a variant like `gb203`.

If a metric returns `None` on your kernel, first check this doc, then enumerate:

```python
all_names = action.metric_names()   # all metrics collected in this report
```

```bash
ncu --query-metrics --chip gh100    # all metrics available on H100/H200/H800
ncu --query-metrics --chip gb202    # all metrics available on B200
ncu --query-metrics --chip gb300    # all metrics available on B300
ncu --list-chips                    # list all chip names known to this ncu build
```

---

## Summary: what changed between architectures

> The table below is based on real H800 (gh100) enumeration via `ncu --query-metrics --chip gh100` (5540 metrics, Nsight Compute 2026.1).

| Metric category | Hopper (sm_90, H200/H800) | Blackwell (sm_100/sm_103, B200/B300) |
|---|---|---|
| Global LD/ST inst counts | **Both** `smsp__inst_executed_op_global_ld.sum` (old) and `smsp__sass_inst_executed_op_global_ld.sum` (new) exist; prefer `smsp__sass_*` | `smsp__sass_inst_executed_op_global_ld.sum` only |
| DRAM bytes total | `dram__bytes.sum` **exists** directly | Only `dram__bytes_read.sum + dram__bytes_write.sum` (split) |
| Sectors/request ratio | `l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio` **exists as direct ratio** | Not available as direct ratio — compute from `.sum / .sum` |
| FP64 heavy pipe | `sm__inst_executed_pipe_fmaheavy.*` | Not present — use `sm__inst_executed_pipe_fma.*` |
| Warp stall names | `smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio` — **same as Blackwell** | `smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio` |
| Hopper-only stall | `smsp__average_warps_issue_stalled_gmma_per_issue_active.ratio` (waiting on `WARPGROUP.ARRIVES` / wgmma) | Not present |
| FP4 tensor ops | Not supported | `sm__ops_path_tensor_op_*_sparsity_off.avg` (FP4 variants) |
| HMMA/BF16 cycle metric | `sm__pipe_tensor_op_hmma_cycles_active.*` (`op_` prefix) | `sm__pipe_tensor_subpipe_hmma_cycles_active.*` (`subpipe_` prefix) |
| TMEM metrics | Not present | `smsp__sass_inst_executed_op_tmem_*.sum` |

---

## Hopper (sm_90) — H100 / H200 / H800 metric names

Hopper uses the **older** NCU naming convention (no `sass_` prefix, `dram__bytes.sum` available directly). The names below have been confirmed on H100/sm_90 with Nsight Compute ≥ 2023.x.

> **H800 vs H100/H200:** identical chip (GH100 die), identical metric names. The only difference is HBM bandwidth (~2.0 TB/s on H800 vs 3.35 TB/s on H100 vs 4.8 TB/s on H200 HBM3e). The bandwidth ceiling reflected in `pct_of_peak_sustained_elapsed` metrics will differ accordingly.

### Launch geometry / occupancy
```
launch__grid_size
launch__block_size
launch__waves_per_multiprocessor
launch__registers_per_thread
launch__shared_mem_per_block_static
launch__shared_mem_per_block_dynamic
launch__occupancy_limit_blocks
launch__occupancy_limit_registers
launch__occupancy_limit_shared_mem
launch__occupancy_limit_warps
device__attribute_multiprocessor_count          # 132 on H100/H200/H800 (SXM5)
sm__maximum_warps_per_active_cycle_pct          # theoretical occupancy %
```

### SOL / throughput
```
sm__throughput.avg.pct_of_peak_sustained_elapsed
l1tex__throughput.avg.pct_of_peak_sustained_active
lts__throughput.avg.pct_of_peak_sustained_elapsed
dram__bytes.sum                                 # total DRAM bytes (read+write, available on Hopper)
dram__bytes_read.sum
dram__bytes_read.sum.pct_of_peak_sustained_elapsed
dram__bytes_read.sum.per_second
dram__bytes_write.sum
dram__bytes_write.sum.pct_of_peak_sustained_elapsed
```

### Warp activity
```
sm__warps_active.avg.pct_of_peak_sustained_active      # achieved occupancy %
smsp__warps_active.avg.per_cycle_active
smsp__warps_eligible.avg.per_cycle_active
```

### Compute pipelines
```
sm__inst_executed.avg.per_cycle_active                  # IPC
sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active
sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_elapsed
sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active
sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active
sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active
sm__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_active

# Tensor Core (Hopper = 4th gen, wgmma)
# NOTE: Hopper uses "op_" prefix (NOT "subpipe_" — that's Blackwell naming)
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed
sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed    # FP16/BF16/FP8 HMMA+GMMA
sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_elapsed    # INT MMA
sm__pipe_tensor_op_dmma_cycles_active.avg.pct_of_peak_sustained_elapsed    # FP64 MMA
sm__inst_executed_pipe_tensor_op_gmma.avg                                  # wgmma (GMMA) instruction count
sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32_sparsity_off.avg             # BF16→FP32 tensor ops
sm__ops_path_tensor_src_fp8_sparsity_off.avg                               # FP8 tensor ops (E4M3/E5M2)
# Note: no FP4 on Hopper — minimum precision is FP8
```

### Memory access counts (Hopper — BOTH old and sass_ names exist)
```
# Both naming conventions coexist on H800/gh100. Prefer smsp__sass_* for consistency.
smsp__sass_inst_executed_op_global_ld.sum      # preferred — same name as Blackwell
smsp__inst_executed_op_global_ld.sum           # also exists (old name, LDG only)
smsp__sass_inst_executed_op_global_st.sum
smsp__inst_executed_op_global_st.sum
smsp__sass_inst_executed_op_local_ld.sum       # local LD (register spill)
smsp__inst_executed_op_local_ld.sum            # also exists (LDL only)
smsp__sass_inst_executed_op_local_st.sum
smsp__sass_inst_executed_op_shared_ld.sum
smsp__sass_inst_executed_op_shared_st.sum

l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum

# Sectors/request — available as a DIRECT ratio on Hopper (unlike Blackwell):
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio
```

### Cache hit rates
```
l1tex__t_sector_hit_rate.pct
lts__t_sector_hit_rate.pct
l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct
l1tex__t_sector_pipe_lsu_mem_global_op_st_hit_rate.pct
```

### Warp stall reasons (Hopper — same `average_` prefix and `.ratio` suffix as Blackwell)
```
# Names confirmed on H800 (gh100): identical to Blackwell naming, NOT the old .pct form.
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio
smsp__average_warps_issue_stalled_wait_per_issue_active.ratio
smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio
smsp__average_warps_issue_stalled_membar_per_issue_active.ratio
smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio
smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio
smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio
smsp__average_warps_issue_stalled_tex_throttle_per_issue_active.ratio
smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio
smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio
smsp__average_warps_issue_stalled_selected_per_issue_active.ratio      # productive
smsp__average_warps_issue_stalled_sleeping_per_issue_active.ratio
smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio
smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio
smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio      # immediate constant cache miss

# Hopper-specific stall (NOT present on Blackwell):
smsp__average_warps_issue_stalled_gmma_per_issue_active.ratio          # waiting for wgmma to complete (WARPGROUP.ARRIVES)
```

### Per-PC stall sampling (requires `--set source --section SourceCounters`)
```
smsp__pcsamp_sample_count
smsp__pcsamp_warps_issue_stalled_long_scoreboard
smsp__pcsamp_warps_issue_stalled_short_scoreboard
smsp__pcsamp_warps_issue_stalled_wait
smsp__pcsamp_warps_issue_stalled_barrier
smsp__pcsamp_warps_issue_stalled_math_pipe_throttle
smsp__pcsamp_warps_issue_stalled_mio_throttle
smsp__pcsamp_warps_issue_stalled_selected
```

### Timing
```
gpu__time_duration.sum
smsp__cycles_active.avg
smsp__issue_active.avg.per_cycle_active
```

> **H800 verification tip:** if you have an H800 machine, run the following to dump and verify all available metric names:
> ```bash
> ncu --query-metrics --chip gh100 > /tmp/h800_metrics.txt
> grep -i "inst_executed_op_global" /tmp/h800_metrics.txt
> grep -i "dram__bytes" /tmp/h800_metrics.txt
> grep -i "issue_stalled" /tmp/h800_metrics.txt | head -30
> ```

---

## Blackwell (sm_100 / sm_103) — B200 / B300 metric names

B200 (sm_100) and B300 (sm_103) share the same metric namespace — code compiled with `-arch=compute_100` (or `compute_100f` family) runs on both. The `sass_` prefix was added to instruction-count metrics.

### Key differences from Hopper

| Category | Change |
|---|---|
| Global LD/ST | Added `sass_` prefix: `smsp__sass_inst_executed_op_global_ld.sum` |
| DRAM bytes | No `dram__bytes.sum` — use `dram__bytes_read.sum + dram__bytes_write.sum` |
| Sectors/request ratio | Not available as direct `.ratio` — compute from `.sum / .sum` |
| `fmaheavy` pipe | Removed — use `sm__inst_executed_pipe_fma.*` |
| Stall ratio suffix | Changed from `.pct` to `.ratio`, added `average_` prefix |
| Store efficiency | `smsp__sass_average_data_bytes_per_sector_mem_global_op_st.ratio` |
| FP4 tensor ops | New: `sm__ops_path_tensor_op_hmma_src_*_sparsity_off.avg` |
| TMEM | New: `smsp__sass_inst_executed_op_tmem_*.sum` |

### Launch geometry / occupancy
```
launch__grid_size
launch__block_size
launch__grid_dim_x, launch__grid_dim_y, launch__grid_dim_z
launch__block_dim_x, launch__block_dim_y, launch__block_dim_z
launch__thread_count
launch__waves_per_multiprocessor
launch__registers_per_thread
launch__shared_mem_per_block
launch__shared_mem_per_block_static
launch__shared_mem_per_block_dynamic
launch__occupancy_limit_blocks
launch__occupancy_limit_registers
launch__occupancy_limit_shared_mem
launch__occupancy_limit_warps
device__attribute_multiprocessor_count          # 148 on B200, 160 on B300
sm__maximum_warps_per_active_cycle_pct          # theoretical occupancy %
```

### SOL / throughput
```
sm__throughput.avg.pct_of_peak_sustained_elapsed
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed
gpu__compute_memory_access_throughput.avg.pct_of_peak_sustained_elapsed
l1tex__throughput.avg.pct_of_peak_sustained_active
lts__throughput.avg.pct_of_peak_sustained_elapsed
dram__bytes_read.sum
dram__bytes_read.sum.pct_of_peak_sustained_elapsed
dram__bytes_read.sum.per_second
dram__bytes_write.sum
dram__bytes_write.sum.pct_of_peak_sustained_elapsed
dram__sectors_read.sum
dram__sectors_write.sum
```

### Timing
```
gpu__time_duration.sum
smsp__cycles_active.avg
smsp__issue_active.avg.per_cycle_active
smsp__issue_active.avg.pct_of_peak_sustained_active
```

### Warp activity
```
sm__warps_active.avg.pct_of_peak_sustained_active      # achieved occupancy %
sm__warps_active.avg.per_cycle_active
smsp__warps_active.avg.per_cycle_active
smsp__warps_eligible.avg.per_cycle_active
smsp__warps_eligible.max.per_cycle_active
```

### Compute pipelines
```
sm__inst_executed.avg.per_cycle_active                  # IPC
sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active
sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_elapsed
sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active
sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active
sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active
sm__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_active
sm__inst_executed_pipe_adu.avg.pct_of_peak_sustained_active

# Tensor Core (Blackwell = 5th gen, tcgen05)
# NOTE: Blackwell uses "subpipe_" prefix (NOT "op_" — that's Hopper naming)
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed
sm__pipe_tensor_subpipe_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed   # BF16/FP16 MMA
sm__pipe_tensor_subpipe_imma_cycles_active.avg.pct_of_peak_sustained_elapsed   # INT MMA
sm__pipe_tensor_subpipe_dmma_cycles_active.avg.pct_of_peak_sustained_elapsed   # FP64 MMA
sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32_sparsity_off.avg   # BF16→FP32 tensor ops
# FP4 is Blackwell-only — B300 has higher FP4 throughput (15 PFLOPS vs 9 PFLOPS on B200)
```

### Memory access counts (Blackwell — WITH sass_ prefix)
```
smsp__sass_inst_executed_op_global_ld.sum          # global LD instruction count
smsp__sass_inst_executed_op_global_st.sum          # global ST count
smsp__sass_inst_executed_op_local_ld.sum           # local LD (register spill)
smsp__sass_inst_executed_op_local_st.sum           # local ST (register spill)
smsp__sass_inst_executed_op_shared.sum             # total shared mem ops
smsp__sass_inst_executed_op_shared_ld.sum
smsp__sass_inst_executed_op_shared_st.sum

l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum
l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum
l1tex__t_requests_pipe_lsu_mem_global_op_st.sum
# sectors/request = sectors.sum / requests.sum (ideal = 4 for 128B coalesced)

smsp__sass_average_data_bytes_per_sector_mem_global_op_st.ratio   # store efficiency (max 32)
```

### Cache hit rates
```
l1tex__t_sector_hit_rate.pct
lts__t_sector_hit_rate.pct
l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct
l1tex__t_sector_pipe_lsu_mem_global_op_st_hit_rate.pct
```

### Warp stall reasons — aggregate (Blackwell: `average_` prefix, `.ratio` suffix)
```
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio
smsp__average_warps_issue_stalled_wait_per_issue_active.ratio
smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio
smsp__average_warps_issue_stalled_membar_per_issue_active.ratio
smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio
smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio
smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio
smsp__average_warps_issue_stalled_tex_throttle_per_issue_active.ratio
smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio
smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio
smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio
smsp__average_warps_issue_stalled_drain_per_issue_active.ratio
smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio
smsp__average_warps_issue_stalled_sleeping_per_issue_active.ratio
smsp__average_warps_issue_stalled_misc_per_issue_active.ratio
smsp__average_warps_issue_stalled_selected_per_issue_active.ratio       # productive (= 1.0)
```

### Per-PC stall sampling (requires `--set source --section SourceCounters`)
```
smsp__pcsamp_sample_count
smsp__pcsamp_warps_issue_stalled_long_scoreboard
smsp__pcsamp_warps_issue_stalled_short_scoreboard
smsp__pcsamp_warps_issue_stalled_wait
smsp__pcsamp_warps_issue_stalled_barrier
smsp__pcsamp_warps_issue_stalled_math_pipe_throttle
smsp__pcsamp_warps_issue_stalled_mio_throttle
smsp__pcsamp_warps_issue_stalled_lg_throttle
smsp__pcsamp_warps_issue_stalled_tex_throttle
smsp__pcsamp_warps_issue_stalled_not_selected
smsp__pcsamp_warps_issue_stalled_dispatch_stall
smsp__pcsamp_warps_issue_stalled_drain
smsp__pcsamp_warps_issue_stalled_no_instructions
smsp__pcsamp_warps_issue_stalled_selected
smsp__pcsamp_warps_issue_stalled_branch_resolving
smsp__pcsamp_warps_issue_stalled_membar
```

### PM sampling (time series)
```
pmsampling:smsp__warps_issue_stalled_long_scoreboard.avg
pmsampling:smsp__warps_issue_stalled_short_scoreboard.avg
pmsampling:smsp__warps_issue_stalled_wait.avg
pmsampling:smsp__warps_issue_stalled_dispatch_stall.avg
pmsampling:smsp__warps_issue_stalled_branch_resolving.avg
pmsampling:smsp__warps_issue_stalled_math_pipe_throttle.avg
pmsampling:smsp__warps_issue_stalled_mio_throttle.avg
pmsampling:smsp__warps_issue_stalled_lg_throttle.avg
pmsampling:smsp__warps_issue_stalled_no_instruction.avg
pmsampling:smsp__warps_issue_stalled_drain.avg
pmsampling:smsp__warps_issue_stalled_barrier.avg
```

Note: some `pmsampling:` metrics (notably `pmsampling:sm__throughput.*` and `pmsampling:dram__throughput.*`) may return empty instance arrays depending on ncu version / driver — always check `m.num_instances() > 0` before using.

---

## Quick-reference: names that changed between Hopper and Blackwell

> Based on real H800 (gh100) metric enumeration. Corrections from assumed vs. actual:

| Metric | Hopper (sm_90, H200/H800) | Blackwell (sm_100/sm_103, B200/B300) |
|---|---|---|
| Global LD inst count | `smsp__sass_inst_executed_op_global_ld.sum` ✓ (old `smsp__inst_executed_op_global_ld.sum` also exists) | `smsp__sass_inst_executed_op_global_ld.sum` only |
| Global ST inst count | `smsp__sass_inst_executed_op_global_st.sum` ✓ (both exist) | `smsp__sass_inst_executed_op_global_st.sum` only |
| Local LD/ST (spill) | `smsp__sass_inst_executed_op_local_ld.sum` (both exist) | `smsp__sass_inst_executed_op_local_ld.sum` only |
| DRAM total bytes | `dram__bytes.sum` **exists directly** ✓ | Not available — use `dram__bytes_read.sum + dram__bytes_write.sum` |
| Sectors/request | `l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio` **exists directly** ✓ | Not available as ratio — compute `sectors.sum / requests.sum` |
| FP64 heavy pipe | `sm__inst_executed_pipe_fmaheavy.*` | Not present — use `sm__inst_executed_pipe_fma.*` |
| Warp stall names | `smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio` — **same as Blackwell** ✓ | `smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio` |
| Hopper-only stall | `smsp__average_warps_issue_stalled_gmma_per_issue_active.ratio` | Not present |
| HMMA/BF16 TC cycles | `sm__pipe_tensor_op_hmma_cycles_active.*` (`op_` prefix) | `sm__pipe_tensor_subpipe_hmma_cycles_active.*` (`subpipe_` prefix) |
| wgmma instruction | `sm__inst_executed_pipe_tensor_op_gmma.*` | Not present (uses tcgen05 path) |
| FP4 tensor ops | Not supported | `sm__ops_path_tensor_op_*` FP4 variants |
| TMEM metrics | Not present | `smsp__sass_inst_executed_op_tmem_*.sum` |

---

## Gotchas

1. **Metric exists in ncu's list but returns `None` from Python**: the metric wasn't *collected* in this report. Rerun ncu with the right `--section` or `--set`.
2. **Metric value is `0.0`**: either the hardware counter reports zero (e.g., no tensor core activity), or the metric is synthetic and depends on other metrics that weren't collected.
3. **`.avg` vs `.sum` vs `.max`**: each aggregate is a separate metric name. `.avg` is most useful for rates/percentages; `.sum` for counts; `.max` for worst-case analysis.
4. **`pct_of_peak_sustained_elapsed` vs `pct_of_peak_sustained_active`**: `_elapsed` normalizes against total kernel time (including idle SMs); `_active` normalizes against cycles where the SM was actually running. `_elapsed` is more honest for under-utilized kernels.
5. **H800 peak bandwidth**: `dram__bytes_read.sum.pct_of_peak_sustained_elapsed` uses the H800's reduced HBM bandwidth (~2 TB/s) as the denominator — not H100's 3.35 TB/s. This is correct behavior; the same absolute bytes/s will show a higher percentage on H800.
6. **`pcsamp` and `pmsampling` metrics are not listed in `--query-metrics` output** — they are collected only when you use `--set source --section SourceCounters` (pcsamp) or `--section PmSampling` (pmsampling) at profile time. This applies to both Hopper and Blackwell.
7. **Hopper-only stall `gmma`**: if `smsp__average_warps_issue_stalled_gmma_per_issue_active.ratio` is high on H200/H800, warps are stalled waiting for `WARPGROUP.ARRIVES` (wgmma pipeline completion). Fix: increase pipeline depth (more independent wgmma in flight), or widen the MMA tile so each wgmma does more work per stall.
