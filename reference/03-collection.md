# Profile Collection Commands

This document lists the exact `ncu` commands you should run, in what order, and what each flag does.

---

## Prerequisites recap

- `-lineinfo` in the compile flags (see `02-harness-guide.md`).
- `ncu` available on PATH.
- Writable `$HOME` so ncu can cache section files.
- Kernel name known (check with `cuobjdump --dump-function-names your_binary` if unsure).

Quick permission test:
```bash
ncu --section SpeedOfLight -k "regex:YOUR_KERNEL_NAME" -c 1 ./harness [args]
# If you see ERR_NVGPUCTRPERM: need sudo or edit /etc/modprobe.d/ncu.conf (see 09-common-issues.md).
# If you see the SpeedOfLight table and "regex" matched, you're good.
```

---

## Recipe 1: Full overview (first pass)

Collects all standard sections plus PM sampling (time-series data). This is the mandatory first run.

```bash
ncu --set full \
    --section PmSampling \
    --section PmSampling_WarpStates \
    -k "regex:KERNEL_REGEX" \
    -c 1 \
    -o $PROFILE_RUN_DIR/reports/full_<tag> \
    ./harness [args]
```

| Flag | Meaning |
|---|---|
| `--set full` | Run all built-in sections — SOL, Occupancy, Memory, Compute, Scheduler, Launch, etc. |
| `--section PmSampling` | Add performance-monitor time-series data (not included in `full`). Needed to see tail effects. |
| `--section PmSampling_WarpStates` | Time-series of warp stall states. |
| `-k "regex:..."` | Only profile kernels whose demangled name matches. Reduces replay count. |
| `-c 1` | Only profile the first matching kernel launch. Avoids duplicates when the kernel is called in a loop. |
| `-o $PROFILE_RUN_DIR/reports/full_<tag>` | Output path — `.ncu-rep` is appended automatically. |

Replay count: typically 45-50 passes (5-40 seconds of wall time). Slow because each pass reruns the kernel to collect a different metric group.

---

## Recipe 2: Source-level profile (second pass)

Collects per-PC stall sampling data. Requires `-lineinfo` at compile time. Fast (~5 passes).

```bash
ncu --set source \
    --section SourceCounters \
    -k "regex:KERNEL_REGEX" \
    -c 1 \
    -o $PROFILE_RUN_DIR/reports/source_<tag> \
    ./harness [args]
```

Use Recipe 2's output with `extract_stall_hotspots.py` to get per-source-line stall samples.

---

## Recipe 3: Details page (quick rule summary)

No need to collect again — just import an existing `full_<tag>.ncu-rep`:

```bash
ncu --import $PROFILE_RUN_DIR/reports/full_<tag>.ncu-rep --page details > analysis/details_<tag>.txt
```

This produces NCU's human-readable details page, including the built-in rule-engine suggestions. Each rule looks like:

```
OPT   Est. Speedup: <pct>%
      <description of the pattern NCU detected, e.g. "On average, each warp
      spends N cycles stalled waiting for a scoreboard dependency on a L1TEX
      operation. Find the instruction producing the data being waited upon
      to identify the culprit.">
```

**Always read `details_<tag>.txt` first.** The rule engine is shockingly accurate and often points straight at the answer.

---

## Recipe 4: CSV / raw export (scripting)

```bash
# Full metric table as CSV — one row per kernel launch, one column per metric
ncu --import $PROFILE_RUN_DIR/reports/full_<tag>.ncu-rep --page raw --csv > analysis/raw_<tag>.csv

# Source page as text
ncu --import $PROFILE_RUN_DIR/reports/source_<tag>.ncu-rep --page source > analysis/source_<tag>.txt
```

Most of the time you don't need these — the Python API (`04-python-api.md`) is easier. But CSV is handy for quick `grep`/`awk` one-liners.

---

## Recipe 5: Targeted metrics only (fast)

If you already know which metrics you want (e.g., you're re-running after a code change and only want to check if the fix worked), collect just those:

```bash
ncu --metrics \
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    sm__warps_active.avg.pct_of_peak_sustained_active,\
    dram__bytes_read.sum.pct_of_peak_sustained_elapsed,\
    l1tex__t_sector_hit_rate.pct,\
    gpu__time_duration.sum,\
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
    l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum \
  -k "regex:KERNEL_REGEX" -c 1 \
  ./harness [args]
```

Takes one or two replay passes. Prints a table directly to stdout. No `-o` output file.

---

## Recipe 6: A/B comparison (before vs after optimization)

```bash
# Before
ncu --set full -k "regex:my_kernel" -c 1 \
    -o $PROFILE_RUN_DIR/reports/v1 ./harness_v1 [args]

# After
ncu --set full -k "regex:my_kernel" -c 1 \
    -o $PROFILE_RUN_DIR/reports/v2 ./harness_v2 [args]
```

Then use `analyze_reports.py` with multiple tags to produce a side-by-side comparison, or manually:

```python
import ncu_report
r1 = ncu_report.load_report("$PROFILE_RUN_DIR/reports/v1.ncu-rep")
r2 = ncu_report.load_report("$PROFILE_RUN_DIR/reports/v2.ncu-rep")
a1 = r1.range_by_idx(0).action_by_idx(0)
a2 = r2.range_by_idx(0).action_by_idx(0)
t1 = a1["gpu__time_duration.sum"].value()
t2 = a2["gpu__time_duration.sum"].value()
print(f"Speedup: {t1/t2:.2f}x")
```

---

## What each `--set` contains

```bash
ncu --list-sets           # list all sets
ncu --list-sections       # list all sections
```

Rough mapping (Blackwell B200/B300 and Hopper H200/H800 are similar; B200 with ncu 2026.1 shown):

| Set | Sections included | Replay passes | Use when |
|---|---|---|---|
| `basic` (default) | SOL, LaunchStats, Occupancy | ~3-5 | Smoke test — is ncu working at all? |
| `detailed` | basic + Scheduler, WarpState, ComputeWorkload, MemoryWorkload, InstructionStats | ~15 | Middle-ground, often too limited |
| `full` | everything except Source | ~45 | First-pass profile. Always start here. |
| `source` | full + SourceCounters (per-PC data) | ~50 | Per-line stall attribution. Needs `-lineinfo`. |
| `pmsampling` | only PM sampling (time series) | ~3 | Already covered by `--section PmSampling` on top of full |

---

## Common section additions

```bash
# PM sampling (not in any --set)
--section PmSampling
--section PmSampling_WarpStates

# Source counters (not in --set full, is in --set source)
--section SourceCounters

# NVLink bandwidth (if profiling multi-GPU)
--section Nvlink_Topology
--section Nvlink_Tables
```

---

## Profiling multiple kernel launches

If the kernel is called multiple times and you want different iterations:

```bash
# Skip the first 5 invocations, then profile 3 consecutive launches
ncu -k "regex:my_kernel" -s 5 -c 3 -o report ./harness
```

`-s N` skips the first N matches. `-c N` limits to N matches. Useful for ignoring warmup or focusing on a steady-state iteration.

---

## GPU frequency locking (for reproducibility)

```bash
# Check current clocks
nvidia-smi -q -d CLOCK

# Lock to boost frequency (sudo required)
sudo nvidia-smi -lgc <boost_clock_mhz>

# Unlock when done
sudo nvidia-smi -rgc
```

For Blackwell (B200/B300) and Hopper (H200/H800) this is usually unnecessary — the GPU boosts to steady-state during profiling because ncu replays the kernel 45+ times. If your results are jittery between runs, lock the clock.

---

## Gotchas

- **Profile run time blows up with `--set full`**: that's normal, each replay is a full kernel execution.
- **`regex:...` doesn't match anything**: check with `cuobjdump --dump-function-names ./harness` and make sure you're looking at the demangled name. Templates produce names like `void my_kernel<(int)8, (int)256>(...)` — the regex needs to match this string.
- **Report file is empty / 0 KB**: profile terminated before the kernel launched. Usually means `-k` regex didn't match, or the harness crashed.
- **PM Sampling returns nothing**: check you used `--section PmSampling` and the GPU isn't a vGPU (vGPU doesn't support PM sampling).
- **"Could not deploy stock section files to $HOME"**: set `HOME` to a writable directory first.
