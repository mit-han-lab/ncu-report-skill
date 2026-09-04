# Helpers

Reusable code for profiling harnesses and report analysis. See `../SKILL.md` for context.

## C++ / CUDA

| File | Purpose |
|---|---|
| `harness_template.cu` | Starting point for a profiling harness. Copy into your run dir, fill in the `TODO(you)` sections. |
| `safetensors_loader.h` | Header-only safetensors reader (no external deps). Use from your harness to load real workload tensors. |

### Typical harness setup

```bash
cd profile/<run_name>/harness/
cp /path/to/skills/kernel-profiling/helpers/harness_template.cu my_kernel_harness.cu
cp /path/to/skills/kernel-profiling/helpers/safetensors_loader.h .
# edit my_kernel_harness.cu to include your kernel + fill in main()
nvcc -O2 -std=c++17 -lineinfo -gencode=arch=compute_100,code=sm_100 \
     my_kernel_harness.cu -o my_kernel_harness
```

## Python

| File | Purpose |
|---|---|
| `ncu_utils.py` | Shared helpers: `load_report`, `safe`, `per_pc_values`, `B200_KEY_METRICS`, `rule_speedups`, ... |
| `analyze_reports.py` | Extract key metrics + side-by-side comparison from one or more `.ncu-rep`s |
| `extract_stall_hotspots.py` | Aggregate per-PC stall samples → per-source-line rankings (requires source-level report) |
| `plot_timeline.py` | ASCII plot PM sampling timelines (reveals tail effect, pipeline bubbles) |
| `sass_source_counters.py` | Per-instruction view of one or two reports: top stall PCs, divergent branches, execution counts of barrier/sync/atomic instructions, optional instruction-by-instruction diff (uses `ncu --import --page source --print-source sass --csv`) |
| `list_flashinfer_workloads.py` | Browse a flashinfer-trace dataset: show axes, histogram workload shapes, print safetensors paths for specific UUIDs |

### Typical Python workflow

```bash
export PROFILE_RUN_DIR=profile/<run_name>
HELPERS=/path/to/skills/kernel-profiling/helpers
export FIB_DATASET_PATH=/path/to/flashinfer-trace  # if using FIB workloads

# (Optional) Browse workload shapes for a flashinfer-trace dataset
python3 $HELPERS/list_flashinfer_workloads.py --definition <def_name>
python3 $HELPERS/list_flashinfer_workloads.py --definition <def_name> --unique-axes <axis1>,<axis2> --no-paths

# Extract key metrics for each report
python3 $HELPERS/analyze_reports.py --run-dir $PROFILE_RUN_DIR \
    --report $PROFILE_RUN_DIR/reports/full_<tag1>.ncu-rep --tag <tag1> \
    --report $PROFILE_RUN_DIR/reports/full_<tag2>.ncu-rep --tag <tag2>

# Per-line stall hotspots (requires source-level reports, collected with --set source)
python3 $HELPERS/extract_stall_hotspots.py --run-dir $PROFILE_RUN_DIR \
    --report $PROFILE_RUN_DIR/reports/source_<tag1>.ncu-rep --tag <tag1> \
    --report $PROFILE_RUN_DIR/reports/source_<tag2>.ncu-rep --tag <tag2>

# ASCII PM timeline plots
python3 $HELPERS/plot_timeline.py --run-dir $PROFILE_RUN_DIR \
    --report $PROFILE_RUN_DIR/reports/full_<tag1>.ncu-rep --tag <tag1> \
    --report $PROFILE_RUN_DIR/reports/full_<tag2>.ncu-rep --tag <tag2>

# SASS-level counters: where do warps arrive at barriers split, which sync instructions execute how often
python3 $HELPERS/sass_source_counters.py $PROFILE_RUN_DIR/reports/full_<tag1>.ncu-rep
python3 $HELPERS/sass_source_counters.py $PROFILE_RUN_DIR/reports/full_<tag1>.ncu-rep --diff $PROFILE_RUN_DIR/reports/full_<tag2>.ncu-rep
```

All three run-dir scripts take `--run-dir` and write under `<run-dir>/analysis/`; `sass_source_counters.py` prints to stdout (or `--out FILE`).

`ncu_utils.py` tries to auto-locate `ncu_report` from common CUDA install paths. If that fails, set `PYTHONPATH`:

```bash
export PYTHONPATH=$PYTHONPATH:/usr/local/cuda-13.2/nsight-compute-2026.1.0/extras/python
```
