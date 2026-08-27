# Final Report Template

The report is the deliverable. Everything else (`.ncu-rep`, Python artifacts, CSVs) is evidence. Structure matters: a busy reader should see the top findings in 30 seconds and be able to drill into details if they want.

Save as `$PROFILE_RUN_DIR/REPORT.md`.

---

## Template

```markdown
# `<kernel_name>` Profiling Report

**Kernel:** `<exact kernel name or template instantiation>`
**Target GPU:** NVIDIA <B200 (148 SM, CC 10.0) | B300 (160 SM, CC 10.3) | H200 (132 SM, CC 9.0) | H800 (132 SM, CC 9.0)>
**Nsight Compute:** 2026.x.x
**Compile flags:** `nvcc -O2 -std=c++17 -lineinfo <-gencode=arch=compute_100,code=sm_100 | -gencode=arch=compute_103,code=sm_103 | -gencode=arch=compute_90,code=sm_90>`
**Profile date:** YYYY-MM-DD
**Run directory:** `profile/<run_name>/`

---

## 0. Profiling setup

> How exactly did we get these numbers? Required for reproducibility.

- Harness: `profile/<run_name>/harness/*.cu` — what it is (standalone driver / the original binary / something else). Why.
- Workloads: which real tensors / shapes were used. Cite the workload UUID or shape tuple.
- Dispatch paths covered: list each `(SF / template params, grid, block)` combination profiled.
- Metric-name caveats: any metric names that differ from stock NCU docs. Blackwell (sm_100/sm_103) adds `sass_` prefix to inst_executed_op_* metrics and splits `dram__bytes.sum`; Hopper (sm_90) uses older names. See [`08-gpu-metric-names.md`](reference/08-gpu-metric-names.md).

Minimal runnable command listing:

    # Compile
    nvcc -O2 -std=c++17 -lineinfo -gencode=arch=compute_100,code=sm_100 harness.cu -o harness

    # Profile (full + PM)
    ncu --set full --section PmSampling --section PmSampling_WarpStates \
        -k "regex:<kernel_regex>" -c 1 \
        -o profile/<run_name>/reports/full_<tag> \
        profile/<run_name>/harness/harness [args]

    # Profile (source-level)
    ncu --set source --section SourceCounters \
        -k "regex:<kernel_regex>" -c 1 \
        -o profile/<run_name>/reports/source_<tag> \
        profile/<run_name>/harness/harness [args]

### Artifacts

    profile/<run_name>/
    ├── REPORT.md                       ← this file
    ├── harness/...                     ← standalone harness
    ├── reports/                        ← raw .ncu-rep files (re-openable with ncu-ui)
    └── analysis/                       ← scripts + extracted metrics

---

## 1. Headline numbers

> A single table that tells the whole story at a glance.

| Metric | `<tag1>` | `<tag2>` | Source |
|---|---:|---:|---|
| **Duration** | X µs | Y µs | `gpu__time_duration.sum` |
| SM throughput (% peak) | X% | Y% | `sm__throughput.avg.pct_of_peak_sustained_elapsed` |
| Memory throughput (% peak) | X% | Y% | `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` |
| DRAM throughput (% peak) | X% | Y% | `dram__bytes_read.sum.pct_of_peak_sustained_elapsed` |
| L1 hit rate | X% | Y% | `l1tex__t_sector_hit_rate.pct` |
| L2 hit rate | X% | Y% | `lts__t_sector_hit_rate.pct` |
| Tensor Core usage | X% | Y% | `sm__pipe_tensor_cycles_active.*` |
| Reg / thread | X | Y | `launch__registers_per_thread` |
| Theoretical / Achieved occupancy | X% / Y% | ... | |
| Waves / SM | X | Y | `launch__waves_per_multiprocessor` |

**One-line read:** <"The kernel runs at X% of peak SM throughput — it's latency-bound on Y, not DRAM-BW-bound."> — this is the punchline.

---

## 2. Per-dimension analysis

> Walk through the six analysis dimensions, cite metrics, state findings.

### 2.1 SM occupancy & launch geometry
<grid size, block size, waves/SM, occupancy, register/shared-mem limits, wave math>

### 2.2 Thread-block balance (tail effect)
<per-SM active cycles, PM timeline shape, input distribution imbalance ratios>

### 2.3 Instruction-level stall analysis
<stall breakdown %, top source-line hotspots (cite file:line + samples + stall type)>

### 2.4 Tensor Core utilization
<value or "0%, n/a">

### 2.5 SM utilization timeline
<shape: flat-high / flat-low / tail / sawtooth — reference the ASCII plot in analysis/>

### 2.6 Memory access pattern
<sectors/request, L1/L2 hits, DRAM throughput, store efficiency, register spill>

### 2.7 Additional findings
<items from NCU rule engine not otherwise mentioned — each with the rule's `Est. Speedup: X%`>

---

## 3. Summary diagnosis

| Factor | `<tag1>` | `<tag2>` | Impact |
|---|---|---|---|
| <factor 1> | <status> | <status> | <ranked impact> |
| <factor 2> | ... | ... | ... |

---

## 4. Optimization directions (ranked by impact)

> Each priority: name the change, cite evidence, estimate magnitude, flag effort.

### Priority 1 — <one-line name>

<what to do, concretely, with line numbers / function names from the existing kernel>

**Evidence:**
- <metric + value>
- <NCU rule + est. speedup>

**Expected impact:** <X% end-to-end, Y% on the hot path>, <which workloads benefit>

**Effort:** <low/medium/high + rough description of the code change>

### Priority 2 — ...

<same structure>

### Priority 3 — ...

(Stop at 3-5. More dilutes the signal.)

---

## 5. Confidence & caveats

- What I'm sure about: <list>
- What I'm uncertain about: <list + what would resolve the uncertainty>
- Anything the profile couldn't answer that the user should know: <list>

---

## 6. Reproduction

    cd /abs/path/to/repo
    export PROFILE_RUN_DIR=profile/<run_name>
    <one-block runnable script that builds the harness, runs ncu, and parses>
```

---

## Style rules

- **Cite specific metric values for every claim.** "SM throughput X.X%" (with the actual number from your report) > "SM throughput is low".
- **Name files and line numbers.** "Line L of `harness.cu`" (pasting the actual file/line) > a high-level description like "the main memory load".
- **Use NCU's own estimates.** Rule-engine `Est. Speedup: X%` numbers are usually in the right ballpark — use them instead of guessing.
- **Rank by magnitude.** Fix the 50% problem before the 5% problem.
- **Keep the top-line summary dense.** A reader should be able to get the #1 finding in 10 seconds of reading.
- **Link to artifacts.** Don't paste huge tables into the prose — link to `analysis/compare_<tag1>_vs_<tag2>.txt` etc.

## Anti-patterns

- ❌ Generic advice without evidence ("you might consider using shared memory").
- ❌ More than 5 "priorities" — you're probably padding.
- ❌ Re-running the same profile with different tags and copy-pasting the same analysis — consolidate.
- ❌ Reporting from the CLI table directly. Extract, interpret, write — don't dump.
- ❌ Omitting the setup section. Without it, nobody can reproduce or trust the numbers.
