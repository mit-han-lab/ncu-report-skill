---
name: ncu-report-skill
description: Profile CUDA kernels with Nsight Compute on B200 / B300 / H200 / H800 (sm_100 / sm_103 / sm_90). Use when the user asks to profile a kernel, analyze its performance, diagnose bottlenecks, read an ncu report, or write an optimization plan — including variants in Chinese ("profile 一下", "为什么慢", "ncu 报告").
---

# Skill: CUDA Kernel Profiling (Nsight Compute — Blackwell / Hopper)

**When to use:** user asks to profile a CUDA kernel, analyze its performance, find its bottlenecks, or write an optimization plan based on Nsight Compute data. Triggers include: "profile X", "为什么这个 kernel 慢", "ncu report 说...", "下一步怎么优化", "帮我看一下这份 ncu 报告".

**Supported target hardware:**

| GPU | Architecture | CC | SMs | HBM BW | ncu chip | Companion doc |
|---|---|---|---|---|---|---|
| B200 | Blackwell | sm_100 / CC 10.0 | 148 | 8 TB/s | `gb202` | [`blackwell-cuda-programming.md`](blackwell-cuda-programming.md) |
| B300 | Blackwell Ultra | sm_103 / CC 10.3 | 160 | 8 TB/s | `gb300` | [`blackwell-cuda-programming.md`](blackwell-cuda-programming.md) |
| H200 | Hopper | sm_90 / CC 9.0 | 132 | 4.8 TB/s | `gh100` | [`hopper-cuda-programming.md`](hopper-cuda-programming.md) |
| H800 | Hopper | sm_90 / CC 9.0 | 132 | ~2.0 TB/s | `gh100` | [`hopper-cuda-programming.md`](hopper-cuda-programming.md) |

Most advice in this skill is **architecture-agnostic**. Architecture-specific notes are explicitly marked with 🔵 **Hopper** or 🟠 **Blackwell**.

---

## Golden rule

**Profile → Diagnose → Plan, in that order. Never guess.**

Most under-performing CUDA kernels are under-performing for exactly one reason that ncu can tell you in 10 seconds. Don't invent hypotheses before you have the report. Don't start coding a fix before you've matched the observed pattern to a known diagnosis. Don't write a wall of suggestions — rank them by evidence and expected impact.

---

## Quickstart (what to do when someone says "profile this kernel")

0. **Create a new run directory first** under `profile/<run_name>/` at the repo root — **one directory per run**, never reuse an existing one. Each run contains its own `harness/`, `reports/`, `analysis/`, and `REPORT.md`. This rule is mandatory in this repo. See [`reference/00-directory-layout.md`](reference/00-directory-layout.md).

1. **Decide what you're profiling.** What inputs? Which dispatch path? What question do you want answered? If the kernel takes variable-sized inputs (variable seq lengths, variable batch sizes), you must pick specific representative shapes from the user's workload — don't profile with arbitrary inputs.

2. **Build a standalone harness** unless the user is profiling through their existing binary. Harnesses compile in seconds, run the kernel in isolation, and let you use `-lineinfo` cleanly so ncu can map SASS back to source. Compile into `profile/<run_name>/harness/`. See [`reference/02-harness-guide.md`](reference/02-harness-guide.md) and the template in [`helpers/harness_template.cu`](helpers/harness_template.cu).

3. **Run two profiles**: `--set full` (with `PmSampling` sections) for the overview, and `--set source --section SourceCounters` for per-line stall attribution. Write outputs to `profile/<run_name>/reports/`. See [`reference/03-collection.md`](reference/03-collection.md).

4. **Parse with `ncu_report`** Python module — not by eye-balling the CLI. Write analysis outputs to `profile/<run_name>/analysis/`. Use the helpers in [`helpers/`](helpers/). See [`reference/04-python-api.md`](reference/04-python-api.md).

5. **Work through the six analysis dimensions.** See [`reference/05-analysis-dimensions.md`](reference/05-analysis-dimensions.md). Every one matters, but on any given kernel only 1–2 will dominate.

6. **Match patterns to the diagnosis playbook.** See [`reference/06-diagnosis-playbook.md`](reference/06-diagnosis-playbook.md). It maps NCU signal → likely cause → concrete fix, with example counts for "how big is this".

7. **Write the report** at `profile/<run_name>/REPORT.md` with evidence-backed recommendations, ranked by expected impact. See [`reference/07-report-template.md`](reference/07-report-template.md).

---

## File index

### Reference docs (read these when you need details)

| File | Purpose |
|---|---|
| [`reference/00-directory-layout.md`](reference/00-directory-layout.md) | **Read first.** Directory / naming conventions — one run = one subdirectory, no cross-contamination |
| [`reference/01-workflow.md`](reference/01-workflow.md) | End-to-end checklist from "user request" to "final report" |
| [`reference/02-harness-guide.md`](reference/02-harness-guide.md) | When and how to build a standalone harness (mandatory for TVM-FFI, PyTorch kernels, JIT-compiled code) |
| [`reference/03-collection.md`](reference/03-collection.md) | ncu command recipes: full, source-level, PM sampling, custom sections |
| [`reference/04-python-api.md`](reference/04-python-api.md) | `ncu_report` Python API patterns with copy-pasteable code |
| [`reference/05-analysis-dimensions.md`](reference/05-analysis-dimensions.md) | Six analysis dimensions: occupancy, balance, stalls, tensor core, timeline, memory |
| [`reference/06-diagnosis-playbook.md`](reference/06-diagnosis-playbook.md) | Pattern → diagnosis → fix. Merges Blackwell programming principles with NCU signals |
| [`reference/07-report-template.md`](reference/07-report-template.md) | How to structure the final report |
| [`reference/08-gpu-metric-names.md`](reference/08-gpu-metric-names.md) | Metric names by architecture: sm_90 (Hopper) vs sm_100/sm_103 (Blackwell) vs older GPUs |
| [`reference/09-common-issues.md`](reference/09-common-issues.md) | Permissions, PM sampling gaps, TVM-FFI / PyTorch gotchas |

### Helpers (reusable code)

| File | Purpose |
|---|---|
| [`helpers/harness_template.cu`](helpers/harness_template.cu) | Standalone harness template — paste your kernel, fill in input allocation, done |
| [`helpers/safetensors_loader.h`](helpers/safetensors_loader.h) | Header-only safetensors reader (no external deps) for loading real workload tensors |
| [`helpers/analyze_reports.py`](helpers/analyze_reports.py) | Extract key metrics, produce side-by-side comparisons |
| [`helpers/extract_stall_hotspots.py`](helpers/extract_stall_hotspots.py) | Per-line stall aggregation via `action.source_info(pc)` |
| [`helpers/plot_timeline.py`](helpers/plot_timeline.py) | ASCII PM-sampling timeline plotter — makes tail effect visible |
| [`helpers/list_flashinfer_workloads.py`](helpers/list_flashinfer_workloads.py) | Browse a flashinfer-trace dataset — shape histograms, filter by axis, resolve safetensors paths for specific UUIDs |
| [`helpers/ncu_utils.py`](helpers/ncu_utils.py) | Shared Python helpers: safe metric access, per-instance extraction, report loading |

---

## Critical lessons (don't skip)

1. **Metric names differ by architecture.** Names like `smsp__inst_executed_op_global_ld.sum` and `dram__bytes.sum` return `None` on Blackwell (sm_100/sm_103); the correct names have a `smsp__sass_` prefix or are split into `_read`/`_write` suffixes. Hopper (sm_90) uses the older names. Always check [`reference/08-gpu-metric-names.md`](reference/08-gpu-metric-names.md) first, or enumerate via `action.metric_names()`. Use `ncu --query-metrics --chip <chip>` to query all available names for a given chip.

2. **Always compile with `-lineinfo`.** Without it, ncu's source view is blank and you cannot do per-line stall analysis. If you can't add `-lineinfo` to the build system (TVM-FFI, PyTorch inline, JIT), **build a standalone harness** — that's the whole point.

3. **PM sampling is the only way to see tail effects.** Static metrics average over the whole kernel; only the time-series (either `pmsampling:` metrics or the ASCII plotter in `helpers/`) shows the shape of utilization over time.

4. **Load-imbalance on variable-length inputs is often the #1 bottleneck.** If the user's workload has sequences of varying length, per-SM active-cycle variance will often dwarf every other effect. Always check the input distribution.

5. **NCU's rule engine (`--page details`) already does half the work.** Each rule comes with `Est. Speedup: X%`. Read them first — they often point straight at the answer.

6. **Don't delegate understanding.** Run the profiles yourself, open the reports, cite specific metric values. Never write "the profile shows it's memory-bound" — instead, name the two or three metric values that back your conclusion (e.g., "`dram__bytes_read.sum.pct_of_peak_sustained_elapsed` well under 10%, and `long_scoreboard` stalls dominate the pcsamp histogram, so the kernel is **latency-bound on L1**, not DRAM-bandwidth-bound"). Fill in the actual numbers from your report. Specificity is the deliverable.

---

## Related skills

- [`blackwell-cuda-programming.md`](blackwell-cuda-programming.md) — Blackwell-specific programming principles and checklists (B200 / B300, sm_100/sm_103). Use it when proposing *new* kernel designs for Blackwell.
- [`hopper-cuda-programming.md`](hopper-cuda-programming.md) — Hopper-specific programming principles (H200 / H800, sm_90). Use it when proposing *new* kernel designs for Hopper.
