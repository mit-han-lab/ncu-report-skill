# Common Issues & Gotchas

Collected solutions for the recurring frustrations of profiling CUDA kernels.

---

## ncu permissions

### `ERR_NVGPUCTRPERM: The user does not have permission to access NVIDIA GPU Performance Counters on the target device`

Two solutions:

**A) Use sudo (simplest on dedicated servers):**
```bash
sudo ncu [...]
```

**B) Make it persistent (preferred on shared servers):**
```bash
sudo sh -c 'echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" > /etc/modprobe.d/ncu.conf'
sudo update-initramfs -u
# reboot, then regular user can run ncu
```

### `Could not deploy stock section files to "/home/USER/Documents/NVIDIA Nsight Compute/..."`

Set `HOME` to a writable directory:
```bash
export HOME=/any/writable/path
ncu [...]
```

This warning is harmless but noisy. ncu falls back to reading from the CUDA install dir.

---

## `-k "regex:..."` matches nothing

1. **Use the demangled name.** Templates produce something like `void my_kernel<(int)8, (int)256>(...)`. Check:
   ```bash
   cuobjdump --dump-function-names ./my_harness
   ```
   Match against the visible string.
2. **Escape regex metacharacters.** `<` and `>` don't need escaping in most regex flavors, but be careful with parentheses.
3. **The kernel might not have been launched.** Run the harness without ncu and confirm the kernel actually runs.

---

## Source view is empty / `action.source_info(pc)` returns None

The binary was compiled without `-lineinfo`. Add it to the nvcc invocation:
```bash
nvcc -O2 -std=c++17 -lineinfo -gencode=... kernel.cu -o harness
```

For JIT / framework-integrated builds:

- **TVM-FFI**: hard to inject `-lineinfo`. Easiest fix: build a standalone harness (see `02-harness-guide.md`).
- **PyTorch `torch.utils.cpp_extension.load`**: pass `extra_cuda_cflags=["-lineinfo"]`. But also switch out of `-O3 -G` to avoid heavy debug instrumentation.
- **CUTLASS**: pass `-lineinfo` via `CMAKE_CUDA_FLAGS`.
- **Triton**: harder — Triton's JIT codegen ignores user nvcc flags. To do source-level on Triton, dump the generated PTX, rebuild as a standalone, and profile.

---

## PM sampling returns nothing

1. **You didn't request it.** Add `--section PmSampling --section PmSampling_WarpStates` to the ncu invocation.
2. **vGPU / MIG environment.** PM sampling isn't supported under virtualization. Use metric-based (non-timeline) analysis only.
3. **Kernel too short.** Kernels under ~20 µs produce few PM samples; what comes back is dominated by warmup noise.
4. **Specific PM metric just isn't available on your GPU / driver / ncu combination.** Some `pmsampling:sm__throughput.*` or `pmsampling:dram__throughput.*` variants may return empty instance arrays even when other `pmsampling:smsp__warps_issue_stalled_*` series work fine on the same report. Always check `m.num_instances() > 0`; if the SM/DRAM timeline is empty, the stall-reason timelines are a reliable proxy.

---

## `No kernels were profiled`, or the profile shows a cold cache the timing loop never sees

1. **The kernel launches once** (persistent / cooperative kernels that loop internally), so `-s 2 -c 1` skips it. Use `-s 0`.
2. **Kernel replay defeats cross-launch residency.** The default replay restores memory and flushes L2 between passes, so a kernel that relies on data left in L2 by earlier launches reports DRAM traffic and hit rates that contradict the event timing (1.5 % L2 hit under kernel replay, 93 % in reality for a 16.8 MB matrix reused every step). Use `--replay-mode application --cache-control none` (Recipe 4b in `03-collection.md`).
3. **Four iterations is not steady state for a persistent kernel.** With `-c 1` on a single launch the one-time setup (loading a resident working set into shared memory) dominates the stall samples. Either profile a longer launch or subtract the setup with an iteration-scaling run in the timing harness.

---

## ncu takes forever to finish

1. **`--set full` needs 45+ replay passes.** That's normal — each pass reruns the kernel for a different metric group. If your kernel takes 3 ms, full profile takes ~15 s; 3 ms → 300 ms kernels are miserable. Mitigation: profile a smaller representative workload if possible.
2. **Kernel launches aren't isolated.** If your binary does other expensive work (data loading, CUDA context init) before every kernel launch, that runs on every replay too. Move it outside the profile window (ncu only profiles kernel launches matching `-k`).
3. **Don't use `-G` (debug).** It regresses performance ~100× and is useless for perf profiling.

---

## Kernel crashes / produces NaN only under ncu

1. **Profiler clock jitter.** Between replays, ncu resets GPU state. Kernels that depend on specific uninitialized values (bad practice) can behave differently. Fix: initialize all inputs.
2. **Out-of-order pre-replay memory**: ncu saves/restores GPU memory between replays but that can expose latent bugs like reading uninitialized global memory.

---

## Metric returns `None`

1. **Wrong metric name.** See `08-b200-metric-names.md`. Many stock docs use names that don't exist on B200.
2. **Metric not in the collected sections.** Add the relevant `--section` or `--set`.
3. **Value is legitimately missing.** Some metrics (like tensor pipe counters) return 0 rather than None when the feature wasn't used; but others return None for hardware not present.

Always wrap metric reads in a helper that returns a default:
```python
def safe(action, name, default=None):
    try:
        return action[name].value()
    except Exception:
        return default
```

---

## `ncu_report` import fails

```bash
find /usr/local/cuda* -name "ncu_report*" -type f 2>/dev/null
# e.g. /usr/local/cuda-13.2/nsight-compute-2026.1.0/extras/python/ncu_report.py
export PYTHONPATH=$PYTHONPATH:/usr/local/cuda-13.2/nsight-compute-2026.1.0/extras/python
python3 -c "import ncu_report; print('OK')"
```

If there's still an `ImportError`, check that the module is compatible with your Python version. The `_ncu_report*.so` compiled extension alongside `ncu_report.py` is built for one specific Python version.

---

## TVM-FFI specific

### "I can't profile my kernel, it's built by `tvm_ffi.cpp.build`"

- Locate the cached `.so` and `kernel.cu`:
  ```
  ~/.cache/flashinfer_bench/cache/tvm_ffi/<solution_hash>/
  ```
- You'll see `build.ninja` with the nvcc invocation — note it does *not* include `-lineinfo`.
- **Workaround:** build a standalone harness from the cached `kernel.cu` (see `02-harness-guide.md`).
- **Alternative:** inject `-lineinfo` into `build.ninja` and recompile manually. But this breaks on the next TVM rebuild.

### "The Python benchmarking script runs but ncu sees no matching kernel"

`-k "regex:..."` must match the demangled name. TVM-FFI wraps kernels in `__global__ void kernel(...)` with a fixed name — check with `cuobjdump --dump-function-names <path to .so>`.

---

## PyTorch specific

### Profiling a PyTorch model's kernel

1. Identify the kernel. Use `torch.profiler` to name it, or look for the generated kernel from `torch.compile`.
2. The kernel is often auto-generated (Triton, CUTLASS, cuDNN). For Triton kernels specifically, Triton recompiles between runs and the kernel name changes. Profile a single script invocation.
3. For `torch.compile`-generated Triton, inspect `TORCH_LOGS=+dynamo` or `TORCH_COMPILE_DEBUG=1` to see the emitted code.

### Profiling CUDA Graph-captured kernels

ncu handles CUDA Graph launches fine — each captured kernel shows up as a separate "kernel launch". Use `-k` regex + `-c N` to target.

---

## Reproducibility

### Results jitter between runs

1. **Lock GPU clocks:**
   ```bash
   sudo nvidia-smi -lgc <boost_clock_mhz>    # check with nvidia-smi -q -d CLOCK
   # profile
   sudo nvidia-smi -rgc                      # unlock
   ```
2. **Enable persistent mode** (avoids driver unload between invocations):
   ```bash
   sudo nvidia-smi -pm 1
   ```
3. **Pin the CUDA stream** explicitly rather than relying on default stream.

### Reports don't match colleague's results

- Check ncu version (`ncu --version`). Metric names change between major versions.
- Check GPU driver version (`nvidia-smi`). Some metrics only exist on certain drivers.
- Check exact nvcc invocation — a stray `-G` or missing `-lineinfo` makes a big difference.

---

## Output interpretation

### "`sm__throughput = X%`, is that good?"

It depends on the kernel type:
- GEMM / matmul: should be 50%+ on B200. Below 30% is bad.
- Element-wise / reduction: usually 10-30%, because they're DRAM-BW-bound.
- Attention / recurrence kernels: varies wildly; compare against a reference implementation.

Always check Speed-of-Light alongside: `dram__bytes_read.sum.pct_of_peak_sustained_elapsed`. If DRAM is saturated, low SM throughput is expected and OK. If DRAM is idle AND SM is idle, you're latency-bound.

### "The details page says `Est. Speedup: X%` — is that reliable?"

Yes, mostly. NCU's rule engine does a reasonable job estimating individual rule impact. Caveats:

- The sum of all `Est. Speedup`s is usually > 100%, because rules overlap (fixing A might also help B). Don't add them.
- Rules are per-pattern; the rule engine doesn't know which one is hardest/easiest to fix in your codebase.
- Use `Est. Speedup: X%` to rank patterns by magnitude; use your judgement for ease of implementation.
