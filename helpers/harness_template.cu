// ============================================================================
// harness_template.cu — starting point for a profiling harness.
//
// Copy this file into your run directory, e.g.:
//     profile/<run_name>/harness/my_kernel_harness.cu
// and customize the sections marked with TODO(you).
//
// Compile with (choose the line that matches your target GPU):
//
//   🟠 Blackwell B200  (sm_100, CC 10.0):
//     nvcc -O2 -std=c++17 -lineinfo \
//          -gencode=arch=compute_100,code=sm_100 \
//          my_kernel_harness.cu -o my_kernel_harness
//
//   🟠 Blackwell B300  (sm_103, CC 10.3):
//     nvcc -O2 -std=c++17 -lineinfo \
//          -gencode=arch=compute_103,code=sm_103 \
//          my_kernel_harness.cu -o my_kernel_harness
//
//   🟠 Blackwell B200 + B300 family (compute_100f runs on both sm_100 and sm_103):
//     nvcc -O2 -std=c++17 -lineinfo \
//          -gencode=arch=compute_100,code=compute_100 \
//          my_kernel_harness.cu -o my_kernel_harness
//
//   🔵 Hopper H200 / H800  (sm_90, CC 9.0):
//     nvcc -O2 -std=c++17 -lineinfo \
//          -gencode=arch=compute_90,code=sm_90 \
//          my_kernel_harness.cu -o my_kernel_harness
//
// Usage modes this template supports:
//     ./harness --workload <path.safetensors>        # real tensor values
//     ./harness <shape_arg1> <shape_arg2> ...        # synthetic at specific shape
//
// Why it's structured this way: see ../reference/02-harness-guide.md
// ============================================================================

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// Keeps the safetensors_loader header next to this file for easy relocation.
#include "safetensors_loader.h"

// ---------------------------------------------------------------------------
// CUDA error helpers
// ---------------------------------------------------------------------------
#define CUDA_CHECK(expr)                                                               \
    do {                                                                               \
        cudaError_t err = (expr);                                                      \
        if (err != cudaSuccess) {                                                      \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #expr, __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                          \
            std::exit(1);                                                              \
        }                                                                              \
    } while (0)

template <typename T>
static T* alloc_device(size_t n) {
    T* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
    return p;
}

// ---------------------------------------------------------------------------
// Synthetic fill helpers — replace scale with something sensible for your kernel
// ---------------------------------------------------------------------------
static void fill_bf16_random(std::vector<__nv_bfloat16>& h, uint64_t seed, float scale = 0.5f) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> d(-scale, scale);
    for (auto& x : h) x = __float2bfloat16(d(rng));
}
static void fill_f32_random(std::vector<float>& h, uint64_t seed, float scale = 0.5f) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> d(-scale, scale);
    for (auto& x : h) x = d(rng);
}

// ============================================================================
// TODO(you): paste / include the kernel source here
//
//   - Include any __device__ helpers, intrinsics, and constants.
//   - Use `extern "C"` or plain C++ — ncu picks up either.
//   - Add explicit template instantiations for every variant you want
//     to be able to target with `ncu -k "regex:..."`.
//
// Example:
//     template<int TILE_M, int TILE_N>
//     __global__ void my_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
//         // ... kernel body ...
//     }
//     // Explicit instantiations — without these, unused variants get stripped
//     // and ncu's regex won't find them.
//     template __global__ void my_kernel<64, 64>(const float*, const float*, float*, int, int, int);
//     template __global__ void my_kernel<128, 64>(const float*, const float*, float*, int, int, int);
// ============================================================================

// ... your kernel goes here ...

// ============================================================================
// TODO(you): launch helper
//
// Take the input pointers and shape parameters, run the kernel with the
// chosen template parameters / grid / block. Keep this simple — a single
// `kernel<<<grid, block>>>(...)` call.
// ============================================================================
static void launch_kernel(/* input pointers, shape params, cuda stream */) {
    // dim3 grid(...);
    // dim3 block(...);
    // my_kernel<TILE_M, TILE_N><<<grid, block>>>(A, B, C, M, N, K);
    // CUDA_CHECK(cudaDeviceSynchronize());
    // CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// CLI parsing + input prep. Shape-match mode fills synthetic values at an
// exact shape (e.g., from a user-specified (M, N, K) tuple). --workload mode
// loads a real safetensors file.
// ============================================================================

static void usage(const char* argv0) {
    fprintf(stderr,
            "Usage:\n"
            "  %s --workload <safetensors_path>\n"
            "      Load real tensor values from a workload file.\n"
            "  %s <shape_arg1> <shape_arg2> ...\n"
            "      Shape-match mode (synthetic at given shape).\n",
            argv0, argv0);
}

int main(int argc, char** argv) {
    if (argc < 2) { usage(argv[0]); return 2; }

    bool use_workload = false;
    std::string workload_path;

    // --- Parse arguments -----------------------------------------------------
    if (std::string(argv[1]) == "--workload") {
        if (argc < 3) { usage(argv[0]); return 2; }
        use_workload = true;
        workload_path = argv[2];
    } else {
        // TODO(you): parse your shape args from argv here.
        // E.g., int M = std::atoi(argv[1]); int N = std::atoi(argv[2]); ...
    }

    // --- Load real workload if requested ------------------------------------
    SafetensorsFile st;
    if (use_workload) {
        st = SafetensorsFile::load(workload_path);
        // TODO(you): read shapes from the safetensors header.
        //   const auto& a_entry = st.entry("A");
        //   int M = (int)a_entry.shape[0];
        //   int K = (int)a_entry.shape[1];
        fprintf(stderr, "[harness] loaded workload: %s\n", workload_path.c_str());
    }

    // --- TODO(you): allocate inputs / outputs on host & device --------------
    // std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N, 0.f);
    // float *d_A = alloc_device<float>(h_A.size());
    // float *d_B = alloc_device<float>(h_B.size());
    // float *d_C = alloc_device<float>(h_C.size());

    // --- TODO(you): fill inputs ---------------------------------------------
    // if (use_workload) {
    //     std::memcpy(h_A.data(), st.tensor_bytes("A"), h_A.size() * sizeof(float));
    //     std::memcpy(h_B.data(), st.tensor_bytes("B"), h_B.size() * sizeof(float));
    // } else {
    //     fill_f32_random(h_A, 0xA0A0ULL);
    //     fill_f32_random(h_B, 0xB0B0ULL);
    // }

    // --- TODO(you): copy inputs to device ----------------------------------
    // CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(float), cudaMemcpyHostToDevice));
    // CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(float), cudaMemcpyHostToDevice));

    // --- Launch --------------------------------------------------------------
    fprintf(stderr, "[harness] launching kernel...\n");
    launch_kernel(/* d_A, d_B, d_C, M, N, K */);
    fprintf(stderr, "[harness] done.\n");

    // --- Cleanup -------------------------------------------------------------
    // CUDA_CHECK(cudaFree(d_A));
    // CUDA_CHECK(cudaFree(d_B));
    // CUDA_CHECK(cudaFree(d_C));

    return 0;
}
