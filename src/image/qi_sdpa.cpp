// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/qi_sdpa.cpp - see include/strata/image/qi_kernels.hpp (qi_sdpa): the image attention's unmasked rows through
// a GPU maker's kernel, the tensors' strides the engine's [rows][heads * head_dim] as they are, a plan built once a shape
// (the query rows, the key rows) and kept:
//   * Intel GPUs: oneDNN's SDPA through its graph API (Q . K^T, the scale, the softmax, . V fused into one partition);
//   * NVIDIA GPUs (contrib-llvm): cuDNN's SDPA through cudnn-frontend, cuDNN opened at the first attention
//     (the user's installation, not free software: open_cudnn), on the queue's CUDA stream;
//   * AMD GPUs: aotriton's flash attention (MIT; its library and its kernels' images beside the engine), on the
//     queue's HIP stream.
// A GPU or a build without one, or a failure, leaves it to the engine's own kernel.

#include "strata/image/qi_kernels.hpp"

#include "qi_sdpa.hpp"
#include "strata/core/runtime.hpp"

#include <sycl/sycl.hpp>

namespace strata::image {

bool qi_sdpa(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t q0, int64_t q1, int64_t T,
             int64_t heads, int64_t head_dim, uint16_t* out16, void* stream) {
    if (q1 <= q0) return true;
    sycl::queue& q = core::Runtime::get().stream(stream);
    [[maybe_unused]] const sycl::backend be = q.get_device().get_backend();
#if STRATA_AOTRITON
    if (be == sycl::backend::ext_oneapi_hip) return sdpa::aotriton(q, q16, k16, v16, q0, q1, T, heads, head_dim, out16);
#endif
#if STRATA_CUDNN
    if (be == sycl::backend::ext_oneapi_cuda) return sdpa::cudnn(q, q16, k16, v16, q0, q1, T, heads, head_dim, out16);
#endif
#if STRATA_ONEDNN
    // oneDNN's GPU kernels are Intel's: its Level Zero and OpenCL devices
    if (be == sycl::backend::ext_oneapi_level_zero || be == sycl::backend::opencl)
        return sdpa::onednn(q, q16, k16, v16, q0, q1, T, heads, head_dim, out16);
#endif
    (void) k16, (void) v16, (void) T, (void) heads, (void) head_dim, (void) out16, (void) q16;
    return false;
}

}  // namespace strata::image
