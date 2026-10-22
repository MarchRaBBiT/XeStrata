// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/kernels/warp32.hpp - the kernels' 32-lane warp on a GPU whose sub-groups are 64 wide.
//
// The kernels are written for warps of 32 lanes, as CUDA's: a kernel asks for 32-wide sub-groups, takes the lane as
// the sub-group's local id and sums with xor butterflies from 16 down.  A wave64 GPU (AMD's GCN and CDNA, gfx9) has
// only 64-wide sub-groups, so there a sub-group holds two of those warps, one in each half (as upstream's gfx906 build
// maps a CUDA warp to half of the wavefront).  The functions here give the warp's view of the sub-group: the lane in
// the warp, the warp's index in the work-group and the collectives within the warp.  An xor shuffle by 16 or less
// stays in its half, so permute_group_by_xor needs nothing.  On a GPU with 32-wide sub-groups each one is the
// sub-group's own call, unchanged.
#pragma once

#include <sycl/sycl.hpp>

// 1 in the device compile for a wave64 GPU, 0 in every other compile (the host's included).
#if defined(__SYCL_DEVICE_ONLY__) && defined(__AMDGCN__) && defined(__GFX9__)
#define STRATA_WAVE64 1
#else
#define STRATA_WAVE64 0
#endif

// The sub-group size a kernel written for SG-wide warps asks for in this device compile: 64 on a wave64 GPU, whose
// sub-groups then hold 64 / SG of them (SG is 32 wherever a kernel reaches such a GPU).
#define STRATA_HW_SUB_GROUP(SG) (STRATA_WAVE64 ? 64 : (SG))

namespace strata::kernels::warp32 {

constexpr int kLanes = 32;

// The work-item's lane in its warp, 0..31.
inline int lane(const sycl::sub_group& sg) {
    const int l = (int) sg.get_local_linear_id();
    return STRATA_WAVE64 ? (l & (kLanes - 1)) : l;
}

// The warp's index in the work-group: the sub-group's index, or on a wave64 GPU two warps a sub-group.  The sub-groups
// of a work-group are its work-items in order, 64 at a time, so warp w is the work-items [32 w, 32 w + 32) as on a
// 32-wide GPU.
inline int index(const sycl::sub_group& sg) {
    const int g = (int) sg.get_group_linear_id();
    return STRATA_WAVE64 ? 2 * g + ((int) sg.get_local_linear_id() >> 5) : g;
}

// The sub-group's id of lane `src` of this work-item's warp.
inline int at(const sycl::sub_group& sg, int src) {
    return STRATA_WAVE64 ? (((int) sg.get_local_linear_id() & ~(kLanes - 1)) | src) : src;
}

// select_from_group within the warp: `v` of lane `src`.
template <typename T>
inline T shfl(const sycl::sub_group& sg, T v, int src) {
    return sycl::select_from_group(sg, v, (size_t) at(sg, src));
}

// group_broadcast within the warp.
template <typename T>
inline T bcast(const sycl::sub_group& sg, T v, int src) {
#if STRATA_WAVE64
    return shfl(sg, v, src);
#else
    return sycl::group_broadcast(sg, v, src);
#endif
}

// reduce_over_group within the warp.  On a wave64 GPU an xor butterfly from 16 down, so every lane gets the same sum.
template <typename T, typename Op>
inline T reduce(const sycl::sub_group& sg, T v, Op op) {
#if STRATA_WAVE64
    for (int o = kLanes / 2; o > 0; o >>= 1) v = op(v, sycl::permute_group_by_xor(sg, v, o));
    return v;
#else
    return sycl::reduce_over_group(sg, v, op);
#endif
}

// any_of_group within the warp.
inline bool any(const sycl::sub_group& sg, bool p) {
#if STRATA_WAVE64
    return reduce(sg, (int) p, sycl::bit_or<int>()) != 0;
#else
    return sycl::any_of_group(sg, p);
#endif
}

// inclusive_scan_over_group within the warp (Hillis and Steele on a wave64 GPU).
template <typename T, typename Op>
inline T inclusive_scan(const sycl::sub_group& sg, T v, Op op) {
#if STRATA_WAVE64
    const int l = lane(sg);
    for (int d = 1; d < kLanes; d <<= 1) {
        const T u = shfl(sg, v, l >= d ? l - d : l);
        if (l >= d) v = op(u, v);
    }
    return v;
#else
    return sycl::inclusive_scan_over_group(sg, v, op);
#endif
}

// exclusive_scan_over_group within the warp, for sycl::plus (its identity is T(0)).
template <typename T>
inline T exclusive_scan_plus(const sycl::sub_group& sg, T v) {
#if STRATA_WAVE64
    return inclusive_scan(sg, v, sycl::plus<T>()) - v;
#else
    return sycl::exclusive_scan_over_group(sg, v, sycl::plus<T>());
#endif
}

}  // namespace strata::kernels::warp32
