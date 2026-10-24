// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/kernels/work_group.hpp - the largest work-group a kernel is launched with, for the device compile.
//
// AMD's device compile sizes a kernel's registers by the work-groups it may run in: without a bound it takes 1024
// work-items, which on a wave64 GPU (gfx906) leaves 64 VGPRs a work-item and spills the rest to scratch memory, however
// small the work-groups it is launched with (2,298 of the engine's 4,007 kernels spilled there).  NVIDIA's compile
// takes the bound as reqntid / maxntid, Intel's as the kernel's work-group size.
//
// A kernel lambda launched with a fixed local range takes it as [[sycl::reqd_work_group_size(...)]] (the range's
// sizes, in its order), which intel/llvm v7.2.0 gives an AMD kernel as its flat work-group size range itself.  A
// kernel functor's get(properties_tag) gives sycl::ext::oneapi::experimental::max_work_group_size<...> when its
// launches take less than the bound: intel/llvm passes it on to AMD's code with the fix hip-flat-work-group-size
// (third_party/main/intel-llvm/patches).  v7.2.0 removed the intel::max_work_group_size attribute (ignored as unknown),
// and a lambda takes no properties of its own.
#pragma once

#include "strata/kernels/warp32.hpp"   // STRATA_HW_SUB_GROUP

#include <sycl/sycl.hpp>

#include <cstddef>
#include <utility>

namespace strata::kernels {

// A kernel lambda F of a D-dimensional range whose registers are sized for OCC of its work-groups of X work-items on a
// compute unit at once: the bound is OCC * X (the launch takes X; NVIDIA's compile sizes its registers by maxntid
// alike).  For a kernel whose local memory lets OCC work-groups share a compute unit but whose registers, sized for
// one, would not (intel/llvm takes no occupancy hint for AMD: min_work_groups_per_cu is NVIDIA's only).  The lambda
// carries no sub-group attribute of its own: SG is the warp it is written for, its sub-group size a property beside the
// bound, STRATA_HW_SUB_GROUP(SG) there (64 on a wave64 GPU) and not in the type, which the host compile and every
// device compile must name alike.
template <int D, std::size_t X, int OCC, int SG, class F>
struct OccupancyKernel {
    F f;
    void operator()(sycl::nd_item<D> it) const { f(it); }
    auto get(sycl::ext::oneapi::experimental::properties_tag) const {
        namespace syclex = sycl::ext::oneapi::experimental;
        constexpr std::size_t bound = (std::size_t) OCC * X;
        constexpr int sg = STRATA_HW_SUB_GROUP(SG);
        if constexpr (D == 1)
            return syclex::properties{syclex::sub_group_size<sg>, syclex::max_work_group_size<bound>};
        else if constexpr (D == 2)
            return syclex::properties{syclex::sub_group_size<sg>, syclex::max_work_group_size<1, bound>};
        else
            return syclex::properties{syclex::sub_group_size<sg>, syclex::max_work_group_size<1, 1, bound>};
    }
};

template <int D, std::size_t X, int OCC, int SG, class F>
OccupancyKernel<D, X, OCC, SG, F> occupancy_kernel(F f) {
    return {std::move(f)};
}

}  // namespace strata::kernels
