// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/kernels/work_group.hpp - the largest work-group a kernel is launched with, for the device compile.
//
// AMD's device compile sizes a kernel's registers by the work-groups it may run in: without a bound it takes 1024
// work-items, which on a wave64 GPU (gfx906) leaves 64 VGPRs a work-item and spills the rest to scratch memory, however
// small the work-groups it is launched with (2,298 of the engine's 4,007 kernels spilled there).  intel/llvm passes a
// kernel's max_work_group_size on as its flat work-group size range (third_party/main/intel-llvm/patches:
// hip-flat-work-group-size); NVIDIA's compile takes it as maxntid, Intel's as the kernel's largest work-group size.
//
// A launch with a larger work-group fails, so the bound is the launch's own local range: STRATA_MAX_WG(Z, Y, X) on a
// kernel lambda, with the sizes in sycl::range's order and a 1-D or 2-D range padded with 1s on the left (a 1-D kernel
// of THREADS work-items is STRATA_MAX_WG(1, 1, THREADS)); a kernel functor's get(properties_tag) gives
// sycl::ext::oneapi::experimental::max_work_group_size<...> with its range's sizes.
#pragma once

#define STRATA_MAX_WG(Z, Y, X) intel::max_work_group_size(Z, Y, X)

// A 1-D kernel of X work-items whose registers are sized for OCC of its work-groups on a compute unit at once: the bound
// is OCC * X (a launch takes X; NVIDIA's compile sizes its registers by maxntid alike).  For a kernel whose local memory
// lets OCC work-groups share a compute unit but whose registers, sized for one, would not (intel/llvm takes no occupancy
// hint for AMD: min_work_groups_per_cu is NVIDIA's only).
#define STRATA_MAX_WG_OCC(X, OCC) intel::max_work_group_size(1, 1, (OCC) * (X))
