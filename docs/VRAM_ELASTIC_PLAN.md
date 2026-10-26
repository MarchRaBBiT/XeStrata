<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# Plan for vram_elastic on HIP

English | [日本語](VRAM_ELASTIC_PLAN.ja.md)

Status: stopped at step 1 (2026-10-26).
On AMD GPUs (ROCm), virtual memory mapped again with other physical memory at an address that was unmapped can send the
data written there to other memory, so `vram_elastic` on HIP does not go ahead.
What was checked is in [Step 1 results](#step-1-results), and what it takes to start again in
[Conditions for resuming](#conditions-for-resuming). Steps 2 to 6 were not carried out.

The plan was to enable the existing `vram_elastic` on the SYCL HIP path and in resident mode, so the expert cache
could give VRAM back for images and take it again without restarting the text engine.
Current usage is in [DETAILS](DETAILS.md#leaving-vram-for-other-programs).

The machines, GPUs, toolchains, containers, settings, service operations and capability query results are in
`LOCAL.md` at the repository root. This plan holds no machine-specific values.
Read existing measurement records first; measure only the differences needed for a new binary or setting,
and keep them under `<data>/records/` (`<data>`: `LOCAL.md`).

## Purpose and scope

- Shrink and restore the cache between requests through `POST /v1/vram`.
- Use the existing experts in RAM and files in the resident modes that keep a RAM copy too (`--resident-experts`,
  the automatic switch when RAM is low, `--resident-budget-gib` and `--resident-cpu-experts`).
- For image input, run the encoder only while needed through `vision.elastic`.
- Apply image generation's `gpu_policy: shrink` only after checking its capacity and speed separately from image input.

New APIs or configuration keys, multiple GPUs and remote caches, continuous VRAM monitoring that shrinks the cache
on its own, and forcing a copy of every expert into RAM are outside this change.

## Step 1 results

On a Radeon VII (gfx906), a probe called HIP's `hipMem*` and, below it, ROCr's `hsa_amd_vmem_*` directly, and did the
cache's operations through a SYCL queue.
ROCm was 7.14 (a build that includes gfx906), the kernel Linux 7.0, and the SYCL side intel/llvm v7.2.0.
v7.2.0's UR has no HIP VMM, so XeStrata's path (`sycl_ext_oneapi_virtual_mem`) was not exercised.

What worked:

- The capability query reports support; the granularity is 4 KiB minimum and 2 MiB recommended. A map at a nonzero
  offset fails with `hipErrorInvalidValue` (XeStrata maps at offset 0).
- At addresses never mapped before, copies to and from pinned and pageable host memory, copies to and from device
  memory, memset, fill, kernel reads and writes and graph replay all matched.
  Copies, memsets and kernels across two physical allocations matched too.
- `sycl::get_pointer_type` returns device for a mapped address and unknown for one that is only reserved (UR's assert
  is not hit).
- After `hipMemRelease` on mapped memory, the memory stays usable until the unmap and is released there.
- After 20 shrink and restore cycles on four 512 MiB segments, neither HIP nor the kernel showed a leak (with 100 ms
  between the shrink and the restore).
- The speed was that of ordinary device memory (host copies 7.1 GB/s, a kernel read of 2 GiB 871 GB/s).
  Creating, mapping and setting access for a 512 MiB segment took 0.1 ms, unmapping and releasing it 0.2 ms.
- With the VRAM used up, `hipMemCreate` returns `hipErrorOutOfMemory` and the earlier allocations keep their contents.

Why it stopped:

- Other physical memory mapped at an unmapped address and written at once: the writes go to the physical memory mapped
  there before (6 to 10 of 10 tries at 2 MiB).
  With 1 ms or more between the unmap and the map, it did not happen.
- After that address is written, other physical memory mapped at the next address can leave the first address pointing
  at the new memory: what was written at the first address is lost, and it reads and writes the same memory as its
  neighbour (0 to 4 of 10 tries at 2 MiB; at 512 MiB, 1 of 5 even with 100 ms between the unmap and the map).
- Calling ROCr's `hsa_amd_vmem_*` directly gives the same, so the fault is below the HIP runtime (ROCr's virtual memory
  or the kernel driver).
  Neither a UR patch nor XeStrata can fix it.
- `vram_elastic`'s restore maps new physical memory at the addresses the shrink unmapped, and the following segments
  next to them.
  `--kv-grow` maps the physical memory it takes from the cache at the K/V's addresses at once, and when it gives it back
  maps other physical memory at the cache's addresses again.
  Both are this sequence: the experts or the K/V would be corrupted without an error.

What was learned about free memory (step 4's premises):

- Released VRAM shows in HIP's free figure at once and in the kernel's count 40 ms later (`hipMemRelease` and
  `hipFree` alike).
- A Vulkan encoder using about 1 GiB of the same GPU does not lower HIP's free figure.
  Allocating up to HIP's figure pushed about 870 MiB of the encoder's into GTT, and one encode went from 1.37 s to 5.5 s.
  Freeing the VRAM did not bring the speed back until the encoder restarted.

## Conditions for resuming

- When a ROCm (ROCr and the kernel driver) handles other physical memory mapped again, and a map next to it, right
  both straight after the unmap and with time between, run step 1 again; if it passes, start at step 2.
- intel/llvm's sycl branch now implements HIP VMM (c13ad6c7, #22671).
  With an intel/llvm that includes it, HIP devices report virtual memory, and with a 2 MiB granularity `--kv-grow` turns
  on in launches that do not ask for it ([DETAILS](DETAILS.md#where-the-kv-cache-lives-and-its-precision)).
  Before moving to such a version, add a check that keeps virtual memory unused with a runtime that has this fault.
- The parts that do not depend on HIP (`vram_elastic` in resident mode, capacity decisions from the kernel's free
  figure) would serve other GPUs too, but have not been started.

## Implementation evidence

| Location | Current constraint and response |
| --- | --- |
| intel/llvm v7.2.0's UR HIP adapter | The VMM capability query in `device.cpp` always returns false; `virtual_mem.cpp` and `physical_mem.cpp` are not implemented. intel/llvm's sycl branch implements them (c13ad6c7, #22671), and the change applies to the v7.2.0 tree as it is. To add them, port it through the existing numbered patches. |
| The `vram_elastic` check in [generate.cpp](../src/program/generate.cpp) | Resident mode (`resident_cpu_experts` or `resident_gib`) is refused. Relax this after testing file reads and safe restoration. |
| The `--kv-grow` check in `generate.cpp` | It looks at `resident_cpu_experts` only, and is decided before the automatic switch to resident mode when RAM is low. Once VMM works on HIP, `--kv-grow` can turn on in resident mode for a launch that did not ask for it. |
| [ExpertCache](../include/strata/core/expert_cache.hpp) | Graphs, verify plans and prefill hold cache addresses. Keep the reserved virtual addresses and attach or detach only the physical memory. |
| [FileExpertSource](../src/core/expert_source.cpp) | Existing file reads can supply experts absent from RAM. The RAM copy holds the experts the GPU does not and those of the prefill loan's slots, so when a shrink moves the loan to slots the copy does not hold, prefill reads them from the files on every chunk (upstream #669). Do not enlarge the RAM copy as the cache shrinks. |
| The grow path in `generate.cpp`'s `vram_command` | `blob_to_copy`'s temporary buffer can be reused before a queued transfer finishes. Follow the existing prefill refill's lifetime handling. |
| The capacity calculation in `generate.cpp`'s `vram_command` | The shrink and the answer's `vram_free_mib` come from `gpu::mem_info` (hipMemGetInfo on HIP). HIP's free memory does not subtract other programs' allocations on the same GPU (checked in step 1 for a Vulkan allocation). |
| [gpu.cpp](../src/core/gpu.cpp) and [expert_cache.cpp](../src/core/expert_cache.cpp) | Shrink cannot receive a failure from `vmem_unmap`. Do not report a region as released after a failed release. |
| [server.py](../serve/server.py)'s `_vision_up` and `vision_attempts` | An elastic encoder is not created at startup, so the configured fallback is tried at startup only. An encoder that fails to start in an image turn stays stopped until the server restarts. |

## Implementation steps

Steps 2 onwards were not carried out, as step 1 did not pass. They stay as the steps for when the
[conditions for resuming](#conditions-for-resuming) are met.

### 1. Check HIP VMM operations

Follow [AMD's VMM guide](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/virtual_memory.html)
with a small region first: capability, minimum and recommended granularity, address reservation, physical allocation,
mapping, access rights, reads and writes, unmapping, release, and a new allocation at the same virtual address.
Check reads and writes with the operations XeStrata performs on the cache: copies from pinned and pageable host
memory, copies back from the device to the host, memset, kernel reads and writes, and graph replay.
Also check pointer-type queries on VMM addresses (`sycl::get_pointer_type`) and what releasing still-mapped physical
memory does.
The capability flag alone is not enough to proceed: matching data and release without leaks are acceptance criteria.
If it fails, isolate the cause and keep the ordinary cache.

Compare HIP's free VRAM with the kernel's report for the same GPU, including how they follow allocations and
releases, coexistence with an external Vulkan encoder, and eviction into GTT.
Take the target PCI address and test environment from `LOCAL.md`.

Check a new allocation at the same address both straight after the unmap and with time between, with the same and
with other physical memory, and then the first address's contents after a map at the next address.
The results of 2026-10-26 are in [Step 1 results](#step-1-results).

### 2. Implement VMM in the UR HIP adapter

Port the HIP VMM implementation on intel/llvm's sycl branch (c13ad6c7, #22671).
Add `NN-hip-virtual-memory.patch` under `third_party/main/intel-llvm/patches/`, covering UR's `device.cpp`,
`physical_mem.hpp/.cpp` and `virtual_mem.cpp`.
Start the patch, as the existing ones do, with its SPDX lines and a description of what it fixes.
Check that HIP capability queries, granularity, device selection, access rights, handle reference counts, error
conversion and cleanup after partial failure are handled as in the CUDA adapter.
Devices without support must report no support.

XeStrata's core keeps using `sycl_ext_oneapi_virtual_mem`; HIP APIs stay in UR.
Derive segment sizes from the reported granularity and existing settings, without GPU-name branches.
Apply the patch to both the free and contrib-llvm toolchains following the
[intel/llvm build instructions](DEVTOOLS.md#intelllvm-from-source).
`tools/intel_llvm_build.py` does not build UR's tests, so build UR with its tests separately and run its VMM
conformance tests (`unified-runtime/test/conformance/virtual_memory`).
If they cannot be built, substitute step 1's checks and `vmm_test`, and say so in the report.
Describe the fix in the intel/llvm section of `docs/DEVTOOLS.ja.md` and `docs/DEVTOOLS.md`, Japanese first.

### 3. Connect resident mode to safe shrink and restore

Relax `generate.cpp`'s resident restriction together with an implementation that passes the following tests.
Keep the restrictions outside this change, including one GPU, serve mode and no remote caches.

- Read evicted experts through `FileExpertSource` from RAM, or from files when absent from RAM.
- Keep the RAM copy holding only the experts the GPU does not and those of the prefill loan's slots.
  When grow's refill picks a substitute for a slot the adaptive tier already refilled, do not pick an expert the RAM
  copy holds (`has_resident`).
- When a shrink moves the loan to slots the RAM copy does not hold, prefill reads their experts from the files on every chunk.
  Measure short and long prefill after a shrink and decide whether to accept that or lend fewer slots.
  Enlarge the RAM copy only if measurements show there is no other way.
- In grow, check `srcp->transient()`: use `fill_slot_blocking` for temporary buffers and queued transfers otherwise.
- Finish adaptive swaps and synchronize the GPU before changing regions; update `host_res` and device residency.
- Move the prefill loan into the remaining slots and reduce its chunk through the existing code when needed.
- Decide `--kv-grow` after the automatic switch to resident mode when RAM is low, and align its resident condition with
  the `vram_elastic` check (`resident_cpu_experts` or `resident_gib`).
  If `--kv-grow` is allowed in resident mode, show the reason and its speed in tests.
- After a partial map, unmap, transfer or synchronization failure, publish only usable regions and fully copied slots.
  Return unrecoverable device errors to the request instead of continuing inference with incomplete state.

Do not add RAM residency or cache placement tables unless the tests show they are needed.
Candidate changes are `generate.cpp`, `gpu.cpp` and its header, `expert_cache.cpp` and its header, and
`expert_source.cpp` and its header.

### 4. Align free-memory and server decisions

Establish why HIP and the kernel report different free memory and make capacity decisions account for external allocations.
`vram_command` takes the shrink and the answer's `vram_free_mib` from `gpu::mem_info`, so fix the engine's calculation first.
If supplementary Linux queries are needed, put them in `src/platform/` and match the target GPU by PCI address.
Define what happens when a report is unavailable; do not substitute zero or total capacity as measured free memory.
Check VRAM and maximum-allocation limits with VMM allocations included too.

In [server.py](../serve/server.py)'s `_vision_up` and [images.py](../serve/images.py)'s `engine_elastic`,
use the engine's effective state, not just its arguments. Do not override `INFO vram_elastic=0` with a launch argument.
A 200 response from `POST /v1/vram` does not prove enough room.
The free memory in the engine's answer comes from the engine's own query, so the server also checks the kernel's
figure that the existing [telemetry.py](../serve/telemetry.py) reads (sysfs on AMD).
Before starting the encoder, wait for room with a deadline and cancellation instead of relying only on a fixed sleep.

Add a switch to the configured fallback encoder for an image turn where the room cannot be made or the GPU encoder
does not start.
Report the switch through the existing `vision_fallback`. Without a fallback, return the reason without starting
excess allocations.
Restore the reserve and cache after a failure or cancellation too.

### 5. Pass tests and builds

Add small failing tests for the changes first, then check the conditions below after implementation.
Reuse the existing tests and fixtures that apply; cover normal and failure cases for new non-trivial branches.
Reach at least 80% coverage for added and changed host C++ and Python code.
The repository does not measure coverage yet: choose the tools and steps at the start of the implementation, and
describe any added tool in DEVTOOLS.
SYCL device code and paths that run only on a GPU are outside the measurement; the tests below check them.

| Layer | Tests and acceptance criteria |
| --- | --- |
| VMM | UR's conformance tests (step 2) and [vmm_test](../tests/core/vmm_test.cpp). No skip on the target HIP GPU; reads, writes and remapping match, and memory is released. |
| Cache | Add segmented-cache tests under `tests/core/` and in `CMakeLists.txt`. Check variable blob sizes, segment boundaries, minimum size, partial grow and consistency after failures. |
| Resident | Extend [file_expert_source_test](../tests/core/file_expert_source_test.cpp). Check the RAM complement and file reads for GGUF experts absent from it. |
| Engine integration | Send VRAM commands to an engine subprocess to check grow's temporary-buffer lifetime, restoration after adaptive swaps, that grow's refill picks no expert the RAM copy holds, and short and long prefill after shrink (resident mode included). CTest has no model fixture that runs the engine: choose the model at the start of the implementation, and record the steps and results of tests CTest cannot hold in the report. Source unit tests alone are insufficient. |
| Server | Extend `serve/` unittests for capability decisions, insufficient memory, delayed release, concurrent requests, cancellation, the switch to the fallback in an image turn and restoration failure. Distinguish a 200 response from meeting the requested reserve. |
| E2E | Cover text only, first image input, repeated image input, conversation after an image, and restoration after idle. Ordinary shrink and restore must not restart the text engine. |
| Regression | Check free and contrib-llvm builds and CTest, NVIDIA execution, reduced VRAM, smaller maximum allocations, no matrix engines, fewer CPU threads, the existing Intel GPU paths, an intel/llvm runtime without the patch (`vram_elastic` stays off), and the Windows cross build. |

Reporting VMM support automatically enables `--kv-grow` where the device's granularity is 2 MiB or more.
Include launches without `vram_elastic`, and automatic, explicit and disabled `--kv-grow`, in regression tests, both in
resident mode (the automatic switch included) and outside it.
Compare prefill and decode with the cache in VMM chunks on HIP against the cache without them; if they are slower,
decide the automatic default by a condition the device reports, never by GPU name or maker.
For memory and undefined-behavior changes, run address/undefined sanitizers and CTest in a separate build.
Update the documents (the VRAM section of DETAILS: HIP and resident mode) and the comments (the adapters named in
`gpu.hpp` and `vmm.hpp`) to match the implementation.
Before submission, run `tools/lint/run.sh`, license classification and `reuse lint`, leaving no new warnings or findings.
Record the results and any conditions that could not be exercised in the implementation report.

### 6. Roll out image input first

Start with manual shrink and restore: check `INFO vram_elastic=1`, cache size, free VRAM and resumed inference.
Then enable the configuration's `vram_elastic: true` and `vision.elastic: true`.
Choose `vision.elastic_mib` after checking the encoder's peak usage and rounding to segments;
reduce the previous permanent reserve only after confirming operation with the encoder absent.
Compare RAM, swap, GTT, first-image latency, prefill and decode, and cache restoration time.

Try image generation's `gpu_policy: shrink` only after separately checking that its required memory and the text
engine's fixed regions fit together. Keep the existing `unload` policy during initial rollout and for settings that do not fit.
VMM support alone does not establish that the whole image generation workload can coexist.

## Completion criteria and rollback

- HIP VMM follows the target device's capabilities, and `vram_elastic` works in resident mode too.
- The surviving slots keep their addresses across shrink and restore; expert contents and inference recover correctly.
- Actual capacity and residency agree after an unmet reserve, partial failure or cancellation.
- The cache reclaims the encoder's VRAM between images; ordinary image input does not restart the text engine.
- `--kv-grow`'s automatic behavior in launches that do not ask for it is as steps 3 and 5 decided.
- Launches without the feature and existing supported GPUs do not regress; required builds and tests pass.

Before rollout, save the settings and the binary/runtime pair (locations: `LOCAL.md`).
To roll back, disable `vram_elastic` and `vision.elastic`, restore the previous permanent reserve and image generation's
`unload` policy, and restart. If the runtime is affected too, restore the saved binary/runtime pair.
While shrunk, experts absent from RAM may require more file reads and slow prefill and decode, so operational speed as
well as correctness is an adoption criterion.
Once complete, describe the usage in DETAILS and remove this plan and the link to it from DETAILS.
