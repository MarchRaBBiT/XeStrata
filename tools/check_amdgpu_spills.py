#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Check that no AMD GPU kernel of a linked binary spills its registers to scratch memory, beyond the ones listed here.

AMD's device compile gives a kernel the registers its work-groups leave it: a kernel bound to 256 work-items has 256
VGPRs on gfx906, one with no bound 64, and what does not fit goes to scratch memory, a slow path the profile shows only
as a slower kernel (include/strata/kernels/work_group.hpp: 2,298 of the engine's 4,007 kernels spilled on gfx906 before
the kernels' bounds reached the compile).  The check reads the code objects of every AMD target in the binaries (the
SYCL device images that clang-offload-extract takes out, offload bundles of amdgcn code objects) and the AMDGPU
metadata of each kernel (its VGPR and SGPR spill counts, its scratch bytes, its work-group bound), and fails when a
kernel spills that ALLOWED does not name for its target.  A target ALLOWED does not list has not been looked at: its
spilling kernels are printed and do not fail the check, until its entry (empty if none spills) is added.

    check_amdgpu_spills.py [--extract clang-offload-extract] BINARY ...
    check_amdgpu_spills.py --list BINARY ...     every spilling kernel, allowed or not

Exit 0 = clean, 1 = a kernel spills that is not allowed, 77 = cannot run here (no clang-offload-extract, or no AMD code
object in the binaries; ctest SKIP_RETURN_CODE).
"""
from __future__ import annotations

import argparse
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

# Kernels that may spill, by AMD target (the "gfx..." name, without features such as ":xnack-") and a regular
# expression searched in the kernel's mangled name, each with the reason.  A kernel that spills on a target and matches
# none of its target's entries fails the check.  The RDNA targets are the packages' (measured with intel/llvm 7.1.1),
# gfx906's with v7.2.0.
_XE2_ONLY = {
    r"GdnRecLane": "Xe2's: gdn_lane_ok takes it where the EU is 16 wide, so no AMD GPU runs it",
}
_KEY_HEAD = {
    r"GdnRecKh": "the key-head form keeps 3 x 32 state rows a work-item in registers (128 work-items), and the prompt "
                 "path times it against D-2 at its first call and takes the faster (src/prefill/kernels.cpp)",
}
_PA_WMMA = ("prompt_attn.cpp's WMMA attention: bounded at its launch's W x 32 work-items, and the tile still takes "
            "more than 256 VGPRs; not looked at yet")
_RDNA3 = {
    r"WmmaKernelILi256ELi8ELi8E": _PA_WMMA,
    r"AttnKernelILi3E": "the image engine's RDNA attention: bounded at its largest launch (8 sub-groups), and it still "
                        "takes more than 256 VGPRs; not looked at yet",
    r"6KernelILi(12|20|23|42)ELi2ELi2ELb0E|6KernelILi6ELi2ELi1ELb0E":
        "iq_mmq.cpp's int8 WMMA tiles: bounded at their launch's work-group (256 or 128 work-items), and the tile "
        "still takes more than 256 VGPRs; not looked at yet",
}
_RDNA4 = {
    r"WmmaKernelILi256ELi12ELi4E": _PA_WMMA,
    r"6KernelILi12ELi2ELi2ELb0E": "iq_mmq.cpp's int8 WMMA tile of Q4_K: bounded at its launch's 256 work-items, and "
                                  "the tile still takes more than 256 VGPRs; not looked at yet",
}
ALLOWED: dict[str, dict[str, str]] = {
    "gfx906": {
        **_XE2_ONLY,
        r"GdnRecKh": _KEY_HEAD[r"GdnRecKh"] + "; on gfx906 it is the faster (7,443 tokens: 17.6 against 18.7 ms)",
        r"gdn_step_plainILb[01]E|gdn_step_prefetchILb[01]E":
            "512 work-items a work-group (S x RG) leave 128 VGPRs a work-item: 30-52 spill with intel/llvm v7.2.0 "
            "(v7.1.1: gdn_step_plain<true> alone), and Swift 1.5 IQ3_XXS decoded as fast (40.5 tok/s against 41.4 and "
            "40.0, 2026-10-25); not looked at yet",
        r"GdnILi16ELi(32|48)EE10recurrence": "the prompt's recurrence, 512 work-items (S x RG), 128 VGPRs: 24 spill with "
                                             "intel/llvm v7.2.0, and Qwen3.6's 6.3K-token prefill was as fast (748-754 "
                                             "tok/s against 746-749); not looked at yet",
        r"launch_chunkILi256ELi12ELi[0-3]E": "sized for two work-groups a compute unit (occupancy_kernel<..., 2, ...>): "
                                             "3-14 VGPRs spill with intel/llvm v7.2.0 (v7.1.1: one, the first KV form "
                                             "alone), and the prompt's attention took 2,654 -> 1,979 ms",
        r"dequant_kernelILi(14|1481834497)EfE": "Q6_K's dequantization to FP32, 32 values a work-item: a range kernel, "
                                                "whose work-group the runtime chooses (no bound); 45-61 spill with "
                                                "intel/llvm v7.2.0; not looked at yet",
        r"mf_vae_detail9pixel_mlp": "Mage-Flow's VAE pixel MLP (upstream's): a range kernel, whose work-group the "
                                    "runtime chooses (no bound); not looked at yet",
        r"RowsI8": "the image engine's W8A8 rows (upstream's): a work-group a row, up to the device's largest (its "
                   "groups of 256 values across it); 6 spill; not looked at yet",
    },
    "gfx1030": {**_XE2_ONLY, **_KEY_HEAD},
    "gfx1100": {**_XE2_ONLY, **_KEY_HEAD, **_RDNA3},
    "gfx1101": {**_XE2_ONLY, **_KEY_HEAD, **_RDNA3},
    "gfx1151": {**_XE2_ONLY, **_KEY_HEAD, **_RDNA3},
    "gfx1200": {**_XE2_ONLY, **_RDNA4},
    "gfx1201": {**_XE2_ONLY, **_RDNA4},
}

# intel/llvm builds each range kernel twice, the second for a launch with an offset (SYCL 1.2.1's, deprecated), which
# the engine never makes: those copies are not counted
OFFSET_SUFFIX = "_with_offset"

BUNDLE_MAGIC = b"__CLANG_OFFLOAD_BUNDLE__"
COMPRESSED_MAGIC = b"CCOB"
ELF_MAGIC = b"\x7fELF"
EM_AMDGPU = 224
SHT_NOTE = 7
NT_AMDGPU_METADATA = 32


@dataclass(frozen=True)
class Kernel:
    target: str
    name: str
    vgprs: int
    vgpr_spills: int
    sgpr_spills: int
    scratch: int
    max_wg: int

    @property
    def spills(self) -> bool:
        """To scratch memory: spilled VGPRs, or spilled SGPRs with scratch (without, they went to VGPR lanes)."""
        return self.vgpr_spills > 0 or (self.sgpr_spills > 0 and self.scratch > 0)


def msgpack_load(data: bytes):
    """The subset of MessagePack the AMDGPU metadata uses (no ext types)."""
    def at(pos: int):
        b = data[pos]
        pos += 1
        if b <= 0x7F:
            return b, pos
        if b >= 0xE0:
            return b - 0x100, pos
        if 0x80 <= b <= 0x8F:
            return mapping(b & 0x0F, pos)
        if 0x90 <= b <= 0x9F:
            return array(b & 0x0F, pos)
        if 0xA0 <= b <= 0xBF:
            n = b & 0x1F
            return data[pos:pos + n].decode("utf-8", "replace"), pos + n
        if b == 0xC0:
            return None, pos
        if b in (0xC2, 0xC3):
            return b == 0xC3, pos
        fixed = {0xCA: ">f", 0xCB: ">d", 0xCC: ">B", 0xCD: ">H", 0xCE: ">I", 0xCF: ">Q", 0xD0: ">b", 0xD1: ">h",
                 0xD2: ">i", 0xD3: ">q"}
        if b in fixed:
            fmt = fixed[b]
            return struct.unpack_from(fmt, data, pos)[0], pos + struct.calcsize(fmt)
        sized = {0xC4: (">B", bytes), 0xC5: (">H", bytes), 0xC6: (">I", bytes), 0xD9: (">B", str), 0xDA: (">H", str),
                 0xDB: (">I", str)}
        if b in sized:
            fmt, kind = sized[b]
            n = struct.unpack_from(fmt, data, pos)[0]
            pos += struct.calcsize(fmt)
            raw = data[pos:pos + n]
            return (raw.decode("utf-8", "replace") if kind is str else bytes(raw)), pos + n
        if b in (0xDC, 0xDD, 0xDE, 0xDF):
            fmt = ">H" if b in (0xDC, 0xDE) else ">I"
            n = struct.unpack_from(fmt, data, pos)[0]
            pos += struct.calcsize(fmt)
            return (array if b in (0xDC, 0xDD) else mapping)(n, pos)
        raise ValueError(f"msgpack: type byte 0x{b:02x} at {pos - 1}")

    def array(n: int, pos: int):
        out = []
        for _ in range(n):
            v, pos = at(pos)
            out.append(v)
        return out, pos

    def mapping(n: int, pos: int):
        out = {}
        for _ in range(n):
            k, pos = at(pos)
            v, pos = at(pos)
            out[k] = v
        return out, pos

    return at(0)[0]


def bundle_entries(blob: bytes) -> list[tuple[str, bytes]]:
    """An uncompressed clang offload bundle: (triple, contents) for each entry."""
    count = struct.unpack_from("<Q", blob, len(BUNDLE_MAGIC))[0]
    pos = len(BUNDLE_MAGIC) + 8
    out = []
    for _ in range(count):
        off, size, tlen = struct.unpack_from("<QQQ", blob, pos)
        pos += 24
        triple = blob[pos:pos + tlen].decode("ascii", "replace")
        pos += tlen
        out.append((triple, blob[off:off + size]))
    return out


def amdgpu_metadata(elf: bytes) -> dict | None:
    """The NT_AMDGPU_METADATA note of a 64-bit little-endian AMDGPU ELF, or None."""
    if elf[:4] != ELF_MAGIC or elf[4] != 2 or elf[5] != 1 or struct.unpack_from("<H", elf, 18)[0] != EM_AMDGPU:
        return None
    shoff = struct.unpack_from("<Q", elf, 0x28)[0]
    shentsize, shnum = struct.unpack_from("<HH", elf, 0x3A)
    for i in range(shnum):
        sh = shoff + i * shentsize
        if struct.unpack_from("<I", elf, sh + 4)[0] != SHT_NOTE:
            continue
        off, size = struct.unpack_from("<QQ", elf, sh + 0x18)
        pos, end = off, off + size
        while pos + 12 <= end:
            namesz, descsz, ntype = struct.unpack_from("<III", elf, pos)
            pos += 12
            name = elf[pos:pos + namesz].rstrip(b"\0")
            pos += (namesz + 3) & ~3
            desc = elf[pos:pos + descsz]
            pos += (descsz + 3) & ~3
            if name == b"AMDGPU" and ntype == NT_AMDGPU_METADATA:
                return msgpack_load(desc)
    return None


def target_of(meta: dict, triple: str) -> str:
    t = meta.get("amdhsa.target") or triple
    m = re.search(r"(gfx[0-9a-f]+)", t)
    return m.group(1) if m else t


def kernels_of(elf: bytes, triple: str = "") -> list[Kernel]:
    meta = amdgpu_metadata(elf)
    if meta is None:
        return []
    target = target_of(meta, triple)
    return [Kernel(target, str(k.get(".name", "?")), int(k.get(".vgpr_count", 0)), int(k.get(".vgpr_spill_count", 0)),
                   int(k.get(".sgpr_spill_count", 0)), int(k.get(".private_segment_fixed_size", 0)),
                   int(k.get(".max_flat_workgroup_size", 0)))
            for k in meta.get("amdhsa.kernels", []) if not str(k.get(".name", "")).endswith(OFFSET_SUFFIX)]


def image_kernels(blob: bytes) -> tuple[list[Kernel], int]:
    """An extracted device image's AMD kernels and its count of compressed bundles (not read)."""
    if blob.startswith(BUNDLE_MAGIC):
        out = []
        for triple, part in bundle_entries(blob):
            if "amdgcn" in triple:
                out += kernels_of(part, triple)
        return out, 0
    if blob.startswith(COMPRESSED_MAGIC):
        return [], 1
    return kernels_of(blob), 0   # a bare code object, or another vendor's image (none)


def binary_kernels(binary: Path, extract: str) -> tuple[list[Kernel], int]:
    with tempfile.TemporaryDirectory() as d:
        stem = Path(d) / "img"
        subprocess.run([extract, "-q", f"--stem={stem}", str(binary)], check=True, stdout=subprocess.DEVNULL)
        kernels, compressed = [], 0
        for f in sorted(Path(d).glob("img.*")):
            k, c = image_kernels(f.read_bytes())
            kernels += k
            compressed += c
    return kernels, compressed


def allowed_reason(k: Kernel, allowed: dict[str, dict[str, str]]) -> str | None:
    for pattern, reason in allowed.get(k.target, {}).items():
        if re.search(pattern, k.name):
            return reason
    return None


def demangled(names: list[str]) -> dict[str, str]:
    """The kernels' C++ names (a SYCL kernel's name is its type's typeinfo name), where a demangler is installed."""
    tool = shutil.which("llvm-cxxfilt") or shutil.which("c++filt")
    if tool is None or not names:
        return {}
    out = subprocess.run([tool], input="\n".join(names), capture_output=True, text=True).stdout.splitlines()
    if len(out) != len(names):
        return {}
    return {n: d.removeprefix("typeinfo name for ") for n, d in zip(names, out)}


def describe(k: Kernel, pretty: dict[str, str]) -> str:
    return (f"{k.target} {pretty.get(k.name, k.name)}\n      VGPRs {k.vgprs}, VGPR spills {k.vgpr_spills}, SGPR spills "
            f"{k.sgpr_spills}, scratch {k.scratch} bytes, work-group bound {k.max_wg}")


def merged(kernels: list[Kernel]) -> list[Kernel]:
    """One entry a (target, kernel): a kernel in several images or binaries keeps its largest spills."""
    seen: dict[tuple[str, str], Kernel] = {}
    for k in kernels:
        prev = seen.get((k.target, k.name))
        if prev is None or (k.vgpr_spills, k.sgpr_spills) > (prev.vgpr_spills, prev.sgpr_spills):
            seen[(k.target, k.name)] = k
    return sorted(seen.values(), key=lambda k: (k.target, k.name))


def judge(kernels: list[Kernel], allowed: dict[str, dict[str, str]]) -> tuple[list[Kernel], list[Kernel], list[str]]:
    """(spilling kernels not allowed on a listed target, spilling kernels on a target not listed, allowed patterns
    whose kernels are in the binaries and none of them spills)"""
    spilling = [k for k in kernels if k.spills]
    bad = [k for k in spilling if k.target in allowed and allowed_reason(k, allowed) is None]
    unchecked = [k for k in spilling if k.target not in allowed]
    stale = []
    for t in sorted({k.target for k in kernels} & set(allowed)):
        for p in allowed[t]:
            hits = [k for k in kernels if k.target == t and re.search(p, k.name)]
            if hits and not any(k.spills for k in hits):
                stale.append(f"{t}: {p}")
    return bad, unchecked, stale


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("binaries", nargs="+", type=Path)
    ap.add_argument("--extract", default=None, help="clang-offload-extract (default: from PATH)")
    ap.add_argument("--list", action="store_true", help="list every spilling kernel, allowed or not")
    args = ap.parse_args(argv)
    extract = args.extract or shutil.which("clang-offload-extract")
    if extract is None or not Path(extract).exists():
        print("check_amdgpu_spills: no clang-offload-extract (the SYCL compiler's): skipped")
        return 77
    found: list[Kernel] = []
    compressed = 0
    for b in args.binaries:
        ks, c = binary_kernels(b, extract)
        found += ks
        compressed += c
    if compressed:
        print(f"check_amdgpu_spills: {compressed} compressed offload bundles not read")
    kernels = merged(found)
    if not kernels:
        print("check_amdgpu_spills: no AMD code object in the binaries: skipped")
        return 77
    bad, unchecked, stale = judge(kernels, ALLOWED)
    spilling = [k for k in kernels if k.spills]
    pretty = demangled([k.name for k in (spilling if args.list else bad + unchecked)])
    for t in sorted({k.target for k in kernels}):
        n, s = sum(k.target == t for k in kernels), sum(k.target == t for k in spilling)
        state = (f"{sum(k.target == t for k in bad)} not allowed" if t in ALLOWED
                 else "not checked: ALLOWED has no entry for this target")
        print(f"check_amdgpu_spills: {t}: {n} kernels, {s} spill ({state})")
    for k in spilling if args.list else bad + unchecked:
        reason = allowed_reason(k, ALLOWED)
        tag = "allowed" if reason else "SPILLS" if k.target in ALLOWED else "not checked"
        print(f"  [{tag}] {describe(k, pretty)}" + (f"\n      {reason}" if reason else ""))
    for entry in stale:
        print(f"  note: no kernel spills that ALLOWED's {entry} names: remove the entry")
    if bad:
        print("check_amdgpu_spills: bound the kernel's work-group (reqd_work_group_size or max_work_group_size, "
              "include/strata/kernels/work_group.hpp) or, if it must spill, add it to ALLOWED in "
              "tools/check_amdgpu_spills.py with the reason")
    return 1 if bad and not args.list else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
