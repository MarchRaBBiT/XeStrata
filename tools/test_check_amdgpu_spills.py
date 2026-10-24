# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for tools/check_amdgpu_spills.py: reading the AMDGPU metadata out of offload bundles, and which spilling
kernels fail the check.  No binary, no GPU.

    python -m unittest tools.test_check_amdgpu_spills
"""
from __future__ import annotations

import struct
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_amdgpu_spills as C  # noqa: E402


def pack(v) -> bytes:
    """MessagePack for the types the AMDGPU metadata holds (the encoder side of C.msgpack_load)."""
    if isinstance(v, bool):
        return b"\xc3" if v else b"\xc2"
    if isinstance(v, int):
        if 0 <= v <= 0x7F:
            return bytes([v])
        if 0 <= v <= 0xFFFF:
            return b"\xcd" + struct.pack(">H", v)
        return b"\xd2" + struct.pack(">i", v)
    if isinstance(v, str):
        raw = v.encode()
        return (bytes([0xA0 | len(raw)]) if len(raw) < 32 else b"\xd9" + bytes([len(raw)])) + raw
    if isinstance(v, list):
        return bytes([0x90 | len(v)]) + b"".join(pack(x) for x in v)
    if isinstance(v, dict):
        return bytes([0x80 | len(v)]) + b"".join(pack(k) + pack(x) for k, x in v.items())
    raise TypeError(v)


def kernel_meta(name: str, vgprs: int = 32, vspill: int = 0, sspill: int = 0, scratch: int = 0, wg: int = 256) -> dict:
    return {".name": name, ".vgpr_count": vgprs, ".vgpr_spill_count": vspill, ".sgpr_spill_count": sspill,
            ".private_segment_fixed_size": scratch, ".max_flat_workgroup_size": wg}


def code_object(meta: dict) -> bytes:
    """A 64-bit little-endian AMDGPU ELF with one note section holding `meta` as NT_AMDGPU_METADATA."""
    desc = pack(meta)
    name = b"AMDGPU\0"
    note = struct.pack("<III", len(name), len(desc), C.NT_AMDGPU_METADATA)
    note += name + b"\0" * (-len(name) % 4) + desc + b"\0" * (-len(desc) % 4)
    note_off = 64
    shoff = note_off + len(note)
    hdr = bytearray(64)
    hdr[:4] = C.ELF_MAGIC
    hdr[4], hdr[5] = 2, 1
    struct.pack_into("<H", hdr, 18, C.EM_AMDGPU)
    struct.pack_into("<Q", hdr, 0x28, shoff)
    struct.pack_into("<HH", hdr, 0x3A, 64, 2)
    null_sh = bytes(64)
    note_sh = bytearray(64)
    struct.pack_into("<I", note_sh, 4, C.SHT_NOTE)
    struct.pack_into("<QQ", note_sh, 0x18, note_off, len(note))
    return bytes(hdr) + note + null_sh + bytes(note_sh)


def bundle(entries: list[tuple[str, bytes]]) -> bytes:
    head = C.BUNDLE_MAGIC + struct.pack("<Q", len(entries))
    table_len = sum(24 + len(t) for t, _ in entries)
    off = len(head) + table_len
    table, body = b"", b""
    for triple, part in entries:
        table += struct.pack("<QQQ", off + len(body), len(part), len(triple)) + triple.encode()
        body += part
    return head + table + body


def K(name: str, spills: int = 0, target: str = "gfx906") -> C.Kernel:
    return C.Kernel(target, name, 64, spills, 0, 4 * spills, 1024)


class Reading(unittest.TestCase):
    def test_msgpack_round_trip(self):
        v = {"amdhsa.kernels": [kernel_meta("k" * 40, 300, 7)], "x": [True, False, -3, 70000]}
        self.assertEqual(C.msgpack_load(pack(v)), v)

    def test_kernels_of_a_bundle(self):
        meta = {"amdhsa.target": "amdgcn-amd-amdhsa--gfx906:xnack-",
                "amdhsa.kernels": [kernel_meta("_ZTS3Foo", 64, 5, 1, 24, 1024), kernel_meta("_ZTS3Bar")]}
        blob = bundle([("host-x86_64-unknown-linux-gnu", b""), ("hipv4-amdgcn-amd-amdhsa--gfx906", code_object(meta))])
        ks, compressed = C.image_kernels(blob)
        self.assertEqual(compressed, 0)
        self.assertEqual(ks, [C.Kernel("gfx906", "_ZTS3Foo", 64, 5, 1, 24, 1024),
                              C.Kernel("gfx906", "_ZTS3Bar", 32, 0, 0, 0, 256)])
        self.assertTrue(ks[0].spills)
        self.assertFalse(ks[1].spills)

    def test_sgpr_spills_count_only_with_scratch(self):
        self.assertFalse(C.Kernel("gfx1200", "k", 99, 0, 11, 0, 1024).spills)   # into VGPR lanes
        self.assertTrue(C.Kernel("gfx1100", "k", 192, 0, 2, 16, 1024).spills)
        self.assertTrue(C.Kernel("gfx1100", "k", 192, 8, 0, 36, 1024).spills)

    def test_offset_copies_are_not_counted(self):
        meta = {"amdhsa.target": "amdgcn-amd-amdhsa--gfx1200",
                "amdhsa.kernels": [kernel_meta("_ZTS3Foo", vspill=2),
                                   kernel_meta("_ZTS3Foo" + C.OFFSET_SUFFIX, vspill=9)]}
        ks = C.kernels_of(code_object(meta))
        self.assertEqual([(k.target, k.name, k.vgpr_spills) for k in ks], [("gfx1200", "_ZTS3Foo", 2)])

    def test_target_from_the_triple_without_metadata_target(self):
        ks = C.kernels_of(code_object({"amdhsa.kernels": [kernel_meta("k")]}), "hipv4-amdgcn-amd-amdhsa--gfx1030")
        self.assertEqual(ks[0].target, "gfx1030")

    def test_other_images_hold_no_amd_kernels(self):
        self.assertEqual(C.image_kernels(b"\x03\x02\x23\x07" + bytes(60)), ([], 0))   # SPIR-V
        self.assertEqual(C.image_kernels(C.COMPRESSED_MAGIC + bytes(60)), ([], 1))
        self.assertIsNone(C.amdgpu_metadata(b"\x7fELF\x02\x01" + bytes(58)))   # an ELF of another machine


class Policy(unittest.TestCase):
    ALLOWED = {"gfx906": {r"GdnRecKh": "by design", r"launch_chunkILi256E": "measured faster"}}

    def test_an_allowed_spill_passes_and_a_new_one_fails(self):
        ks = [K("_ZTSN6strata7prefill3GdnILi16ELi32EE8GdnRecKhE", 49), K("_ZTS6NewOne", 3), K("_ZTS5Clean")]
        bad, unchecked, stale = C.judge(ks, self.ALLOWED)
        self.assertEqual([k.name for k in bad], ["_ZTS6NewOne"])
        self.assertEqual(unchecked, [])
        self.assertEqual(stale, [])   # launch_chunk is not in these binaries: nothing to say about it

    def test_a_target_without_an_entry_is_not_checked(self):
        bad, unchecked, _ = C.judge([K("_ZTS6NewOne", 3, "gfx1200")], self.ALLOWED)
        self.assertEqual(bad, [])
        self.assertEqual([k.target for k in unchecked], ["gfx1200"])

    def test_an_entry_is_per_target(self):
        bad, _, _ = C.judge([K("_ZTS8GdnRecKh", 4, "gfx1030")], {**self.ALLOWED, "gfx1030": {}})
        self.assertEqual([k.target for k in bad], ["gfx1030"])

    def test_an_entry_whose_kernel_no_longer_spills_is_stale(self):
        ks = [K("_ZTSZN6strata12launch_chunkILi256ELi12ELi0EEE", 0), K("_ZTS8GdnRecKh", 49)]
        self.assertEqual(C.judge(ks, self.ALLOWED)[2], ["gfx906: launch_chunkILi256E"])

    def test_a_kernel_in_several_images_keeps_its_largest_spills(self):
        ks = C.merged([K("_ZTS3Foo", 0), K("_ZTS3Foo", 6), K("_ZTS3Foo", 2), K("_ZTS3Foo", 1, "gfx1030")])
        self.assertEqual([(k.target, k.vgpr_spills) for k in ks], [("gfx1030", 1), ("gfx906", 6)])

    def test_the_engines_allowed_kernels_name_one_target_each(self):
        for target, entries in C.ALLOWED.items():
            self.assertRegex(target, r"^gfx[0-9a-f]+$")
            for pattern, reason in entries.items():
                self.assertTrue(reason.strip(), pattern)


if __name__ == "__main__":
    unittest.main()
