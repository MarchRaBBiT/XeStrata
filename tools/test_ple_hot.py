# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""tools/ple_hot.py: its hash against the engine's oracle vectors, the rows it chooses, the file it writes.

Run .venv/bin/python -m unittest discover -s tools -p test_ple_hot.py."""
from pathlib import Path
import re
import struct
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))   # runnable from the repo root too

import ple_hot  # noqa: E402

ORACLE = Path(__file__).resolve().parents[1] / "src" / "kernels" / "ple_oracle_vectors.inc"


def oracle_cases():
    """(name, tokens, prev, rows) of every hash case the engine's parity test checks."""
    text = ORACLE.read_text()
    arrays = {m.group(2): [int(x) for x in m.group(3).replace("\n", " ").split(",") if x.strip()]
              for m in re.finditer(r"inline const (int32_t|uint32_t) (case\d+_\w+)\[\d+\] = \{([^}]*)\}", text)}
    names = sorted({k.split("_")[0] for k in arrays}, key=lambda c: int(c[4:]))
    return [(n, arrays[n + "_tokens"], arrays[n + "_prev"], arrays[n + "_rows"]) for n in names
            if n + "_rows" in arrays]


def table_bytes(rows, rb):
    """A synthetic table whose every row encodes its own index (a wrong offset shows as a mismatch)."""
    t = (np.arange(rows, dtype=np.uint64)[:, None] * np.uint64(2654435761) + np.arange(rb, dtype=np.uint64) * 97)
    t = ((t >> np.uint64(7)) & np.uint64(0xFF)).astype(np.uint8)
    t[:, :4] = np.arange(rows, dtype=np.uint32).view(np.uint8).reshape(rows, 4)
    return t


class HashTest(unittest.TestCase):
    def test_engine_oracle(self):
        cases = oracle_cases()
        self.assertGreaterEqual(len(cases), 3)
        for name, tokens, prev, rows in cases:
            with self.subTest(case=name):
                got = ple_hot.ngram_rows(tokens, prev).reshape(-1).tolist()
                self.assertEqual(got, rows)

    def test_keys_give_the_same_rows(self):
        rng = np.random.default_rng(7)
        ids = rng.integers(0, 248320, size=200).tolist()
        ids[50] = ple_hot.EOS                              # an EOS in the text cuts the n-grams after it
        prev = []
        for i in range(len(ids)):
            prev += [ids[i - 2] if i >= 2 else -1, ids[i - 1] if i >= 1 else -1]
        want = ple_hot.ngram_rows(ids, prev)
        keys = ple_hot.doc_keys(ids)
        rows = ple_hot.key_rows(keys)
        n = len(ids)
        np.testing.assert_array_equal(rows[:n], want[:, :ple_hot.HEADS])            # bigrams: heads 0-7
        np.testing.assert_array_equal(rows[n:], want[:, ple_hot.HEADS:])            # trigrams: heads 8-15

    def test_fnv1a64(self):
        self.assertEqual(ple_hot.fnv1a64(b""), 0xcbf29ce484222325)
        self.assertEqual(ple_hot.fnv1a64(b"a"), 0xaf63dc4c8601ec8c)
        self.assertEqual(ple_hot.fnv1a64(b"foobar"), 0x85944171f73967e8)


class ChooseTest(unittest.TestCase):
    def test_most_frequent_first_and_within_budget(self):
        keys = ple_hot.doc_keys([11, 22, 33, 44, 55])
        counts = np.arange(len(keys), dtype=np.uint64) + 1    # the last key the most frequent
        ids, (distinct, looked) = ple_hot.choose_rows(keys, counts, 20)
        self.assertEqual(len(ids), 20)
        self.assertTrue(np.all(np.diff(ids.astype(np.int64)) > 0))
        top = set(ple_hot.key_rows(keys[-1:]).reshape(-1).tolist())
        self.assertTrue(top <= set(ids.tolist()))           # the most frequent n-gram's 8 rows are all in
        everything, _ = ple_hot.choose_rows(keys, counts, 10_000)
        self.assertEqual(len(everything), len(set(ple_hot.key_rows(keys).reshape(-1).tolist())))

    def test_documents_across_blocks(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.txt"
            p.write_text("one\n\n\ntwo two\n\n\n\n\n\nthree", encoding="utf-8")
            self.assertEqual(list(ple_hot.documents(p, block=4)), ["one", "two two", "three"])


class HotFileTest(unittest.TestCase):
    def test_cut_rows_and_the_file(self):
        rb, rows, offset = 90, 5000, 192
        table = table_bytes(rows, rb)
        ids = np.array([0, 3, 4095, 4096, 4999], dtype=np.uint32)
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "t.gguf"
            src.write_bytes(b"\xab" * offset + table.tobytes())
            old = ple_hot.CHUNK_ROWS
            ple_hot.CHUNK_ROWS = 1000                       # several chunks, rows on both sides of their edges
            try:
                with open(src, "rb", buffering=0) as f:
                    data = ple_hot.cut_rows(f, offset, rb, ids)
                    sample = ple_hot.sample_hash(f, offset, rows, rb)
            finally:
                ple_hot.CHUNK_ROWS = old
            np.testing.assert_array_equal(data, table[ids.astype(np.int64)])
            step = rows // ple_hot.SAMPLES
            self.assertEqual(sample, ple_hot.fnv1a64(table[[i * step for i in range(ple_hot.SAMPLES)]].tobytes()))

            out = Path(d) / "x.hot"
            ple_hot.write_hot(out, rb, rows, offset, sample, ids, data)
            b = out.read_bytes()
            self.assertEqual(b[:8], b"XSPLEHOT")
            self.assertEqual(struct.unpack("<IIQQQQ", b[8:48]), (1, rb, rows, offset, sample, len(ids)))
            self.assertEqual(np.frombuffer(b[48:48 + 4 * len(ids)], dtype="<u4").tolist(), ids.tolist())
            self.assertEqual(b[48 + 4 * len(ids):], data.tobytes())
            self.assertEqual(len(b), 48 + len(ids) * (4 + rb))
            self.assertFalse((Path(d) / "x.hot.part").exists())


if __name__ == "__main__":
    unittest.main()
