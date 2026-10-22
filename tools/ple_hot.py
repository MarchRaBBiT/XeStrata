# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""tools/ple_hot.py - the n-gram table's rows for a corpus' most frequent n-grams, for the engine to keep in RAM.

    python tools/ple_hot.py --gguf <the shard with per_layer_token_embd> --tokenizer <pack>/tokenizer \\
        --gib 3 --out <data>/ple-hot/<model>.hot corpus1.txt corpus2.txt ...

and then `--ple-hot <that file>` in the engine's arguments (a config's "args": two entries).

The engine reads the PLE table (per_layer_token_embd, 28.8 GB) from the disk, 16 rows a token at random (--ple-io
direct), and its row cache keeps only what the process has read. A rotational disk does a few hundred random reads a
second, so a token whose n-grams are new waits on the disk. A token's 16 rows depend on its bigram (8 rows) and its
trigram (8 rows) only, so the rows of a corpus' most frequent n-grams are the ones a model's text asks for most: on
Swift 1.5's answers (Japanese and English, with its English reasoning), 4.3M n-grams of 125M tokens of Wikipedia and
chat text in English and Japanese covered 77% of the bigram rows and 20% of the trigram rows (3 GiB).

The corpus: UTF-8 text files, documents separated by two blank lines ("\\n\\n\\n"). Each document is tokenized alone,
and the predecessors its first tokens lack count as the engine counts them (EOS). The text is not kept: only the
n-gram counts, and in the output the table's rows.

The output is described in include/strata/ngram/ple_hot.hpp. The rows are read from the GGUF in one pass (28.8 GB, a
few minutes from a rotational disk), so the file holds the table's own bytes; the engine refuses it for any other table.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import pathlib
import struct
import sys
import time
from typing import Any

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gguf_reader import BLOCK_GEOMETRY, GGUFFile  # noqa: E402

# The hash's constants as the engine writes them (ple_artifact_consts in src/kernels/ngram.cpp); the GGUF's metadata
# must agree, or the rows chosen here would not be the rows the engine asks for.
MULT = (23703573157769, 20109073645365, 8052911324071)
VOCAB = (20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
         20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171)
OFFSET = (0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
          160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275)
HEADS = 8                                    # per n-gram size: heads 0-7 bigrams, 8-15 trigrams
EOS = 248044                                 # qwen4exp.ple.eos_token_id; also what a missing predecessor reads as
TOKEN_BITS = 18                              # token ids < 262144: a trigram key is 3 x 18 bits
TRIGRAM = np.uint64(1 << 63)                 # the key's mark for a trigram
TABLE = "per_layer_token_embd.weight"
MAGIC, VERSION, SAMPLES = b"XSPLEHOT", 1, 64
KEPT_ROW_EXTRA = 4                           # bytes a kept row costs the engine besides the row: its id
CHUNK_ROWS = 1 << 22                         # rows a read of the table pass (~360 MB of IQ4_NL)
_tok: Any = None                             # a tokenizer process' tokenizer (_init)


def ngram_rows(tokens, prev) -> np.ndarray:
    """The engine's 16 rows per token (ngram_rows in src/kernels/ngram.cpp): `prev` holds two predecessors a token,
    the older first, -1 for none. An EOS or a missing predecessor cuts that one and every older one."""
    out = np.zeros((len(tokens), 2 * HEADS), dtype=np.uint64)
    m = [np.uint64(x) for x in MULT]
    for i, t in enumerate(tokens):
        ctx = [int(t)]
        cut = False
        for s in (1, 2):
            p = prev[2 * i + (2 - s)]
            cut = cut or p < 0 or p == EOS
            ctx.append(EOS if cut else int(p))
        c = np.array(ctx, dtype=np.uint64)
        mixed2 = (c[0] * m[0]) ^ (c[1] * m[1])
        mixed3 = mixed2 ^ (c[2] * m[2])
        for h in range(HEADS):
            out[i, h] = mixed2 % np.uint64(VOCAB[h]) + np.uint64(OFFSET[h])
            out[i, HEADS + h] = mixed3 % np.uint64(VOCAB[HEADS + h]) + np.uint64(OFFSET[HEADS + h])
    return out


def doc_keys(ids) -> np.ndarray:
    """The bigram and trigram keys of one document's tokens (its start cut to EOS)."""
    if len(ids) == 0:
        return np.empty(0, dtype=np.uint64)
    a = np.asarray(ids, dtype=np.uint64)
    p1 = np.concatenate([np.array([EOS], dtype=np.uint64), a[:-1]])
    p2 = np.concatenate([np.array([EOS, EOS], dtype=np.uint64), a[:-2]])[:len(a)]
    p2 = np.where(p1 == np.uint64(EOS), np.uint64(EOS), p2)         # an EOS one back cuts the one before it too
    bi = (a << np.uint64(TOKEN_BITS)) | p1
    tri = (bi << np.uint64(TOKEN_BITS)) | p2 | TRIGRAM
    return np.concatenate([bi, tri])


def key_rows(keys: np.ndarray) -> np.ndarray:
    """The 8 rows of each n-gram key, [len(keys), 8], in the order of the keys."""
    mask = np.uint64((1 << TOKEN_BITS) - 1)
    tri = (keys & TRIGRAM) != 0
    k = keys & ~TRIGRAM
    b = np.uint64(TOKEN_BITS)
    t = np.where(tri, (k >> (b + b)) & mask, k >> b)
    p1 = np.where(tri, (k >> b) & mask, k & mask)
    p2 = np.where(tri, k & mask, np.uint64(0))
    with np.errstate(over="ignore"):
        mixed = (t * np.uint64(MULT[0])) ^ (p1 * np.uint64(MULT[1]))
        mixed = np.where(tri, mixed ^ (p2 * np.uint64(MULT[2])), mixed)
    out = np.empty((len(keys), HEADS), dtype=np.uint64)
    for h in range(HEADS):
        bi_row = mixed % np.uint64(VOCAB[h]) + np.uint64(OFFSET[h])
        tri_row = mixed % np.uint64(VOCAB[HEADS + h]) + np.uint64(OFFSET[HEADS + h])
        out[:, h] = np.where(tri, tri_row, bi_row)
    return out


def _init(tokenizer_dir: str) -> None:
    global _tok
    import strata_tokenizer
    _tok = strata_tokenizer.Tokenizer.from_dir(tokenizer_dir)


def _keys_of(doc: str) -> np.ndarray:
    return doc_keys(_tok.encode(doc))


def documents(path: pathlib.Path, block: int = 64 << 20):
    """The documents of a corpus file, read a block at a time."""
    tail = ""
    with open(path, encoding="utf-8", errors="replace") as f:
        while True:
            text = f.read(block)
            if not text:
                break
            parts = (tail + text).split("\n\n\n")
            tail = parts.pop()
            yield from (p for p in parts if p.strip())
    if tail.strip():
        yield tail


def merge(parts: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Keys and counts with each key once, its counts summed."""
    if not parts:
        return np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.uint64)
    u = np.concatenate([p[0] for p in parts])
    c = np.concatenate([p[1] for p in parts])
    order = np.argsort(u, kind="stable")
    u, c = u[order], c[order]
    first = np.flatnonzero(np.concatenate([[True], u[1:] != u[:-1]]))
    return u[first], np.add.reduceat(c, first).astype(np.uint64)


def count_ngrams(files, tokenizer_dir: str, workers: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Every corpus file's n-gram keys and their counts, and the number of tokens."""
    parts, pending, n_tok = [], [], 0
    with mp.Pool(workers, initializer=_init, initargs=(tokenizer_dir,)) as pool:
        for path in files:
            t0 = time.time()
            before = n_tok
            for k in pool.imap_unordered(_keys_of, documents(path), chunksize=64):
                pending.append(k)
                n_tok += len(k) // 2
                if sum(len(p) for p in pending) > 40_000_000:      # folded as it goes: memory stays bounded
                    u, c = np.unique(np.concatenate(pending), return_counts=True)
                    parts.append((u, c.astype(np.uint64)))
                    pending = []
            print(f"  {path}: {n_tok - before:,} tokens ({time.time() - t0:.0f} s)", flush=True)
    if pending:
        u, c = np.unique(np.concatenate(pending), return_counts=True)
        parts.append((u, c.astype(np.uint64)))
    keys, counts = merge(parts)
    return keys, counts, n_tok


def choose_rows(keys: np.ndarray, counts: np.ndarray, budget: int) -> tuple[np.ndarray, tuple[int, int]]:
    """The rows of the most frequent n-grams, ascending, at most `budget`: n-grams by count (the order of the keys
    breaks ties), each one's 8 rows, a row two n-grams share kept once.  Also (distinct rows, n-grams looked at)."""
    order = np.argsort(-counts.astype(np.int64), kind="stable")
    take = min(len(order), budget // HEADS + budget // (8 * HEADS) + 1)   # a few more n-grams than fit: shared rows
    rows = key_rows(keys[order[:take]]).reshape(-1)
    uniq, first = np.unique(rows, return_index=True)
    keep = np.sort(first)[:budget]
    return np.sort(rows[keep]).astype(np.uint32), (uniq.size, take)


def table_of(gguf: pathlib.Path) -> tuple[int, int, int]:
    """(byte offset of row 0, rows, bytes a row) of the n-gram table in this GGUF; the hash's constants checked."""
    g = GGUFFile(gguf)
    t = next((t for t in g.tensors if t.name == TABLE), None)
    if t is None:
        sys.exit(f"{gguf.name} has no {TABLE} (the n-gram table is in another shard of the model)")
    geom = BLOCK_GEOMETRY.get(t.type_name)
    if len(t.shape) != 2 or geom is None or t.shape[0] % geom[0]:
        sys.exit(f"{TABLE} is {t.type_name} {t.shape}: not a table this tool knows")
    md = g.metadata
    for key, want in (("qwen4exp.ple.layer_multipliers", MULT), ("qwen4exp.ple.head_vocab_sizes", VOCAB),
                      ("qwen4exp.ple.head_offsets", OFFSET)):
        if key in md and tuple(md[key]) != want:
            sys.exit(f"{gguf.name}: {key} differs from the engine's constants: its rows would not be the engine's")
    if md.get("qwen4exp.ple.eos_token_id", EOS) != EOS:
        sys.exit(f"{gguf.name}: qwen4exp.ple.eos_token_id is not {EOS}")
    rows = t.shape[1]
    if rows < OFFSET[-1] + VOCAB[-1]:                # (the GGUFs pad it: 320,001,536 rows for the 320,001,446 hashed)
        sys.exit(f"{TABLE} has {rows} rows, fewer than the {OFFSET[-1] + VOCAB[-1]} the hash addresses")
    return g.data_start + t.offset, rows, t.shape[0] // geom[0] * geom[1]


def fnv1a64(data: bytes, h: int = 0xcbf29ce484222325) -> int:
    for b in data:
        h = ((h ^ b) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h


def sample_hash(f, table_offset: int, rows: int, rb: int) -> int:
    """FNV-1a 64 of the rows i * (rows / 64), back to back: ties the file to this table (ple_hot.hpp)."""
    step = rows // SAMPLES
    data = b"".join(os.pread(f.fileno(), rb, table_offset + i * step * rb) for i in range(SAMPLES))
    return fnv1a64(data)


def cut_rows(f, table_offset: int, rb: int, ids: np.ndarray, progress=None) -> np.ndarray:
    """The bytes of rows `ids` (ascending), [len(ids), rb], read from the table in one pass in order."""
    out = np.empty((len(ids), rb), dtype=np.uint8)
    if len(ids) == 0:
        return out
    last = int(ids[-1]) + 1
    buf = bytearray(CHUNK_ROWS * rb)
    done = 0
    for r0 in range(0, last, CHUNK_ROWS):
        r1 = min(r0 + CHUNK_ROWS, last)
        lo, hi = np.searchsorted(ids, [r0, r1])
        if hi > lo:
            view = memoryview(buf)[:(r1 - r0) * rb]
            f.seek(table_offset + r0 * rb)
            if f.readinto(view) != len(view):
                sys.exit("the table ends early: a short GGUF")
            table = np.frombuffer(view, dtype=np.uint8).reshape(r1 - r0, rb)
            out[lo:hi] = table[ids[lo:hi].astype(np.int64) - r0]
            done = hi
        if hasattr(os, "posix_fadvise"):     # one pass: keep 28.8 GB from pushing everything else out of the cache
            os.posix_fadvise(f.fileno(), table_offset + r0 * rb, (r1 - r0) * rb, os.POSIX_FADV_DONTNEED)
        if progress:
            progress(r1, last, done)
    return out


def write_hot(path: pathlib.Path, rb: int, rows: int, table_offset: int, sample: int, ids: np.ndarray,
              data: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as f:
        f.write(MAGIC + struct.pack("<IIQQQQ", VERSION, rb, rows, table_offset, sample, len(ids)))
        f.write(ids.astype("<u4").tobytes())
        f.write(data.tobytes())
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("corpus", nargs="+", type=pathlib.Path, help="UTF-8 text files, documents split by \"\\n\\n\\n\"")
    ap.add_argument("--gguf", required=True, type=pathlib.Path, help="the model's shard with per_layer_token_embd")
    ap.add_argument("--tokenizer", required=True, help="the model's tokenizer folder (<pack>/tokenizer)")
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--gib", type=float, default=3.0, help="RAM the engine gives the rows (default 3)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2), help="tokenizer processes")
    a = ap.parse_args()
    if a.gib <= 0:
        ap.error("--gib must be positive")
    table_offset, rows, rb = table_of(a.gguf)
    budget = int(a.gib * (1 << 30)) // (rb + KEPT_ROW_EXTRA)
    print(f"table: {rows:,} rows of {rb} bytes at {table_offset:,} in {a.gguf.name}; up to {budget:,} rows "
          f"({a.gib:g} GiB)", flush=True)
    t0 = time.time()
    keys, counts, n_tok = count_ngrams(a.corpus, a.tokenizer, a.workers)
    print(f"corpus: {n_tok:,} tokens, {len(keys):,} distinct n-grams ({time.time() - t0:.0f} s)", flush=True)
    ids, (distinct_rows, n_grams) = choose_rows(keys, counts, budget)
    del keys, counts
    print(f"rows: {len(ids):,} of the {n_grams:,} most frequent n-grams ({distinct_rows:,} distinct rows)", flush=True)
    t1 = time.time()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.gguf, "rb", buffering=0) as f:
        sample = sample_hash(f, table_offset, rows, rb)
        step = [0]

        def progress(r, last, done):
            if r * 10 // last > step[0]:
                step[0] = r * 10 // last
                print(f"  reading the table: {r * 100 // last}% ({time.time() - t1:.0f} s)", flush=True)
        data = cut_rows(f, table_offset, rb, ids, progress)
    write_hot(a.out, rb, rows, table_offset, sample, ids, data)
    size = a.out.stat().st_size
    print(f"wrote {a.out}: {len(ids):,} rows, {size / (1 << 30):.2f} GiB ({time.time() - t0:.0f} s in all)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
