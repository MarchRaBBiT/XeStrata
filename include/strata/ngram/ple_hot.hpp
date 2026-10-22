// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/ngram/ple_hot.hpp - the n-gram table's most used rows, kept in RAM from a file (--ple-hot).
//
// A token's 16 rows of the PLE table are read from the disk at random. A rotational disk does a few hundred such
// reads a second, so a token whose n-grams the row cache has not seen waits on the disk; the row cache only learns
// what this process has already read. tools/ple_hot.py counts the bigrams and trigrams of a text corpus, cuts the rows
// of the most frequent ones out of the table and writes them to a file; the reader keeps those rows in RAM for its
// whole life and reads nothing for them.
//
// The file (little-endian):
//
//     char     magic[8]      "XSPLEHOT"
//     uint32   version       1
//     uint32   row_bytes     one row of the table (90 IQ4_NL)
//     uint64   table_rows    the table's row count
//     uint64   table_offset  the byte offset of row 0 in the GGUF
//     uint64   sample_hash   FNV-1a 64 of the rows i * (table_rows / 64), i = 0..63, back to back
//     uint64   n             rows in this file
//     uint32   ids[n]        ascending, no repeats, each < table_rows
//     uint8    rows[n * row_bytes]
//
// A file cut from another table of the same shape (another model's, another quantization of it) would be served as
// plausible but wrong rows, so `load_hot_rows` reads the 64 sample rows from this table and refuses a file whose hash
// differs.
#pragma once

#include "strata/ngram/ple_reader.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::ngram {

inline constexpr char HOT_MAGIC[8] = {'X', 'S', 'P', 'L', 'E', 'H', 'O', 'T'};
inline constexpr uint32_t HOT_VERSION = 1;
inline constexpr uint32_t HOT_SAMPLES = 64;       ///< rows hashed to tie a file to its table
inline constexpr uint64_t HOT_HEADER_BYTES = 48;

/// FNV-1a 64 over `n` bytes, continuing from `h` (start at the offset basis 0xcbf29ce484222325).
uint64_t fnv1a64(const uint8_t* p, size_t n, uint64_t h = 0xcbf29ce484222325ull);

/// The rows whose bytes `sample_hash` covers: i * (table_rows / HOT_SAMPLES) for i = 0..HOT_SAMPLES-1.
std::vector<uint32_t> hot_sample_rows(uint64_t table_rows);

/// The rows of a hot file, in the order of their ids.
struct HotRows {
    std::vector<uint32_t> ids;      ///< ascending, no repeats
    std::vector<uint8_t> bytes;     ///< row_bytes each
};

/// Reads the hot file at `path` and checks it against the table `reader` has open (row size, row count, offset, and
/// the sample rows' hash, read through `reader`). Call with no ticket in flight. false with a message in `err`.
bool load_hot_rows(const std::string& path, PleReader& reader, uint64_t table_rows, uint64_t table_offset,
                   HotRows& out, std::string& err);

}  // namespace strata::ngram
