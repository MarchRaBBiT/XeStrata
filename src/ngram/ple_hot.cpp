// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/ngram/ple_hot.cpp - see include/strata/ngram/ple_hot.hpp.
#include "strata/ngram/ple_hot.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <new>
#include <system_error>

namespace strata::ngram {

namespace {

constexpr size_t CHUNK = 64u << 20;            // bytes a read: a 4 GB file in pieces the stream surely takes

uint32_t le32(const uint8_t* p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}

uint64_t le64(const uint8_t* p) {
    uint64_t v;
    std::memcpy(&v, p, 8);
    return v;
}

bool read_all(std::ifstream& f, uint8_t* dst, uint64_t n) {
    while (n > 0) {
        const size_t k = (size_t) std::min<uint64_t>(n, CHUNK);
        if (!f.read((char*) dst, (std::streamsize) k)) return false;
        dst += k;
        n -= k;
    }
    return true;
}

}  // namespace

uint64_t fnv1a64(const uint8_t* p, size_t n, uint64_t h) {
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

std::vector<uint32_t> hot_sample_rows(uint64_t table_rows) {
    std::vector<uint32_t> rows(HOT_SAMPLES);
    const uint64_t step = table_rows / HOT_SAMPLES;
    for (uint32_t i = 0; i < HOT_SAMPLES; ++i) rows[i] = (uint32_t) (i * step);
    return rows;
}

bool load_hot_rows(const std::string& path, PleReader& reader, uint64_t table_rows, uint64_t table_offset,
                   HotRows& out, std::string& err) {
    const std::string what = "--ple-hot " + path + ": ";
    std::error_code ec;
    const uint64_t size = std::filesystem::file_size(path, ec);
    if (ec) { err = what + "cannot read it (" + ec.message() + ")"; return false; }
    std::ifstream f(path, std::ios::binary);
    uint8_t h[HOT_HEADER_BYTES];
    if (!f || !f.read((char*) h, sizeof h)) { err = what + "cannot read its header"; return false; }
    if (std::memcmp(h, HOT_MAGIC, sizeof HOT_MAGIC) != 0) { err = what + "not a hot-rows file (tools/ple_hot.py)"; return false; }
    if (le32(h + 8) != HOT_VERSION) { err = what + "version " + std::to_string(le32(h + 8)) + ", not 1"; return false; }
    const uint32_t rb = le32(h + 12);
    const uint64_t rows = le64(h + 16), offset = le64(h + 24), sample_hash = le64(h + 32), n = le64(h + 40);
    if (rb != reader.row_bytes() || rows != table_rows || offset != table_offset) {
        err = what + "cut from another table (" + std::to_string(rows) + " rows of " + std::to_string(rb) + " bytes at " +
              std::to_string(offset) + "; this one is " + std::to_string(table_rows) + " of " +
              std::to_string(reader.row_bytes()) + " at " + std::to_string(table_offset) + ")";
        return false;
    }
    if (n == 0 || n > table_rows || size != HOT_HEADER_BYTES + n * (4 + (uint64_t) rb)) {
        err = what + "its size does not match its " + std::to_string(n) + " rows (a damaged or partial file)";
        return false;
    }
    // the sample rows, through the reader: the same bytes the engine would read for them
    const std::vector<uint32_t> sample = hot_sample_rows(table_rows);
    std::vector<uint8_t> got(sample.size() * rb);
    const PleReader::Ticket t = reader.issue(sample.data(), sample.size(), got.data());
    if (!reader.collect(t, err)) return false;
    if (fnv1a64(got.data(), got.size()) != sample_hash) {
        err = what + "cut from another table of the same shape (its sample rows differ from this one's): make it again "
                     "from this model's GGUF with tools/ple_hot.py";
        return false;
    }
    try {
        out.ids.resize(n);
        out.bytes.resize(n * rb);
    } catch (const std::bad_alloc&) {
        out = HotRows{};
        err = what + "not enough memory for its " + std::to_string(n) + " rows";
        return false;
    }
    if (!read_all(f, (uint8_t*) out.ids.data(), n * 4) || !read_all(f, out.bytes.data(), n * rb)) {
        out = HotRows{};
        err = what + "read failed";
        return false;
    }
    for (uint64_t i = 0; i < n; ++i) {
        if (out.ids[i] >= table_rows || (i > 0 && out.ids[i] <= out.ids[i - 1])) {
            out = HotRows{};
            err = what + "its row ids are not ascending inside the table (a damaged file)";
            return false;
        }
    }
    return true;
}

}  // namespace strata::ngram
