// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/ngram/ple_hot_test.cpp - --ple-hot's file against a synthetic table: the rows it keeps come back as the table
// holds them without a read, the other rows are still read, and a file that does not belong to the table is refused.
//
//   ple_hot_test [--dir D]        CPU and disk only, no model, no GPU
#include "strata/ngram/ple_hot.hpp"
#include "strata/ngram/ple_reader.hpp"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace ng = strata::ngram;

namespace {

int g_fail = 0;
#define CHECK(c, ...)                                                 \
    do {                                                              \
        if (!(c)) {                                                   \
            std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
            std::fprintf(stderr, __VA_ARGS__);                        \
            std::fprintf(stderr, "\n");                               \
            ++g_fail;                                                 \
        }                                                             \
    } while (0)

constexpr uint64_t HEADER = 192;      // the real shard's data offset, so rows are misaligned the same way
constexpr uint32_t RB = ng::ROW_BYTES;
constexpr uint32_t N_ROWS = 300000;   // ~73 buckets of the pinned index

void expected_row(uint32_t row, uint8_t* out, uint32_t salt = 0) {
    for (uint32_t b = 0; b < RB; ++b) out[b] = (uint8_t) (((row ^ salt) * 2654435761u + b * 97u) >> 7);
    std::memcpy(out, &row, 4);
}

bool make_table(const std::string& path, uint32_t salt = 0) {
    std::ofstream f(path, std::ios::binary);
    std::vector<uint8_t> head(HEADER, 0xAB);
    f.write((const char*) head.data(), (std::streamsize) head.size());
    std::vector<uint8_t> r(RB);
    for (uint32_t i = 0; i < N_ROWS; ++i) {
        expected_row(i, r.data(), salt);
        f.write((const char*) r.data(), RB);
    }
    return (bool) f;
}

struct HotFile {
    uint32_t version = ng::HOT_VERSION;
    uint32_t row_bytes = RB;
    uint64_t table_rows = N_ROWS;
    uint64_t table_offset = HEADER;
    uint32_t salt = 0;                // the rows (and the sample) of the table with this salt
    bool reversed = false;            // ids out of order
    uint64_t cut = 0;                 // bytes left off the end
};

void put32(std::ofstream& f, uint32_t v) { f.write(reinterpret_cast<const char*>(&v), 4); }
void put64(std::ofstream& f, uint64_t v) { f.write(reinterpret_cast<const char*>(&v), 8); }

void write_hot(const std::string& path, const std::vector<uint32_t>& ids, const HotFile& h) {
    std::vector<uint8_t> sample;
    std::vector<uint8_t> r(RB);
    for (uint32_t s : ng::hot_sample_rows(h.table_rows)) {
        expected_row(s, r.data(), h.salt);
        sample.insert(sample.end(), r.begin(), r.end());
    }
    std::vector<uint32_t> order(ids);
    if (h.reversed) std::swap(order.front(), order.back());
    {
        std::ofstream f(path, std::ios::binary);
        f.write(ng::HOT_MAGIC, sizeof ng::HOT_MAGIC);
        put32(f, h.version);
        put32(f, h.row_bytes);
        put64(f, h.table_rows);
        put64(f, h.table_offset);
        put64(f, ng::fnv1a64(sample.data(), sample.size()));
        put64(f, order.size());
        for (uint32_t id : order) put32(f, id);
        for (uint32_t id : order) {
            expected_row(id, r.data(), h.salt);
            f.write((const char*) r.data(), RB);
        }
    }
    if (h.cut > 0) std::filesystem::resize_file(path, std::filesystem::file_size(path) - h.cut);
}

/// Every hot id: every 7th row, and the first and last rows of the table (the index's edges).
std::vector<uint32_t> hot_ids() {
    std::vector<uint32_t> ids;
    for (uint32_t r = 0; r < N_ROWS; r += 7) ids.push_back(r);
    if (ids.back() != N_ROWS - 1) ids.push_back(N_ROWS - 1);
    return ids;
}

bool rows_match(ng::PleReader& rd, const std::vector<uint32_t>& rows, const char* what) {
    std::vector<uint8_t> out(rows.size() * RB, 0xCC), want(RB);
    std::string err;
    const auto t = rd.issue(rows.data(), rows.size(), out.data());
    if (!rd.collect(t, err)) { CHECK(false, "%s: collect failed: %s", what, err.c_str()); return false; }
    for (size_t i = 0; i < rows.size(); ++i) {
        if (rows[i] >= N_ROWS) std::memset(want.data(), 0, RB);
        else expected_row(rows[i], want.data());
        if (std::memcmp(want.data(), &out[i * RB], RB) != 0) {
            CHECK(false, "%s: row %u (index %zu) differs", what, rows[i], i);
            return false;
        }
    }
    return true;
}

bool refused(const std::string& table, const std::string& hot, const char* expect, const char* what) {
    ng::PleReader rd;
    std::string err;
    if (!rd.open(table, HEADER, N_ROWS, 16, 0, err)) { CHECK(false, "%s: open: %s", what, err.c_str()); return false; }
    ng::HotRows got;
    const bool ok = ng::load_hot_rows(hot, rd, N_ROWS, HEADER, got, err);
    CHECK(!ok, "%s: the file was taken", what);
    CHECK(err.find(expect) != std::string::npos, "%s: message \"%s\" lacks \"%s\"", what, err.c_str(), expect);
    CHECK(got.ids.empty() && got.bytes.empty(), "%s: rows were left behind", what);
    return !ok;
}

}  // namespace

int main(int argc, char** argv) {
    std::string dir = std::filesystem::temp_directory_path().string();
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--dir" && i + 1 < argc) dir = argv[++i];
        else { std::fprintf(stderr, "usage: ple_hot_test [--dir D]\n"); return 2; }
    }
    const std::string table = dir + "/ple_hot_test.table", hot = dir + "/ple_hot_test.hot";
    if (!make_table(table)) { std::fprintf(stderr, "cannot write %s\n", table.c_str()); return 2; }
    const std::vector<uint32_t> ids = hot_ids();
    write_hot(hot, ids, HotFile{});

    {   // the file loads, and its rows are served without a read
        ng::PleReader rd;
        std::string err;
        CHECK(rd.open(table, HEADER, N_ROWS, 16, 0, err), "open: %s", err.c_str());
        ng::HotRows got;
        CHECK(ng::load_hot_rows(hot, rd, N_ROWS, HEADER, got, err), "load: %s", err.c_str());
        CHECK(got.ids == ids, "the ids came back different (%zu of %zu)", got.ids.size(), ids.size());
        rd.set_pinned(std::move(got.ids), std::move(got.bytes));
        CHECK(rd.pinned_rows() == ids.size(), "pinned %llu rows, not %zu", (unsigned long long) rd.pinned_rows(),
              ids.size());

        rd.reset_stats();
        const uint64_t done0 = ng::reads_done();
        const std::vector<uint32_t> all_hot = {0, 7, 14, 4095 * 7, N_ROWS - 1, 299999 / 7 * 7, 7};
        rows_match(rd, all_hot, "hot rows");
        ng::ReaderStats s = rd.snapshot();
        CHECK(s.reads == 0, "hot rows were read: %llu reads", (unsigned long long) s.reads);
        CHECK(ng::reads_done() == done0, "reads_done counted %llu reads for hot rows",
              (unsigned long long) (ng::reads_done() - done0));
        CHECK(s.pinned_hits == all_hot.size(), "%llu hot hits, not %zu", (unsigned long long) s.pinned_hits,
              all_hot.size());

        rd.reset_stats();
        const std::vector<uint32_t> mixed = {1, 7, 8, 4096, 4097, 28672, 299998, N_ROWS - 1, N_ROWS + 5, 6};
        rows_match(rd, mixed, "hot and cold rows");
        s = rd.snapshot();
        CHECK(s.pinned_hits == 3, "%llu hot hits among the mixed rows, not 3 (7, 28672, the last)",
              (unsigned long long) s.pinned_hits);
        CHECK(s.reads > 0, "the cold rows were not read");
        CHECK(ng::reads_done() - done0 == s.reads, "reads_done counted %llu of the %llu reads (the watchdog's I/O)",
              (unsigned long long) (ng::reads_done() - done0), (unsigned long long) s.reads);

        rd.close();
        CHECK(rd.pinned_rows() == 0, "close kept %llu pinned rows", (unsigned long long) rd.pinned_rows());
    }

    const std::string other = dir + "/ple_hot_test.other";
    write_hot(other, ids, HotFile{.salt = 0x5A5A});
    refused(table, other, "another table of the same shape", "another table's rows");
    write_hot(other, ids, HotFile{.row_bytes = 160});
    refused(table, other, "cut from another table", "another row size");
    write_hot(other, ids, HotFile{.table_offset = HEADER + 32});
    refused(table, other, "cut from another table", "another offset");
    write_hot(other, ids, HotFile{.version = 2});
    refused(table, other, "version 2", "another version");
    write_hot(other, ids, HotFile{.cut = 45});
    refused(table, other, "partial file", "a cut file");
    write_hot(other, ids, HotFile{.reversed = true});
    refused(table, other, "not ascending", "ids out of order");
    write_hot(other, {5, 5, 9}, HotFile{});
    refused(table, other, "not ascending", "a repeated id");
    refused(table, dir + "/ple_hot_test.missing", "cannot read it", "no file");
    {
        std::ofstream f(other, std::ios::binary);
        f << "not a hot file, but long enough to hold a header of 48 bytes";
    }
    refused(table, other, "not a hot-rows file", "another kind of file");

    std::filesystem::remove(table);
    std::filesystem::remove(hot);
    std::filesystem::remove(other);
    if (g_fail != 0) {
        std::fprintf(stderr, "ple_hot_test: %d failure(s)\n", g_fail);
        return 1;
    }
    std::printf("ple_hot_test: ok (%zu hot rows of %u)\n", ids.size(), N_ROWS);
    return 0;
}
