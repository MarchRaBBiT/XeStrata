// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/mf_probe.cpp - strata-image-mf-probe: Mage-Flow's parts on the GPU against the reference implementation's
// outputs of one run (the .npy files of records' mage-flow-golden/golden.py).
//
//   strata-image-mf-probe text TEXT_ENCODER.gguf GOLDEN_DIR
//
// Prints how far each part's output is from the run's (cosine and relative L1) and its time.

#include "strata/core/runtime.hpp"
#include "strata/image/qi_text.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// a .npy file's data as T (its dtype '<f4' or '<i8', C order), and its shape
template<class T> std::vector<T> npy(const std::string& path, std::vector<int64_t>* shape = nullptr) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    char magic[8];
    f.read(magic, 8);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0) throw std::runtime_error(path + ": not a .npy file");
    uint32_t hlen = 0;
    if (magic[6] == 1) {
        uint16_t h = 0;
        f.read(reinterpret_cast<char*>(&h), 2);
        hlen = h;
    } else {
        f.read(reinterpret_cast<char*>(&hlen), 4);
    }
    std::string header(hlen, '\0');
    f.read(header.data(), hlen);
    const char* want = sizeof(T) == 8 ? "'<i8'" : "'<f4'";
    if (header.find(want) == std::string::npos || header.find("'fortran_order': False") == std::string::npos)
        throw std::runtime_error(path + ": expected " + want + " in C order: " + header);
    const size_t l = header.find('('), r = header.find(')');
    std::vector<int64_t> dims;
    size_t n = 1;
    for (size_t i = l + 1; i < r;) {
        size_t j = header.find_first_of(",)", i);
        const std::string tok = header.substr(i, j - i);
        if (tok.find_first_not_of(' ') != std::string::npos) {
            dims.push_back(std::stoll(tok));
            n *= (size_t) dims.back();
        }
        i = j + 1;
    }
    std::vector<T> v(n);
    f.read(reinterpret_cast<char*>(v.data()), (std::streamsize) (n * sizeof(T)));
    if (!f) throw std::runtime_error(path + ": short");
    if (shape) *shape = dims;
    return v;
}

void compare(const char* what, const float* got, const float* want, size_t n) {
    double dot = 0, ng = 0, nw = 0, err = 0, mag = 0;
    for (size_t i = 0; i < n; ++i) {
        dot += (double) got[i] * want[i];
        ng += (double) got[i] * got[i];
        nw += (double) want[i] * want[i];
        err += std::fabs((double) got[i] - want[i]);
        mag += std::fabs((double) want[i]);
    }
    std::printf("%-10s cos %.6f  rel %.3g  (%zu values)\n", what, dot / std::sqrt(ng * nw), err / mag, n);
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// the text encoder: each prompt of the run (ids.npy, ids_lens.npy) encoded alone, its rows after the 34 of the
// system turn against txt.npy
int text(const std::string& gguf, const std::string& dir) {
    constexpr int64_t kDrop = 34;
    void* s = strata::core::Runtime::get().create_stream();
    strata::image::QiText te;
    std::string err;
    if (!te.open(gguf, s, err)) throw std::runtime_error(err);
    te.set_final_norm(true);
    const auto ids = npy<int64_t>(dir + "/ids.npy");
    const auto lens = npy<int64_t>(dir + "/ids_lens.npy");
    std::vector<int64_t> shape;
    const auto want = npy<float>(dir + "/txt.npy", &shape);
    const int64_t C = te.hidden();
    if (shape.size() != 2 || shape[1] != C) throw std::runtime_error("txt.npy is not [rows][hidden]");
    std::vector<float> got;
    size_t off = 0;
    for (int64_t len : lens) {
        std::vector<int32_t> p(ids.begin() + (long) off, ids.begin() + (long) (off + len));
        off += (size_t) len;
        te.encode(p);   // warm
        const auto t0 = std::chrono::steady_clock::now();
        const auto h = te.encode(p);
        std::printf("encode %lld tokens: %.1f ms\n", (long long) len, 1e3 * seconds_since(t0));
        got.insert(got.end(), h.begin() + kDrop * C, h.end());
    }
    if ((int64_t) got.size() != shape[0] * C) throw std::runtime_error("the rows differ from txt.npy's");
    compare("text", got.data(), want.data(), got.size());
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string what = argc > 1 ? argv[1] : "";
        if (what == "text" && argc == 4) return text(argv[2], argv[3]);
        std::fprintf(stderr, "usage: %s text TEXT_ENCODER.gguf GOLDEN_DIR\n", argv[0]);
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-image-mf-probe: %s\n", e.what());
        return 1;
    }
}
