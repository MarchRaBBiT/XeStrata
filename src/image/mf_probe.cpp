// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/mf_probe.cpp - strata-image-mf-probe: Mage-Flow's parts on the GPU against the reference implementation's
// outputs of one run (the .npy files of records' mage-flow-golden/golden.py).
//
//   strata-image-mf-probe text TEXT_ENCODER.gguf GOLDEN_DIR
//   strata-image-mf-probe vae VAE.safetensors GOLDEN_DIR   (golden_vae.py's vae_cond, vae_s, vae_img)
//   strata-image-mf-probe dit DIT.safetensors GOLDEN_DIR [q8|bf16 [VAE.safetensors OUT.f32]]
//       (block 0, the first velocity, the final latent; with the VAE the image as FP32 [3][H][W] to OUT.f32)
//
// Prints how far each part's output is from the run's (cosine and relative L1) and its time.

#include "strata/core/runtime.hpp"
#include "strata/core/gpu.hpp"
#include "strata/image/mf_dit.hpp"
#include "strata/image/mf_vae.hpp"
#include "strata/image/qi_text.hpp"

#include <algorithm>
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

// [C][HW] (the reference's channels-first) of a [HW][C] tensor
std::vector<float> channels_first(const std::vector<float>& x, int64_t HW, int64_t C) {
    std::vector<float> y(x.size());
    for (int64_t p = 0; p < HW; ++p)
        for (int64_t c = 0; c < C; ++c) y[(size_t) (c * HW + p)] = x[(size_t) (p * C + c)];
    return y;
}

std::vector<float> fetch(const float* d, size_t n) {
    std::vector<float> h(n);
    strata::gpu::copy(h.data(), d, n * 4);
    return h;
}

float* put(const std::vector<float>& h) {
    float* d = nullptr;
    if (!strata::gpu::alloc_device(&d, h.size() * 4)) throw std::runtime_error("GPU memory");
    strata::gpu::copy(d, h.data(), h.size() * 4);
    return d;
}

// the VAE's decode of the run's latent, its condition, blocks' state and image against golden_vae.py's
int vae(const std::string& path, const std::string& dir) {
    void* s = strata::core::Runtime::get().create_stream();
    strata::image::MfVae v;
    std::string err;
    if (!v.open(path, s, err)) throw std::runtime_error(err);
    std::vector<int64_t> shape;
    const auto lat = npy<float>(dir + "/latent.npy", &shape);   // [1][h w][128]
    std::vector<int64_t> cs;
    const auto cond = npy<float>(dir + "/vae_cond.npy", &cs);   // [1][384][h][w]
    const int64_t h = cs[2], w = cs[3], C = cs[1], HW = h * w;
    if (shape.size() != 3 || shape[1] != HW) throw std::runtime_error("latent.npy is not [1][h w][128]");
    float* z = put(lat);
    float* rgb = nullptr;
    if (!strata::gpu::alloc_device(&rgb, (size_t) (3 * HW * 256) * 4)) throw std::runtime_error("GPU memory");
    v.decode_raw(z, h, w, rgb);   // warm
    strata::gpu::stream_sync(s);
    const auto t0 = std::chrono::steady_clock::now();
    v.decode_raw(z, h, w, rgb);
    strata::gpu::stream_sync(s);
    std::printf("decode %lld x %lld: %.1f ms\n", (long long) (16 * h), (long long) (16 * w), 1e3 * seconds_since(t0));
    compare("cond", channels_first(fetch(v.last_cond(), (size_t) (HW * C)), HW, C).data(), cond.data(), cond.size());
    const auto st = npy<float>(dir + "/vae_s.npy");
    compare("blocks", channels_first(fetch(v.last_state(), (size_t) (HW * C)), HW, C).data(), st.data(), st.size());
    const auto img = npy<float>(dir + "/vae_img.npy");
    const auto got = fetch(rgb, img.size());
    compare("image", got.data(), img.data(), img.size());
    double se = 0;
    for (size_t i = 0; i < img.size(); ++i) {
        const double a = std::min(1.0, std::max(-1.0, (double) got[i]));
        const double b = std::min(1.0, std::max(-1.0, (double) img[i]));
        se += (a - b) * (a - b) * 127.5 * 127.5;
    }
    std::printf("image PSNR %.2f dB (8-bit)\n", 10 * std::log10(255.0 * 255.0 / (se / (double) img.size())));
    return 0;
}

// the DiT from the run's text states, noise and sigmas: block 0's streams and the first velocity, then the whole
// sampling's latent
int dit(const std::string& path, const std::string& dir, bool q8, const std::string& vae_path = "",
        const std::string& out_path = "") {
    void* s = strata::core::Runtime::get().create_stream();
    strata::image::MfDit d;
    std::string err;
    auto t0 = std::chrono::steady_clock::now();
    if (!d.open(path, s, err, false, q8)) throw std::runtime_error(err);
    std::printf("open (%s): %.1f s\n", q8 ? "Q8_0" : "bf16", seconds_since(t0));
    std::vector<int64_t> ns, ts;
    const auto noise = npy<float>(dir + "/noise.npy", &ns);   // [1][128][h][w]
    const int64_t Cl = ns[1], h = ns[2], w = ns[3], HW = h * w, C = d.hidden();
    const auto txt = npy<float>(dir + "/txt.npy", &ts);
    const auto lens = npy<int64_t>(dir + "/txt_lens.npy");
    const auto sig = npy<float>(dir + "/sigmas.npy");
    std::ifstream af(dir + "/args.json");
    const std::string args((std::istreambuf_iterator<char>(af)), std::istreambuf_iterator<char>());
    const size_t ci = args.find("\"cfg\":");
    const float cfg = ci == std::string::npos ? 1.f : std::stof(args.substr(ci + 6));
    const int64_t D = ts[1];
    d.set_context(0, txt.data(), lens[0]);
    if (lens.size() > 1) d.set_context(1, txt.data() + lens[0] * D, lens[1]);
    const int n = cfg > 1.f && lens.size() > 1 ? 2 : 1;
    std::vector<float> xt((size_t) (HW * Cl));   // the noise as tokens [h w][128]
    for (int64_t c = 0; c < Cl; ++c)
        for (int64_t p = 0; p < HW; ++p) xt[(size_t) (p * Cl + c)] = noise[(size_t) (c * HW + p)];
    float* x = put(xt);
    float* out = nullptr;
    if (!strata::gpu::alloc_device(&out, (size_t) (n * HW * Cl) * 4)) throw std::runtime_error("GPU memory");
    d.keep_block0(true);
    d.forward(x, sig[0], h, w, n, out);
    strata::gpu::stream_sync(s);
    t0 = std::chrono::steady_clock::now();
    d.forward(x, sig[0], h, w, n, out);
    strata::gpu::stream_sync(s);
    std::printf("forward (%d context%s): %.1f ms\n", n, n > 1 ? "s" : "", 1e3 * seconds_since(t0));
    const auto bi = npy<float>(dir + "/blk0_img.npy"), bt = npy<float>(dir + "/blk0_txt.npy");
    compare("blk0 img", fetch(d.block0_img(), (size_t) (HW * C)).data(), bi.data(), (size_t) (HW * C));
    compare("blk0 txt", fetch(d.block0_txt(), (size_t) (lens[0] * C)).data(), bt.data(), (size_t) (lens[0] * C));
    auto o = fetch(out, (size_t) (n * HW * Cl));
    if (n == 2)
        for (size_t i = 0; i < (size_t) (HW * Cl); ++i) o[i] = o[HW * Cl + i] + cfg * (o[i] - o[HW * Cl + i]);
    const auto v0 = npy<float>(dir + "/v0.npy");
    compare("v0", o.data(), v0.data(), v0.size());
    std::vector<float> sigmas(sig.begin(), sig.end());
    t0 = std::chrono::steady_clock::now();
    d.sample(x, h, w, sigmas, cfg);
    strata::gpu::stream_sync(s);
    std::printf("sample %zu steps: %.2f s\n", sigmas.size() - 1, seconds_since(t0));
    const auto lat = npy<float>(dir + "/latent.npy");
    compare("latent", fetch(x, (size_t) (HW * Cl)).data(), lat.data(), lat.size());
    if (!vae_path.empty()) {   // the image: rgb [3][16 h][16 w] FP32 (about -1 .. 1) to out_path
        strata::image::MfVae v;
        if (!v.open(vae_path, s, err)) throw std::runtime_error(err);
        float* rgb = nullptr;
        if (!strata::gpu::alloc_device(&rgb, (size_t) (3 * HW * 256) * 4)) throw std::runtime_error("GPU memory");
        v.decode_raw(x, h, w, rgb);
        strata::gpu::stream_sync(s);
        const auto img = fetch(rgb, (size_t) (3 * HW * 256));
        std::ofstream(out_path, std::ios::binary).write(reinterpret_cast<const char*>(img.data()),
                                                         (std::streamsize) (img.size() * 4));
        std::printf("image: %s\n", out_path.c_str());
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string what = argc > 1 ? argv[1] : "";
        if (what == "text" && argc == 4) return text(argv[2], argv[3]);
        if (what == "vae" && argc == 4) return vae(argv[2], argv[3]);
        if (what == "dit" && (argc == 4 || argc == 5 || argc == 7))
            return dit(argv[2], argv[3], argc == 4 || std::string(argv[4]) != "bf16", argc == 7 ? argv[5] : "",
                       argc == 7 ? argv[6] : "");
        std::fprintf(stderr, "usage: %s text TEXT_ENCODER.gguf GOLDEN_DIR | vae VAE.safetensors GOLDEN_DIR | "
                     "dit DIT.safetensors GOLDEN_DIR [q8|bf16 [VAE.safetensors OUT.f32]]\n", argv[0]);
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-image-mf-probe: %s\n", e.what());
        return 1;
    }
}
