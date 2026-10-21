// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/mf_vae.hpp - Mage-Flow's VAE (Mage-VAE) decoder on the GPU: the DiT's latent tokens [h w][128]
// to an RGBA image [4][16 h][16 w] in 0 .. 1.
//
// The decoder is the reference's DConvDenoiser run once at t = 0 from zero noise: the CoD decoder turns the latent into
// a condition [h w][384] (3 x 3 convolutions, group norms, self-attention in 32 x 32 windows), 21 DiCo blocks (their
// adaLN modulation constant at t = 0, worked out when loaded) refine a per-patch state from it, and a small MLP takes
// each pixel of a 16 x 16 patch to RGB.  The activations are channels-last ([pixel][channel], FP32) at the latent's
// size; the group norms and the blocks' channel attention take statistics over the whole latent, so it is not tiled
// (2048 x 2048 is 128 x 128 there); the per-pixel MLP runs a band of patches at a time.
#pragma once

#include "strata/prefill/gemm.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::image {

class MfVae {
public:
    MfVae();
    ~MfVae();
    MfVae(const MfVae&) = delete;
    MfVae& operator=(const MfVae&) = delete;

    /// Reads the VAE's safetensors (the reference's names: pipeline.* the decoder, student.* the encoder) and puts the
    /// decoder's weights on the current GPU.  `dry`: read for its sizes only, nothing put on the GPU.
    bool open(const std::string& path, void* stream, std::string& err, bool dry = false);

    /// rgba [4][16 h][16 w] (device FP32, 0 .. 1) from the DiT's latent tokens z [h w][128] (device FP32).
    void decode(const float* z, int64_t h, int64_t w, float* rgba);
    /// The image before the clamp, [3][16 h][16 w] (device FP32, about -1 .. 1): the reference's decode output.
    void decode_raw(const float* z, int64_t h, int64_t w, float* rgb);

    /// The GPU memory a decode of an h x w latent works in (its buffers, besides the weights).
    uint64_t scratch_bytes(int64_t h, int64_t w) const;
    void drop_scratch();
    uint64_t scratch_held() const { return cap_hw_ > 0 ? scratch_bytes(cap_hw_, 1) : 0; }
    uint64_t weight_bytes() const { return weight_bytes_; }

    /// The CoD decoder's output [h w][384] and the 21 blocks' output [h w][384] of the last decode (device FP32), for
    /// the probe.
    const float* last_cond() const { return cond_; }
    const float* last_state() const { return s_state_; }

private:
    struct Conv {
        int64_t out = 0, in = 0, k = 1;
        const uint16_t* w = nullptr;   // [out][k][k][in] FP16
        const float* b = nullptr;      // null: no bias
    };
    struct Res {
        const float *n1w = nullptr, *n1b = nullptr, *n2w = nullptr, *n2b = nullptr;
        Conv c1, c2;
    };
    struct Attn {
        const float *nw = nullptr, *nb = nullptr;
        Conv qkv, proj;   // qkv: q, k and v's 1 x 1 convolutions as one of 3 C outputs
    };
    struct DiCo {
        Conv c1, c4, c3, c5;
        const float *dw = nullptr, *db = nullptr;   // the depthwise 3 x 3 convolution [C][9], its bias
        const float *caw = nullptr, *cab = nullptr; // the channel attention's 1 x 1 convolution [C][C] (FP32)
        const float* mod = nullptr;                  // [6][C]: shift, scale, gate (attention), shift, scale, gate (MLP)
    };
    Conv conv_w(class SafeTensors& st, const std::string& name, bool bias = true, int64_t col0 = 0,
                int64_t cols = -1);
    const float* vec(class SafeTensors& st, const std::string& name);
    const float* upload(const std::vector<float>& v);
    void conv(const Conv& c, const uint16_t* in16, int64_t H, int64_t W, float* out);
    void gn_silu16(const float* x, const float* w, const float* b, int64_t HW, int64_t C, bool silu, uint16_t* y16);
    void resblock(const Res& r, float*& x, int64_t H, int64_t W);
    void attention(const Attn& a, float* x, int64_t H, int64_t W);
    void dico(const DiCo& d, float* s, int64_t H, int64_t W);
    void grow(int64_t h, int64_t w);

    void* s_ = nullptr;
    prefill::Gemm gemm_;
    std::vector<void*> allocs_;
    uint64_t weight_bytes_ = 0;
    bool dry_ = false;
    int64_t C_ = 384, F_ = 1536;
    // the CoD decoder
    Conv cin_, cout_;
    Res res_[3];
    Attn attn_[2];
    const float *gnw_ = nullptr, *gnb_ = nullptr;
    // the denoiser
    Conv s_in_;        // s_embedder.proj2's condition columns (x is zero: proj1's part vanishes)
    std::vector<DiCo> blocks_;
    Conv yx_, ce_;     // y_embedder_x [8192][384], dec_net.cond_embed [8192][384]
    const float* px_ = nullptr;   // the per-pixel MLP's weights (FP32, mf_vae.cpp's PixW layout)
    const float* dct_ = nullptr;  // [256][64] the NerfEmbedder's DCT features of a patch's pixels
    // scratch, grown on demand
    int64_t cap_hw_ = 0, cap_band_ = 0;
    float *x_ = nullptr, *t0_ = nullptr, *t1_ = nullptr, *t2_ = nullptr, *cond_ = nullptr, *s_state_ = nullptr;
    float *gn_part_ = nullptr, *vec_ = nullptr, *yxb_ = nullptr, *ceb_ = nullptr, *scores_ = nullptr;
    uint16_t *a16_ = nullptr, *col16_ = nullptr, *w16_ = nullptr;
    int64_t col_cap_ = 0;
};

}  // namespace strata::image
