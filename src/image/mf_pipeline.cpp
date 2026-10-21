// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/mf_pipeline.cpp - see include/strata/image/mf_pipeline.hpp.

#include "strata/image/mf_pipeline.hpp"

#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/image/qi_dit.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace strata::image {

namespace gpu = strata::gpu;

namespace {

constexpr uint64_t kMiB = 1ull << 20;
constexpr uint64_t kScratch = 384 * kMiB;   // the products' dequantization scratch (the DiT's and the encoder's)
constexpr int kTurboSteps = 4;              // Mage-Flow-Turbo's steps (its model card)

uint64_t clampu(uint64_t v, uint64_t lo, uint64_t hi) { return std::max(lo, std::min(v, hi)); }

}  // namespace

std::vector<float> mage_sigmas(int n, float shift) {
    std::vector<float> s;
    for (int i = 0; i < n; ++i) {
        const double t = n > 1 ? 1.0 - (1.0 - 1.0 / n) * i / (n - 1) : 1.0;
        s.push_back((float) (shift * t / (1.0 + (shift - 1.0) * t)));
    }
    s.push_back(0.0f);
    return s;
}

bool MfPipeline::open(const std::string& dit, const std::string& vae, const std::string& text, std::string& err,
                      bool q8) {
    s_ = core::Runtime::get().create_stream();
    dit_path_ = dit;
    vae_path_ = vae;
    text_path_ = text;
    q8_ = q8;
    size_t free = 0, total = 0;
    gpu::mem_info(&free, &total);
    te_size_ = std::make_unique<QiText>();
    dit_size_ = std::make_unique<MfDit>();
    vae_size_ = std::make_unique<MfVae>();
    if (!(te_size_->open(text, s_, err, true) && dit_size_->open(dit, s_, err, true, q8_) &&
          vae_size_->open(vae, s_, err, true)))
        return false;
    auto_budget_ = vram_cap(free);
    return true;
}

void MfPipeline::ensure_open() {
    if (dit_ && vae_ && te_) return;
    std::string err;
    te_ = std::make_unique<QiText>();
    dit_ = std::make_unique<MfDit>();
    vae_ = std::make_unique<MfVae>();
    if (!(te_->open(text_path_, s_, err) && dit_->open(dit_path_, s_, err, false, q8_) &&
          vae_->open(vae_path_, s_, err))) {
        te_.reset();
        dit_.reset();
        vae_.reset();
        throw std::runtime_error(err);
    }
    te_->set_final_norm(true);
    dit_->set_block_budget(dit_->block_min());   // nothing resident until a request plans its budget
    te_->set_layer_budget(te_->layer_min());
    dit_set_ = dit_->block_min();
    te_set_ = te_->layer_min();
    mlp_set_ = -1;
}

uint64_t MfPipeline::held() const {
    if (!(dit_ && vae_ && te_)) return 0;
    return dit_->weight_bytes() + vae_->weight_bytes() + kScratch + dit_->blocks_vram() + te_->layers_vram() +
           dit_->activations_held() + vae_->scratch_held();
}

void MfPipeline::fit(uint64_t reserve) {
    size_t free = 0, total = 0;
    gpu::mem_info(&free, &total);
    const uint64_t have = held() + (uint64_t) free;
    set_budget(have > reserve ? have - reserve : 0);
}

QiNeed MfPipeline::need(int64_t n_text, int64_t width, int64_t height,
                        const std::vector<std::pair<int64_t, int64_t>>& refs, bool mask, int64_t n_neg) {
    (void) refs;
    (void) mask;
    const MfDit& dit = *dit_size_;
    const QiText& te = *te_size_;
    const MfVae& vae = *vae_size_;
    const int64_t hw = (height / 16) * (width / 16), n = n_neg > 0 ? 2 : 1, max_text = std::max(n_text, n_neg);
    const uint64_t fixed = dit.weight_bytes() + vae.weight_bytes() + kScratch;
    const uint64_t te_act = te.activation_bytes(std::max(n_text, n_neg) + 64);   // the system turn's ids too
    QiNeed nd;
    nd.least = fixed + std::max({dit.activation_bytes(hw, n, max_text, 512), vae.scratch_bytes(height / 16, width / 16),
                                 te_act}) +
               dit.block_min() + te.layer_min();
    nd.full = fixed + dit.block_bytes() + te.layer_min() +
              std::max({dit.activation_bytes(hw, n, max_text), vae.scratch_bytes(height / 16, width / 16), te_act});
    nd.held = held();
    size_t free = 0, total = 0;
    gpu::mem_info(&free, &total);
    nd.free = free;
    nd.cap = vram_cap(total);
    return nd;
}

void MfPipeline::set_budget(uint64_t bytes) {
    budget_ = bytes;
    if (!(dit_ && vae_ && te_)) return;
    (void) plan(0, 1, 0, 0);   // shrink at once what no longer fits
}

void MfPipeline::release(bool all) {
    if (!(dit_ && vae_ && te_)) return;
    if (all) {
        gpu::stream_sync(s_);
        te_.reset();
        dit_.reset();
        vae_.reset();
        return;
    }
    dit_->drop_blocks();
    te_->drop_layers();
    dit_->drop_activations();
    vae_->drop_scratch();
    dit_set_ = te_set_ = 0;
    mlp_set_ = -1;
}

QiPlan MfPipeline::plan(int64_t hw, int n, int64_t max_text, int64_t n_ids) {
    QiPlan p;
    p.budget = budget();
    p.fixed = dit_->weight_bytes() + vae_->weight_bytes() + kScratch;
    if (p.budget <= p.fixed)
        throw std::runtime_error("image engine: the VRAM budget (" + std::to_string(p.budget >> 20) +
                                 " MiB) is below the weights that always stay");
    const uint64_t avail = p.budget - p.fixed, dit_min = dit_->block_min(), te_min = te_->layer_min();
    // the DiT's activations, the MLP in bands of rows when the whole takes more than a third of what is there
    int64_t mlp = 0;
    p.activations = hw > 0 ? dit_->activation_bytes(hw, n, max_text) : 0;
    for (int64_t rows : {4096, 2048, 1024, 512}) {
        if (hw == 0 || p.activations * 3 <= avail) break;
        mlp = rows;
        p.activations = dit_->activation_bytes(hw, n, max_text, rows);
    }
    const uint64_t work = std::max(p.activations, te_->activation_bytes(n_ids));
    if (avail < work + dit_min + te_min)
        throw std::runtime_error("image engine: the VRAM budget (" + std::to_string(p.budget >> 20) +
                                 " MiB) is too small for this image");
    p.blocks = clampu(avail - work - te_min, dit_min, dit_->block_bytes());
    p.layers = clampu(avail - work - p.blocks, te_min, te_->layer_bytes());
    p.mlp_rows = mlp;
    if (p.blocks != dit_set_) {
        dit_->set_block_budget(p.blocks);
        dit_set_ = p.blocks;
    }
    if (p.layers != te_set_) {
        te_->set_layer_budget(p.layers);
        te_set_ = p.layers;
    }
    if (hw > 0 && mlp != mlp_set_) {
        dit_->set_mlp_rows(mlp);
        mlp_set_ = mlp;
    }
    if (hw == 0) dit_->drop_activations();
    p.resident_blocks = dit_->resident_blocks();
    p.resident_layers = te_->resident_layers();
    p.vae = avail - p.blocks - p.layers;
    return p;
}

std::vector<uint8_t> MfPipeline::generate(const QiRequest& r, QiTimes* times, QiPlan* plan_out,
                                          const std::function<bool(int, int)>& step,
                                          const std::function<void(const char*)>& stage) {
    using clock = std::chrono::steady_clock;
    if (r.width % 16 != 0 || r.height % 16 != 0 || r.width <= 0 || r.height <= 0)
        throw std::invalid_argument("image engine: the size must be a multiple of 16");
    if (!r.refs.empty() || !r.mask.rgba.empty())
        throw std::invalid_argument("image engine: Mage-Flow's edits are not here yet");
    if ((int64_t) r.ids.size() <= r.drop) throw std::invalid_argument("image engine: no prompt after the system turn");
    const bool guided = r.cfg > 1.0f && !r.neg_ids.empty();
    if (guided && (int64_t) r.neg_ids.size() <= r.drop)
        throw std::invalid_argument("image engine: no negative prompt after the system turn");
    const int64_t h = r.height / 16, w = r.width / 16, HW = r.width * r.height;
    const auto t0 = clock::now();
    ensure_open();
    if (stage) stage("text");
    const int64_t n_ctx = (int64_t) r.ids.size() - r.drop,
                  n_neg = guided ? (int64_t) r.neg_ids.size() - r.drop : 0;
    QiPlan p = plan(h * w, guided ? 2 : 1, std::max(n_ctx, n_neg),
                    (int64_t) std::max(r.ids.size(), r.neg_ids.size()));
    if (r.ids != last_ids_) {
        last_states_ = te_->encode(r.ids);
        last_ids_ = r.ids;
    } else {
        p.text_cached = true;
    }
    const int64_t C = te_->hidden();
    dit_->set_context(0, last_states_.data() + r.drop * C, n_ctx);
    if (guided) {
        if (r.neg_ids != last_neg_ids_) {
            last_neg_states_ = te_->encode(r.neg_ids);
            last_neg_ids_ = r.neg_ids;
        }
        dit_->set_context(1, last_neg_states_.data() + r.drop * C, n_neg);
    } else {
        dit_->set_context(1, nullptr, 0);
    }
    te_->drop_activations();
    const auto t1 = clock::now();
    // the seed's noise as Qwen-Image's ([channels][h][w], PyTorch's CUDA randn), as the latent's tokens [h w][channels]
    const int64_t Cl = dit_->latent_channels(), n = Cl * h * w;
    const auto noise = philox_randn(r.seed, (uint64_t) n);
    std::vector<float> tokens((size_t) n);
    for (int64_t c = 0; c < Cl; ++c)
        for (int64_t q = 0; q < h * w; ++q) tokens[(size_t) (q * Cl + c)] = noise[(size_t) (c * h * w + q)];
    float *x = nullptr, *rgba = nullptr;
    if (!gpu::alloc_device(&x, (size_t) n * 4) || !gpu::alloc_device(&rgba, (size_t) (4 * HW) * 4))
        throw std::runtime_error(std::string("image engine: GPU memory: ") + gpu::last_error());
    std::vector<float> out((size_t) (4 * HW));
    try {
        gpu::copy(x, tokens.data(), (size_t) n * 4);
        if (stage) stage("sampling");
        dit_->sample(x, h, w, r.sigmas.empty() ? mage_sigmas(kTurboSteps) : r.sigmas, guided ? r.cfg : 1.0f, step);
        const auto t2 = clock::now();
        if (stage) stage("decode");
        // the VAE beside the DiT's activations when it fits there; else they give way to it, then the encoder's
        // layers (the next prompt streams them back)
        uint64_t room = p.vae > p.activations ? p.vae - p.activations : 0;
        if (vae_->scratch_bytes(h, w) > room) {
            dit_->drop_activations();
            room = p.vae;
        }
        if (vae_->scratch_bytes(h, w) > room && te_->layers_vram() > 0) {
            te_->drop_layers();
            te_set_ = 0;
            room += p.layers;
        }
        vae_->decode(x, h, w, rgba);
        if (vae_->scratch_bytes(h, w) > p.vae) vae_->drop_scratch();
        gpu::copy(out.data(), rgba, out.size() * 4);
        const auto t3 = clock::now();
        if (times != nullptr) {
            times->text = std::chrono::duration<double>(t1 - t0).count();
            times->sampling = std::chrono::duration<double>(t2 - t1).count();
            times->decode = std::chrono::duration<double>(t3 - t2).count();
        }
    } catch (...) {
        gpu::free(x);
        gpu::free(rgba);
        throw;
    }
    gpu::free(x);
    gpu::free(rgba);
    if (plan_out != nullptr) *plan_out = p;
    std::vector<uint8_t> img((size_t) (4 * HW));
    for (int64_t q = 0; q < HW; ++q)
        for (int c = 0; c < 4; ++c)
            img[(size_t) (q * 4 + c)] =
                (uint8_t) std::lround(std::clamp(out[(size_t) (c * HW + q)], 0.0f, 1.0f) * 255.0f);
    return img;
}

}  // namespace strata::image
