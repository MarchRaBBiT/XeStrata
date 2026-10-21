// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/mf_pipeline.hpp - Mage-Flow's text to image on the GPU: the text encoder's states (Qwen3-VL-4B,
// after its final norm) of the prompt's ids, the DiT's flow sampling from the seed's noise, and Mage-VAE's image,
// within a VRAM budget.
//
// The budget is split per request as QiPipeline splits it: the weights that always stay (the DiT's parts outside its
// blocks, the VAE's) and the products' scratch first, then the DiT's activations (the MLP in bands of rows when the
// whole would take too much), then the DiT's blocks (used at every step) past the text encoder's two slots, then the
// encoder's layers.  The VAE decodes in what is left, the DiT's activations freed when it needs them (it is not tiled:
// its group norms take the whole latent, a few MiB at 2048 x 2048).  The same prompt's text states are kept.
#pragma once

#include "strata/image/mf_dit.hpp"
#include "strata/image/mf_vae.hpp"
#include "strata/image/pipeline.hpp"
#include "strata/image/qi_text.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::image {

/// Mage-Flow's static-shift schedule (shift s) of n steps: sigma = s t / (1 + (s - 1) t) for t from 1 down to 1 / n,
/// and the terminal 0 (the reference's FlowMatchEulerDiscreteScheduler, shift 6).
std::vector<float> mage_sigmas(int n, float shift = 6.0f);

class MfPipeline : public ImagePipeline {
public:
    /// Reads the parts' files for their sizes; the parts go on the GPU with the first request.  `q8`: the DiT's blocks
    /// as Q8_0 (else bf16).
    bool open(const std::string& dit, const std::string& vae, const std::string& text, std::string& err,
              bool q8 = true);

    void set_budget(uint64_t bytes) override;
    uint64_t budget() const override { return budget_ > 0 ? budget_ : auto_budget_; }
    void release(bool all = false) override;
    void fit(uint64_t reserve) override;
    uint64_t held() const override;
    QiNeed need(int64_t n_text, int64_t width, int64_t height,
                const std::vector<std::pair<int64_t, int64_t>>& refs = {}, bool mask = false,
                int64_t n_neg = 0) override;
    std::vector<uint8_t> generate(const QiRequest& r, QiTimes* times = nullptr, QiPlan* plan = nullptr,
                                  const std::function<bool(int, int)>& step = {},
                                  const std::function<void(const char*)>& stage = {}) override;

private:
    QiPlan plan(int64_t hw, int n, int64_t max_text, int64_t n_ids);
    void ensure_open();

    void* s_ = nullptr;
    std::string dit_path_, vae_path_, text_path_;
    bool q8_ = true;
    std::unique_ptr<QiText> te_;
    std::unique_ptr<MfDit> dit_;
    std::unique_ptr<MfVae> vae_;
    std::unique_ptr<QiText> te_size_;
    std::unique_ptr<MfDit> dit_size_;
    std::unique_ptr<MfVae> vae_size_;
    uint64_t budget_ = 0, auto_budget_ = 0, dit_set_ = 0, te_set_ = 0;
    int64_t mlp_set_ = -1;
    std::vector<int32_t> last_ids_, last_neg_ids_;
    std::vector<float> last_states_, last_neg_states_;
};

}  // namespace strata::image
