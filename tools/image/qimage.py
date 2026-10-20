# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""tools/image/qimage.py - Qwen-Image-2.1-Turbo and Qwen-Image-2.1 for XeStrata's image endpoint: the models' four
files at pinned Hugging Face revisions, and the run config's "images" section for XeStrata's own image engine
(strata-image, built with the text engine).

setup.py (--images, IMAGE_MODELS) and tools/image/install.py use these pieces; serve/images.py runs the result.
The two share the VAE, the text encoders and the vision encoder; only the DiT differs, the base model's (Turbo's
undistilled one) sampled in 40 steps on the scheduler's shifted sigmas instead of Turbo's 8 fixed ones.

  files     the denoiser (DiT) quant (Unsloth's LPIPS vs bf16: Q4_K_M 0.153, Q6_K 0.055, Q8_0 0.036) and the
            Qwen3-VL-8B text encoder's (AtomicChat's LPIPS of the pictures vs a bf16 encoder: Q8_0 0.031, Q6_K
            0.043, Q5_K 0.066, Q4_K 0.087): the engine streams what does not fit in VRAM, so the GPU's memory does
            not decide them; the RAM does.  Both Q8_0 when the four files fit the RAM the text model leaves; else
            the encoder is lowered first, then the DiT (fit_ram, the user's decision of 2026-10-19), unless the SSD
            offload is on (then the sizes asked for, the rest read from the SSD).  The encoder is Unsloth's
            (--images qwen-image-turbo) or, with --images qwen-image-turbo-uncensored, AtomicChat's abliterated
            one (the same tensors, its refusal direction projected out).  The VAE (bf16).  Each file is checked
            against the SHA-256 the Hugging Face API lists for it at the pinned revision.
"""
from __future__ import annotations

import os
from pathlib import Path

REVISIONS = {                                       # the `sha` of https://huggingface.co/api/models/<repo>, 2026-10-12
    "unsloth/Qwen-Image-2.1-Turbo-GGUF": "5d1f9f759ea877ae556cda88dbec958b62ad3809",
    "unsloth/Qwen-Image-2.1-FP8": "bb21a8ba7f0371c19ad1f72c82a5e8fc10f39939",
    "unsloth/Qwen3-VL-8B-Instruct-GGUF": "b93a7ee713758252c555be4210c00540df954dc2",
    "AtomicChat/Qwen-Image-2.1-Turbo-Uncensored-GGUF": "df96693dbdfb8738c0bf1bbd0337d149808db948",  # 2026-10-19
    "unsloth/Qwen-Image-2.1-GGUF": "2c31ccd392b367a6637841a143813320a02dff55",                    # 2026-10-21
}
_DIT = "unsloth/Qwen-Image-2.1-Turbo-GGUF"
_BASE = "unsloth/Qwen-Image-2.1-GGUF"
_VL = "unsloth/Qwen3-VL-8B-Instruct-GGUF"
_AC = "AtomicChat/Qwen-Image-2.1-Turbo-Uncensored-GGUF"
# name -> (repo, path in the repo, bytes, sha256): the Hub's LFS pointers at the pinned revisions
FILES = {
    # the DiTs, "<dit>-<QUANTS>"
    "turbo-Q4_K_M": (_DIT, "qwen-image-2.1-turbo-Q4_K_M.gguf", 4_199_565_120,
                     "8940e8b522af886e0c0f5afddf720e82dd223cb9a43580b5f61c4d740452ac20"),
    "turbo-Q6_K": (_DIT, "qwen-image-2.1-turbo-Q6_K.gguf", 6_271_551_296,
                   "f4560fb47ad863ffed7e32737d3c42988c44bd0c459606777d7ffe2e895d0a70"),
    "turbo-Q8_0": (_DIT, "qwen-image-2.1-turbo-Q8_0.gguf", 7_640_860_480,
                   "399d2ddc3636f36456e2f5bfa57187da4e38721bc046430317f42a9eab5b1712"),
    "base-Q4_K_M": (_BASE, "qwen-image-2.1-Q4_K_M.gguf", 4_199_565_024,
                    "631d532e7ca71e8d90a87c71d3699761a812039d22e3370e87498d87754660fe"),
    "base-Q6_K": (_BASE, "qwen-image-2.1-Q6_K.gguf", 6_271_551_200,
                  "2ead1ee665e8ebc5fec517d287a167f5c6a7e985acbf3b1b93143437a04616ac"),
    "base-Q8_0": (_BASE, "qwen-image-2.1-Q8_0.gguf", 7_640_860_384,
                  "c0ed4b2ffd56cbe9c3df1e4a4098045256484ebe94ba5a7e4338d35ec046baa5"),
    "vae": ("unsloth/Qwen-Image-2.1-FP8", "vae/qwen_image_2.1_vae_bf16.safetensors", 675_508_656,
            "71879ffd5321e6d10c3c87513e2b474b1252efa7f3dec2969214a9bf06a6dd5c"),
    # the text encoders, "llm-<encoder>-<TEXT_QUANTS>"
    "llm-stock-Q8_0": (_VL, "Qwen3-VL-8B-Instruct-Q8_0.gguf", 8_709_520_224,
                       "cb8616bf6ed228982d9e47d7b72b42195342efa26044b0ee1873e61d9e78d3d7"),
    "llm-stock-Q6_K": (_VL, "Qwen3-VL-8B-Instruct-UD-Q6_K_XL.gguf", 7_490_550_624,
                       "dc3cc983735542500f40c32c1a9dacdeff198a440fac541056dca6b550389130"),
    "llm-stock-Q5_K": (_VL, "Qwen3-VL-8B-Instruct-UD-Q5_K_XL.gguf", 5_884_767_072,
                       "d138d047fe5736cdc678490d6ba16d2e42f690a6d648b5c8fbdac0e17789fac6"),
    "llm-stock-Q4_K": (_VL, "Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf", 5_148_699_488,
                       "e3d1a6e87c5cb31e054f2c3bc0dd82ffde052f613de5eef3665b7bd33c9b703e"),
    "llm-uncensored-Q8_0": (_AC, "Qwen-Image-2.1-Turbo-Uncensored-Q8_0.gguf", 8_709_520_096,
                            "f8f9f6276b726db5291f4a5343089612cb5fd08aa81e8f0791dc66d2c03d07ca"),
    "llm-uncensored-Q6_K": (_AC, "Qwen-Image-2.1-Turbo-Uncensored-AD-Q6_K.gguf", 7_490_550_720,
                            "f30f66b0fa833d3ce95c522271fc96182cc2c72a1e577d9b3444633b201b334d"),
    "llm-uncensored-Q5_K": (_AC, "Qwen-Image-2.1-Turbo-Uncensored-AD-Q5_K.gguf", 6_281_292_736,
                            "ea5473082841c6a0eda38d2d42d784ca5bc6f3a2e8cf648e305ea4da6373525b"),
    "llm-uncensored-Q4_K": (_AC, "Qwen-Image-2.1-Turbo-Uncensored-AD-Q4_K.gguf", 5_287_517_120,
                            "8123f36d2b3e562deef164dffdae33bdb1d152797ad10bb630827eaa7bce795d"),
    # Qwen3-VL's vision encoder, for the edits: the reference images into the text encoder's prompt
    "mmproj": (_VL, "mmproj-F16.gguf", 1_159_030_336,
               "d406d03ebabefdef86a2c86bf0c1b65f9e046f7a81c218f25de4931b46a07fc4"),
}
QUANTS = ("Q4_K_M", "Q6_K", "Q8_0")
TEXT_QUANTS = ("Q4_K", "Q5_K", "Q6_K", "Q8_0")      # the text encoder's, smallest first like QUANTS
DEFAULT_TEXT_ENCODER = "stock"
# setup's --images choices -> (the run config's model_name, the DiT, the text encoder)
IMAGE_MODELS = {"qwen-image-turbo": ("qwen-image-2.1-turbo", "turbo", "stock"),
                "qwen-image-turbo-uncensored": ("qwen-image-2.1-turbo-uncensored", "turbo", "uncensored"),
                "qwen-image-base": ("qwen-image-2.1", "base", "stock"),
                "qwen-image-base-uncensored": ("qwen-image-2.1-uncensored", "base", "uncensored")}
DEFAULT_MODEL = "qwen-image-turbo"
# the DiT -> its name, its steps (Turbo's are its own 8 sigmas; the base model's, the default of its pipeline,
# diffusers' QwenImage21Pipeline, sampled without guidance as Qwen meant it), its Hugging Face model
DITS = {"turbo": ("Qwen-Image-2.1-Turbo", 8, "Qwen/Qwen-Image-2.1-Turbo"),
        "base": ("Qwen-Image-2.1", 40, "Qwen/Qwen-Image-2.1")}
# every model's files in one folder (named after the first one): they share all but the DiT
MODEL_DIR = "qwen-image-2.1-turbo"
IMAGE_EXE = "strata-image.exe" if os.name == "nt" else "strata-image"
DEFAULT_QUANT = "Q8_0"
DEFAULT_TEXT_QUANT = "Q8_0"
HOST_GB = 2.0          # the image engine's own RAM besides its files (its process, the pinned relay, the prompt states)


def license_note(dit: str) -> str:
    name, _, repo = DITS[dit]
    return (f"{name} is under the Qwen Research License (not Apache 2.0): https://huggingface.co/{repo}/blob/main/"
            "LICENSE - read it before you use the images commercially")


def text_file(text_encoder: str, text_quant: str) -> str:
    """The FILES entry of the text encoder `text_encoder` at `text_quant`."""
    return f"llm-{text_encoder}-{text_quant}"


def file_names(quant: str, text_encoder: str = DEFAULT_TEXT_ENCODER, text_quant: str = DEFAULT_TEXT_QUANT,
               dit: str = "turbo") -> tuple[str, ...]:
    """The FILES entries an image install with that DiT at `quant` and that text encoder uses."""
    return (f"{dit}-{quant}", "vae", text_file(text_encoder, text_quant), "mmproj")


def ram_need_gb(quant: str, text_encoder: str = DEFAULT_TEXT_ENCODER, text_quant: str = DEFAULT_TEXT_QUANT,
                dit: str = "turbo") -> float:
    """The RAM (GB) an image job keeps busy with that DiT at `quant` and that text encoder: the four files, read
    through the OS file cache (the engine maps them), and the engine's own."""
    return sum(FILES[k][2] for k in file_names(quant, text_encoder, text_quant, dit)) / 1e9 + HOST_GB


def fit_ram(ram_left_gb: float, text_encoder: str = DEFAULT_TEXT_ENCODER, quant: str = DEFAULT_QUANT,
            text_quant: str = DEFAULT_TEXT_QUANT, dit: str = "turbo") -> tuple[str, str, bool]:
    """The DiT and text encoder quants whose files fit `ram_left_gb` (the RAM the text model leaves), from `quant`
    and `text_quant` down -> (quant, text_quant, fits): the encoder is lowered first and the DiT only when the
    smallest encoder does not fit, the encoder then the largest that fits beside that DiT.  The smallest of both, not
    fitting, when nothing does."""
    for q in reversed(QUANTS[:QUANTS.index(quant) + 1]):
        for t in reversed(TEXT_QUANTS[:TEXT_QUANTS.index(text_quant) + 1]):
            if ram_need_gb(q, text_encoder, t, dit) <= ram_left_gb:
                return q, t, True
    return QUANTS[0], TEXT_QUANTS[0], False


def file_url(name: str, endpoint: str = "https://huggingface.co") -> str:
    repo, path, _, _ = FILES[name]
    return f"{endpoint.rstrip('/')}/{repo}/resolve/{REVISIONS[repo]}/{path}"


def local_path(models_dir: Path, name: str) -> Path:
    return Path(models_dir) / MODEL_DIR / Path(FILES[name][1]).name


def image_config(quant: str, exe: Path, models_dir: Path, *, model: str = DEFAULT_MODEL, uma: bool = False,
                 policy: str = "auto", ssd_offload: bool = False, text_quant: str = DEFAULT_TEXT_QUANT) -> dict:
    """The run config's "images" section for `model` (IMAGE_MODELS; serve/images.py reads it)."""
    name, dit, text_encoder = IMAGE_MODELS[model]
    return {"model_name": name, "exe": str(exe), "quant": quant,
            "diffusion_model": str(local_path(models_dir, f"{dit}-{quant}")), "steps": DITS[dit][1],
            "schedule": dit, "vae": str(local_path(models_dir, "vae")),
            "llm": str(local_path(models_dir, text_file(text_encoder, text_quant))), "text_encoder": text_encoder,
            "text_quant": text_quant,
            "vision": str(local_path(models_dir, "mmproj")),
            "gpu_policy": policy, "uma": bool(uma),
            "ssd_offload": bool(ssd_offload)}
