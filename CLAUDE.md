# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FLUX.2 Klein Pipeline is a single-file Streamlit web application that generates and edits images from text prompts using two variants of the [FLUX.2 Klein](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) model (4B parameters) from Black Forest Labs via Hugging Face Diffusers: Distilled (4 steps, fast) and Base (50 steps, higher quality). Supports multi-image input for image editing workflows. Includes optional vision-aware prompt upsampling using [SmolVLM-500M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct) to enhance prompts before generation — the VLM can see uploaded images when enhancing editing prompts. Includes pre-built example prompts with bundled images.

## Setup

```bash
uv sync
```

No license acceptance is required — FLUX.2 Klein is Apache 2.0 licensed. Optionally create a `.env` file with `HF_TOKEN=<your-token>` for authenticated Hugging Face access.

## Running

```bash
uv run streamlit run streamlit_app.py
```

## Architecture

Everything lives in `streamlit_app.py`, structured in four sections:

1. **Model initialization** — `_detect_device()` selects hardware (MPS > CUDA > CPU). All devices use bfloat16 to match the model's native dtype. `_load_pipe()` is a shared helper that loads a pipeline by repo ID; `_get_pipe_distilled()` and `_get_pipe_base()` are `@st.cache_resource`-cached getters for `FLUX.2-klein-4B` (Distilled) and `FLUX.2-klein-base-4B` (Base) respectively. The `PIPES` dict maps mode names to their getter functions. `DEFAULT_STEPS` and `DEFAULT_CFG` constants hold per-mode defaults (Distilled: 4 steps, CFG 1.0; Base: 50 steps, CFG 4.0). On CUDA, uses `enable_model_cpu_offload()`; on MPS/CPU, uses `pipe.to(device)`.
2. **Inference** — `infer()` takes prompt, seed, dimensions (512-1440px), `mode` (selects which pipeline via `PIPES`), and `image_list` (optional list of PIL Images for editing). `guidance_scale` and `num_inference_steps` use sentinel `None` defaults that resolve to `DEFAULT_CFG[mode]` and `DEFAULT_STEPS[mode]` at call time. When `image_list` is provided, passes `image=image_list` to the pipeline. FLUX.2 Klein does not support negative prompts. Runs under `torch.inference_mode()` with a CPU-pinned generator for MPS compatibility. Returns a PIL Image and the seed used. `_dimensions_from_images()` is a UI utility defined alongside `infer()` that calculates output dimensions from the first uploaded image's aspect ratio (keeps larger side at 1024, rounds to multiples of 32, clamps to 512–1440; returns 1024x1024 for degenerate images).
3. **Prompt upsampling** — `_get_vlm()` loads SmolVLM-500M-Instruct via `AutoProcessor` + `AutoModelForImageTextToText`, cached with `@st.cache_resource`. Returns a `(processor, model)` tuple. Two system prompts are defined: `UPSAMPLE_PROMPT_TEXT_ONLY` (for text-to-image generation, with guidelines for adding visual detail and quoting text in images, capped at 120 words) and `UPSAMPLE_PROMPT_WITH_IMAGES` (for image editing, with rules for concrete language, preserving unchanged elements, and turning negatives into positives). Both are adapted from the official BFL demo. `upsample_prompt()` accepts an `image_list` parameter (optional list of PIL Images); when images are provided, they are passed to the VLM so it can see them when enhancing editing prompts. Messages use the multimodal list-of-dicts format required by SmolVLM's chat template. Output is extracted by slicing generated token IDs to exclude the input prompt. The VLM is loaded lazily on first use.
4. **UI** — Streamlit interface behind `if __name__ == "__main__"` with text input, example buttons (4 text-only + 1 multi-image editing), multi-image file uploader, enhance prompt button, mode radio (Distilled vs Base), run button, image output, and an expander with advanced settings. Width/height sliders auto-update to match the uploaded image's aspect ratio (tracked via session state); slider defaults for guidance scale and steps update automatically when the mode changes. Example buttons populate the prompt (and images for the editing example) via session state; example state is cleared when the user modifies the prompt or uploads new files. Inference triggers on button click.

## Commands

```bash
uv run ruff check .              # Lint
uv run ruff check --fix .        # Lint with auto-fix
uv run ruff format .             # Format
uv run ruff format --check .     # Check formatting only
uv run ty check .                # Type check
uv run pytest                    # Run all tests
uv run pytest tests/test_streamlit_app.py  # Run a single test file
```

## Gotchas

### Diffusers / FLUX.2 Klein

- **`from_pretrained` requires `torch_dtype`, not `dtype`.** The `Flux2KleinPipeline.from_pretrained` API requires `torch_dtype`. Passing `dtype` causes it to be silently ignored. The `torch_dtype` deprecation warning originated from transformers, not diffusers — diffusers handles the translation internally as of the fix in huggingface/diffusers#12841.
- **On CUDA, use `enable_model_cpu_offload()` instead of `pipe.to(device)`.** This offloads model components to CPU when not in use, reducing VRAM requirements (~13GB). On MPS/CPU, use `pipe.to(device)` since CPU offload is CUDA-only.
- **FLUX.2 Klein does not support negative prompts.** The FLUX pipeline does not accept a `negative_prompt` parameter.
- **diffusers is installed from git.** `Flux2KleinPipeline` requires the latest diffusers from the main branch. The lockfile pins the exact commit for reproducibility. Switch to a PyPI release once `Flux2KleinPipeline` ships in a stable version.
- **Both Distilled and Base variants support `image=` for editing.** Pass a list of PIL Images to the pipeline's `image` parameter. Input images are preprocessed by the pipeline (aligned to VAE multiples, resized if total pixel area exceeds ~1 megapixel). Width/height sliders control output dimensions, not input image sizing.
- **The Base model uses different defaults than Distilled.** Base: 50 steps, guidance scale 4.0. Distilled: 4 steps, guidance scale 1.0. The `infer()` function resolves these from `DEFAULT_STEPS[mode]` and `DEFAULT_CFG[mode]` when not explicitly provided.

### Transformers / SmolVLM

- **SmolVLM uses `AutoProcessor` + `AutoModelForImageTextToText`, not `transformers.pipeline`.** Load with `AutoProcessor.from_pretrained()` and `AutoModelForImageTextToText.from_pretrained()`. The processor handles both tokenization and image preprocessing.
- **`AutoModelForImageTextToText.from_pretrained` uses `torch_dtype`, like diffusers.** This is the standard `PreTrainedModel.from_pretrained` parameter, not the `transformers.pipeline` `dtype` parameter.
- **All message `content` must use list-of-dicts format.** SmolVLM's chat template requires `[{"type": "text", "text": "..."}]`, not plain strings. This applies to system messages and user messages on all paths (text-only and multimodal).
- **`batch_decode` returns the full sequence including the input prompt.** After `model.generate()`, slice the output to exclude input tokens before decoding: `output_ids[:, inputs["input_ids"].shape[1]:]`.
- **Pass sampling parameters directly to `model.generate()`.** Use `max_new_tokens`, `do_sample`, `temperature`, `top_p` as keyword arguments. No `GenerationConfig` wrapper needed.

### General

- **The generator is pinned to CPU, not the inference device.** The model card example uses `torch.Generator(device="cuda")`, but we use `device="cpu"` for cross-device compatibility. MPS generators have known reliability issues, and CPU generators produce equivalent results across all backends.
- **Do not pin `sentencepiece==0.1.99`.** That version has no pre-built wheel for macOS ARM64. The current unpinned version works.
- **Both models and the VLM share memory.** FLUX.2 Klein Distilled (~8GB) + Base (~8GB) + SmolVLM-500M (~1.2GB) in bfloat16 require ~17.2GB peak. Both diffusion models are loaded lazily via `@st.cache_resource`. The VLM is loaded lazily on first "Enhance Prompt" use.
