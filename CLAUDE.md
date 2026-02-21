# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Single-file Gradio web application that generates images from text prompts using the Stable Diffusion 3.5 Medium model (2.6B parameters) via Hugging Face Diffusers.

## Setup

```bash
uv sync
```

**Prerequisites:** Accept the [Stable Diffusion 3.5 Medium license](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) on Hugging Face, then create a `.env` file with `HF_TOKEN=<your-read-access-token>`.

## Running

```bash
uv run python app.py
```

## Architecture

Everything lives in `app.py`, structured in three sections:

1. **Model initialization** — `_detect_device()` selects hardware (MPS > CUDA > CPU) and dtype (float16 for MPS/CPU, bfloat16 for CUDA). `_get_pipe()` lazily loads the pipeline on first inference, moves it to the detected device, and enables attention slicing. HF_TOKEN is loaded from `.env` at import time.
2. **Inference** — `infer()` takes prompt, negative prompt, seed, dimensions (512-1440px), guidance scale, and inference steps. Calls `_get_pipe()`, runs inference under `torch.inference_mode()` with a CPU-pinned generator for MPS compatibility. Returns a PIL Image and the seed used.
3. **UI** — Text input, run button, image output, and an accordion with advanced settings. Inference triggers on button click or prompt submission.

## Commands

```bash
uv run ruff check .              # Lint
uv run ruff check --fix .        # Lint with auto-fix
uv run ruff format .             # Format
uv run ruff format --check .     # Check formatting only
uv run ty check .                # Type check
uv run pytest                    # Run all tests
uv run pytest tests/test_app.py  # Run a single test file
```

## Gotchas

- **`from_pretrained` requires `torch_dtype`, not `dtype`.** A sub-component warns that `torch_dtype` is deprecated, but the `StableDiffusion3Pipeline.from_pretrained` API still requires it. Passing `dtype` causes it to be silently ignored.
- **Use `pipe.to(device)`, not `device_map=`.** `device_map` probes CUDA even on MPS/CPU machines, causing spurious warnings. `pipe.to(device)` is the standard diffusers approach for single-device setups.
- **Do not remove the `warnings.catch_warnings()` blocks.** The block around the diffusers import suppresses a Kandinsky5 `torch.autocast(device_type="cuda")` warning that fires at import time on non-CUDA machines. The block in `_get_pipe()` suppresses the T5 tokenizer `add_prefix_space` warning and the `torch_dtype` deprecation warning.
- **SD3 does not support `enable_vae_slicing()` or `enable_vae_tiling()`.** These methods exist on other pipelines (e.g., SDXL) but not `StableDiffusion3Pipeline`. Only `enable_attention_slicing()` is available.
- **Do not pin `sentencepiece==0.1.99`.** That version has no pre-built wheel for macOS ARM64 and requires `cmake` to build from source. The current unpinned version works with the warning suppression in place.
