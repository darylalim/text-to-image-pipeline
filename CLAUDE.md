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

1. **Model initialization** — Loads HF_TOKEN from `.env`, detects hardware (MPS → CUDA → CPU), selects dtype (float16 for MPS/CPU, bfloat16 for CUDA), loads the StableDiffusion3Pipeline, moves it to the detected device with `pipe.to(device)`, and enables attention slicing for memory optimization.
2. **`infer()` function** — Takes prompt, negative prompt, seed, dimensions (512–1440px), guidance scale, and inference steps. Runs inference under `torch.inference_mode()` with the generator pinned to the pipeline device. Returns a PIL Image and the seed used.
3. **Gradio UI** — Text input, run button, image output, and an accordion with advanced settings. Inference triggers on button click or prompt submission.

## Linting & Formatting

```bash
uv run ruff check .            # Lint
uv run ruff check --fix .      # Lint with auto-fix
uv run ruff format .           # Format
uv run ruff format --check .   # Check formatting only
```

## Type Checking

```bash
uv run ty check .
```

## Testing

```bash
uv run pytest                                          # Run all tests
uv run pytest tests/test_app.py                        # Run a single test file
uv run pytest tests/test_app.py::TestInfer             # Run a specific test class
```

## Gotchas

- **`from_pretrained` uses `torch_dtype`, not `dtype`.** A sub-component emits a "`torch_dtype` is deprecated, use `dtype`" warning, but the `StableDiffusion3Pipeline.from_pretrained` API itself still requires `torch_dtype`. Passing `dtype` instead causes it to be silently ignored. Do not rename this parameter.
- **Use `pipe.to(device)`, not `device_map=`.** The `device_map` parameter uses `accelerate` which probes CUDA even on MPS/CPU machines, causing spurious warnings. `pipe.to(device)` is the standard diffusers approach for single-device setups.
- **Do not remove the `warnings.catch_warnings()` block in model loading.** It suppresses two known upstream warnings: the SD3 T5 tokenizer `add_prefix_space` warning (a `sentencepiece` compatibility issue with no installable fix on macOS ARM64) and the `torch_dtype` deprecation warning.
- **Do not pin `sentencepiece==0.1.99`.** That version has no pre-built wheel for macOS ARM64 and requires `cmake` to build from source. The current unpinned version works with the warning suppression in place.

