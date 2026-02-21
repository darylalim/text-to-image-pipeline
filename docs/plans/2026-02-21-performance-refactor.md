# Performance Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor app.py for faster startup (lazy loading), lower memory (VAE slicing/tiling), better inference stability (CPU generator), and conciseness.

**Architecture:** Replace eager top-level model loading with a lazy `_get_pipe()` that caches on first call. Add VAE slicing/tiling for memory. Use CPU generator for MPS compatibility. Keep single-file structure.

**Tech Stack:** Python, Gradio, Diffusers, PyTorch

---

### Task 1: Refactor app.py — lazy loading, VAE optimizations, CPU generator, conciseness

**Files:**
- Modify: `app.py:1-168` (full rewrite of lines 1-67, lines 166-168 unchanged)

**Step 1: Rewrite app.py model initialization and infer()**

Replace the current top-level loading (lines 1-34) and `infer()` (lines 40-67) with lazy loading via `_get_pipe()`. The Gradio UI block (lines 70-167) stays the same.

Replace `app.py` lines 1-67 with:

```python
import os
import random
import warnings

import gradio as gr
import torch
from diffusers import StableDiffusion3Pipeline
from dotenv import load_dotenv

load_dotenv()

hf_token = os.getenv("HF_TOKEN")

MAX_SEED = 2_147_483_647
MAX_IMAGE_SIZE = 1440

_pipe = None


def _detect_device():
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    return "cpu", torch.float16


def _get_pipe():
    global _pipe
    if _pipe is not None:
        return _pipe

    device, dtype = _detect_device()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*add_prefix_space.*")
        warnings.filterwarnings("ignore", message=".*torch_dtype.*is deprecated.*")
        _pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3.5-medium",
            torch_dtype=dtype,
            use_safetensors=True,
            token=hf_token,
        )
    _pipe.to(device)
    _pipe.enable_attention_slicing()
    _pipe.enable_vae_slicing()
    _pipe.enable_vae_tiling()
    return _pipe


def infer(
    prompt,
    negative_prompt="",
    seed=42,
    randomize_seed=False,
    width=1024,
    height=1024,
    guidance_scale=4.5,
    num_inference_steps=40,
):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    pipe = _get_pipe()
    generator = torch.Generator(device="cpu").manual_seed(seed)

    with torch.inference_mode():
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            generator=generator,
        ).images[0]

    return image, seed
```

Key changes:
- `_detect_device()` — extracts device/dtype logic into a function
- `_get_pipe()` — lazy loads and caches the pipeline in `_pipe`, adds `enable_vae_slicing()` and `enable_vae_tiling()`
- `infer()` — calls `_get_pipe()`, uses `torch.Generator(device="cpu")`, removes `progress` parameter
- Module-level `device` and `torch_dtype` globals are removed (now internal to `_get_pipe`/`_detect_device`)

**Step 2: Run linting and formatting**

Run: `uv run ruff check --fix . && uv run ruff format .`
Expected: Clean output, no errors.

**Step 3: Commit**

```bash
git add app.py
git commit -m "refactor: lazy model loading, VAE slicing/tiling, CPU generator"
```

---

### Task 2: Update tests for new architecture

**Files:**
- Modify: `tests/test_app.py:1-192` (adapt to lazy loading, new module structure)

**Step 1: Rewrite test helpers and fixtures**

The tests need to adapt to:
1. No more module-level `device`/`torch_dtype` — now internal to `_detect_device()`
2. Pipeline loaded lazily via `_get_pipe()` — must call it or `infer()` to trigger
3. Generator always uses `device="cpu"`
4. New methods: `enable_vae_slicing()`, `enable_vae_tiling()`
5. `_pipe` cache must be reset between tests

Replace `tests/test_app.py` entirely with:

```python
import importlib
from unittest.mock import ANY, MagicMock, patch

import gradio as gr
import torch
from PIL import Image


def _make_mock_pipe():
    """Create a mock diffusion pipeline that returns a dummy image."""
    pipe = MagicMock()
    pipe.return_value.images = [Image.new("RGB", (64, 64))]
    return pipe


def _reload_app(mock_pipe, *, mps_available=False, cuda_available=False):
    """Reload app module with mocked heavy dependencies and cleared pipe cache."""
    with (
        patch("diffusers.StableDiffusion3Pipeline") as mock_cls,
        patch("torch.backends.mps.is_available", return_value=mps_available),
        patch("torch.cuda.is_available", return_value=cuda_available),
    ):
        mock_cls.from_pretrained.return_value = mock_pipe
        import app

        importlib.reload(app)
        # Clear the lazy-loaded cache so _get_pipe() re-runs
        app._pipe = None
        return app, mock_cls


class TestConstants:
    def test_max_seed(self):
        import app

        assert app.MAX_SEED == 2_147_483_647

    def test_max_image_size(self):
        import app

        assert app.MAX_IMAGE_SIZE == 1440


class TestDetectDevice:
    def test_mps_when_available(self):
        with (
            patch("torch.backends.mps.is_available", return_value=True),
            patch("torch.cuda.is_available", return_value=False),
        ):
            import app

            importlib.reload(app)
            device, dtype = app._detect_device()
            assert device == "mps"
            assert dtype is torch.float16

    def test_cuda_when_mps_unavailable(self):
        with (
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=True),
        ):
            import app

            importlib.reload(app)
            device, dtype = app._detect_device()
            assert device == "cuda"
            assert dtype is torch.bfloat16

    def test_cpu_fallback(self):
        with (
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            import app

            importlib.reload(app)
            device, dtype = app._detect_device()
            assert device == "cpu"
            assert dtype is torch.float16

    def test_mps_priority_over_cuda(self):
        with (
            patch("torch.backends.mps.is_available", return_value=True),
            patch("torch.cuda.is_available", return_value=True),
        ):
            import app

            importlib.reload(app)
            device, _ = app._detect_device()
            assert device == "mps"


class TestPipelineInit:
    def test_from_pretrained_args(self):
        mock_pipe = _make_mock_pipe()
        app, mock_cls = _reload_app(mock_pipe)
        with (
            patch("diffusers.StableDiffusion3Pipeline") as mock_cls2,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_cls2.from_pretrained.return_value = mock_pipe
            app._pipe = None
            app._get_pipe()
            mock_cls2.from_pretrained.assert_called_once_with(
                "stabilityai/stable-diffusion-3.5-medium",
                torch_dtype=torch.float16,
                use_safetensors=True,
                token=app.hf_token,
            )

    def test_pipeline_moved_to_device(self):
        mock_pipe = _make_mock_pipe()
        app, _ = _reload_app(mock_pipe)
        with (
            patch("diffusers.StableDiffusion3Pipeline") as mock_cls,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_cls.from_pretrained.return_value = mock_pipe
            app._pipe = None
            app._get_pipe()
            mock_pipe.to.assert_called_with("cpu")

    def test_attention_slicing_enabled(self):
        mock_pipe = _make_mock_pipe()
        app, _ = _reload_app(mock_pipe)
        with (
            patch("diffusers.StableDiffusion3Pipeline") as mock_cls,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_cls.from_pretrained.return_value = mock_pipe
            app._pipe = None
            app._get_pipe()
            mock_pipe.enable_attention_slicing.assert_called()

    def test_vae_slicing_enabled(self):
        mock_pipe = _make_mock_pipe()
        app, _ = _reload_app(mock_pipe)
        with (
            patch("diffusers.StableDiffusion3Pipeline") as mock_cls,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_cls.from_pretrained.return_value = mock_pipe
            app._pipe = None
            app._get_pipe()
            mock_pipe.enable_vae_slicing.assert_called()

    def test_vae_tiling_enabled(self):
        mock_pipe = _make_mock_pipe()
        app, _ = _reload_app(mock_pipe)
        with (
            patch("diffusers.StableDiffusion3Pipeline") as mock_cls,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_cls.from_pretrained.return_value = mock_pipe
            app._pipe = None
            app._get_pipe()
            mock_pipe.enable_vae_tiling.assert_called()

    def test_caches_pipe(self):
        mock_pipe = _make_mock_pipe()
        app, _ = _reload_app(mock_pipe)
        with (
            patch("diffusers.StableDiffusion3Pipeline") as mock_cls,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_cls.from_pretrained.return_value = mock_pipe
            app._pipe = None
            pipe1 = app._get_pipe()
            pipe2 = app._get_pipe()
            assert pipe1 is pipe2
            mock_cls.from_pretrained.assert_called_once()


class TestInfer:
    def test_returns_image_and_seed(self):
        mock_pipe = _make_mock_pipe()
        app, _ = _reload_app(mock_pipe)
        with (
            patch("diffusers.StableDiffusion3Pipeline") as mock_cls,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_cls.from_pretrained.return_value = mock_pipe
            image, seed = app.infer("a cat", seed=42)
            assert isinstance(image, Image.Image)
            assert seed == 42

    def test_forwards_args_to_pipeline(self):
        mock_pipe = _make_mock_pipe()
        app, _ = _reload_app(mock_pipe)
        with (
            patch("diffusers.StableDiffusion3Pipeline") as mock_cls,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_cls.from_pretrained.return_value = mock_pipe
            app.infer(
                "a cat",
                negative_prompt="blurry",
                seed=123,
                width=768,
                height=512,
                guidance_scale=3.0,
                num_inference_steps=20,
            )
            mock_pipe.assert_called_once_with(
                prompt="a cat",
                negative_prompt="blurry",
                guidance_scale=3.0,
                num_inference_steps=20,
                width=768,
                height=512,
                generator=ANY,
            )

    def test_fixed_seed(self):
        mock_pipe = _make_mock_pipe()
        app, _ = _reload_app(mock_pipe)
        with (
            patch("diffusers.StableDiffusion3Pipeline") as mock_cls,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_cls.from_pretrained.return_value = mock_pipe
            _, seed = app.infer("a cat", seed=99, randomize_seed=False)
            assert seed == 99

    def test_randomized_seed(self):
        mock_pipe = _make_mock_pipe()
        app, _ = _reload_app(mock_pipe)
        with (
            patch("diffusers.StableDiffusion3Pipeline") as mock_cls,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_cls.from_pretrained.return_value = mock_pipe
            _, seed = app.infer("a cat", seed=42, randomize_seed=True)
            assert 0 <= seed <= app.MAX_SEED

    def test_generator_uses_cpu(self):
        mock_pipe = _make_mock_pipe()
        app, _ = _reload_app(mock_pipe)
        with (
            patch("diffusers.StableDiffusion3Pipeline") as mock_cls,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.Generator") as mock_gen_cls,
        ):
            mock_cls.from_pretrained.return_value = mock_pipe
            mock_gen = MagicMock()
            mock_gen.manual_seed.return_value = mock_gen
            mock_gen_cls.return_value = mock_gen
            app.infer("a cat", seed=42)
            mock_gen_cls.assert_called_once_with(device="cpu")
            mock_gen.manual_seed.assert_called_once_with(42)

    def test_default_params(self):
        mock_pipe = _make_mock_pipe()
        app, _ = _reload_app(mock_pipe)
        with (
            patch("diffusers.StableDiffusion3Pipeline") as mock_cls,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_cls.from_pretrained.return_value = mock_pipe
            app.infer("a cat")
            mock_pipe.assert_called_once_with(
                prompt="a cat",
                negative_prompt="",
                guidance_scale=4.5,
                num_inference_steps=40,
                width=1024,
                height=1024,
                generator=ANY,
            )

    def test_empty_prompt(self):
        mock_pipe = _make_mock_pipe()
        app, _ = _reload_app(mock_pipe)
        with (
            patch("diffusers.StableDiffusion3Pipeline") as mock_cls,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_cls.from_pretrained.return_value = mock_pipe
            image, seed = app.infer("", seed=42)
            assert isinstance(image, Image.Image)
            mock_pipe.assert_called_once_with(
                prompt="",
                negative_prompt="",
                guidance_scale=4.5,
                num_inference_steps=40,
                width=1024,
                height=1024,
                generator=ANY,
            )

    def test_uses_inference_mode(self):
        mock_pipe = _make_mock_pipe()
        app, _ = _reload_app(mock_pipe)
        with (
            patch("diffusers.StableDiffusion3Pipeline") as mock_cls,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.inference_mode") as mock_inference_mode,
        ):
            mock_cls.from_pretrained.return_value = mock_pipe
            mock_cm = MagicMock()
            mock_inference_mode.return_value = mock_cm
            app.infer("a cat", seed=42)
            mock_inference_mode.assert_called_once()
            mock_cm.__enter__.assert_called_once()


class TestGradioUI:
    def test_demo_is_blocks_instance(self):
        mock_pipe = _make_mock_pipe()
        app, _ = _reload_app(mock_pipe)
        assert isinstance(app.demo, gr.Blocks)
```

**Step 2: Run tests**

Run: `uv run pytest tests/test_app.py -v`
Expected: All tests pass.

**Step 3: Run linting and formatting**

Run: `uv run ruff check --fix . && uv run ruff format .`
Expected: Clean output.

**Step 4: Commit**

```bash
git add tests/test_app.py
git commit -m "test: update tests for lazy loading and VAE optimizations"
```

---

### Task 3: Update CLAUDE.md architecture section

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update the Architecture section**

Update the "Model initialization" bullet in the Architecture section to reflect lazy loading, `_detect_device()`, `_get_pipe()`, VAE slicing/tiling, and CPU generator.

Replace:
> 1. **Model initialization** — Loads HF_TOKEN from `.env`, detects hardware (MPS → CUDA → CPU), selects dtype (float16 for MPS/CPU, bfloat16 for CUDA), loads the StableDiffusion3Pipeline, moves it to the detected device with `pipe.to(device)`, and enables attention slicing for memory optimization.
> 2. **`infer()` function** — Takes prompt, negative prompt, seed, dimensions (512–1440px), guidance scale, and inference steps. Runs inference under `torch.inference_mode()` with the generator pinned to the pipeline device. Returns a PIL Image and the seed used.

With:
> 1. **Model initialization** — `_detect_device()` selects hardware (MPS → CUDA → CPU) and dtype (float16 for MPS/CPU, bfloat16 for CUDA). `_get_pipe()` lazily loads the StableDiffusion3Pipeline on first inference, moves it to the detected device, and enables attention slicing, VAE slicing, and VAE tiling for memory optimization. HF_TOKEN is loaded from `.env` at import time.
> 2. **`infer()` function** — Takes prompt, negative prompt, seed, dimensions (512–1440px), guidance scale, and inference steps. Calls `_get_pipe()` to get the pipeline, runs inference under `torch.inference_mode()` with a CPU-pinned generator for MPS compatibility. Returns a PIL Image and the seed used.

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update architecture section for lazy loading"
```
