# MLX Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace PyTorch/diffusers/transformers with mflux + mlx-vlm so the entire app runs natively on MLX (Apple Silicon).

**Architecture:** Single-file Streamlit app (`streamlit_app.py`) with all logic in five sections. The migration swaps imports/model init/inference/VLM while keeping UI code and function signatures intact. Tests mock mflux and mlx-vlm instead of diffusers/transformers/torch.

**Tech Stack:** mflux (FLUX.2 Klein diffusion), mlx-vlm (SmolVLM prompt upsampling), Streamlit, PIL

**Spec:** `docs/superpowers/specs/2026-04-03-mlx-migration-design.md`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `pyproject.toml` | Modify | Replace dependencies |
| `streamlit_app.py` | Modify | Rewrite imports, model init, inference, VLM. UI untouched. |
| `tests/test_streamlit_app.py` | Modify | Rewrite mocks/helpers, delete `TestDetectDevice`, update all assertions |
| `CLAUDE.md` | Modify | Update architecture, gotchas, setup, commands |

---

### Task 1: Update dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Update pyproject.toml**

Replace the dependencies list. Remove torch, diffusers, transformers, accelerate, sentencepiece. Add mflux, mlx-vlm.

```toml
[project]
name = "flux2-klein-pipeline"
version = "0.1.0"
description = "Generate and edit images with FLUX.2 Klein (Distilled and Base), optional prompt enhancement via SmolVLM, and per-step progress tracking"
requires-python = ">=3.12"
dependencies = [
    "mflux",
    "mlx-vlm",
    "streamlit",
    "python-dotenv",
]

[dependency-groups]
dev = [
    "pytest",
    "ruff",
    "ty",
]
```

- [ ] **Step 2: Sync dependencies**

Run: `uv sync`
Expected: Resolves and installs mflux, mlx-vlm, and their transitive dependencies (including mlx). Old packages (torch, diffusers, transformers, accelerate, sentencepiece) are removed.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: replace torch/diffusers/transformers with mflux and mlx-vlm"
```

---

### Task 2: Rewrite test helpers and mock infrastructure

**Files:**
- Modify: `tests/test_streamlit_app.py` (lines 1–59)

This task rewrites the top of the test file: imports, helper functions, and `_reload_app()`. All subsequent test tasks depend on this.

- [ ] **Step 1: Rewrite imports and helpers**

Replace the entire top section of `tests/test_streamlit_app.py` (lines 1–59) with:

```python
import importlib
from unittest.mock import ANY, MagicMock, patch

from PIL import Image


def _make_mock_model():
    """Create a mock mflux model that returns a dummy image."""
    model = MagicMock()
    model.generate_image.return_value = Image.new("RGB", (64, 64))
    model.callbacks = MagicMock()
    return model


class _MockGenerationResult:
    """Mock mlx-vlm GenerationResult with a .text attribute."""

    def __init__(self, text="enhanced prompt"):
        self.text = text


def _make_mock_vlm():
    """Create a mock VLM (model, processor, config) triple."""
    mock_processor = MagicMock()
    mock_model = MagicMock()
    mock_config = MagicMock()
    return mock_model, mock_processor, mock_config


def _reload_app(mock_model, *, mock_vlm=None):
    """Reload app module with mocked heavy dependencies and passthrough cache."""
    with (
        patch("mflux.models.flux2.variants.Flux2Klein", return_value=mock_model) as mock_cls,
        patch("mflux.models.common.config.ModelConfig") as mock_model_config,
        patch("mlx_vlm.load") as mock_load,
        patch("mlx_vlm.generate") as mock_generate,
        patch("mlx_vlm.prompt_utils.apply_chat_template") as mock_chat,
        patch("mlx_vlm.utils.load_config") as mock_load_config,
        patch("streamlit.cache_resource", lambda f: f),
    ):
        if mock_vlm is not None:
            mock_vlm_model, mock_vlm_processor, mock_vlm_config = mock_vlm
            mock_load.return_value = (mock_vlm_model, mock_vlm_processor)
            mock_load_config.return_value = mock_vlm_config

        import streamlit_app

        importlib.reload(streamlit_app)
        return streamlit_app, mock_cls

```

- [ ] **Step 2: Verify file parses**

Run: `python -c "import ast; ast.parse(open('tests/test_streamlit_app.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add tests/test_streamlit_app.py
git commit -m "test: rewrite mock infrastructure for mflux and mlx-vlm"
```

---

### Task 3: Rewrite app imports and model initialization

**Files:**
- Modify: `streamlit_app.py` (lines 1–65)
- Test: `tests/test_streamlit_app.py` — `TestConstants`, delete `TestDetectDevice`, rewrite `TestPipelineLoading` → `TestModelLoading`

- [ ] **Step 1: Write TestConstants updates**

Replace `TestConstants` (lines 61–103 of test file) with:

```python
class TestConstants:
    def test_max_seed(self):
        import streamlit_app

        assert streamlit_app.MAX_SEED == 2_147_483_647

    def test_max_image_size(self):
        import streamlit_app

        assert streamlit_app.MAX_IMAGE_SIZE == 1440

    def test_vlm_model_id(self):
        import streamlit_app

        assert streamlit_app.VLM_MODEL_ID == "mlx-community/SmolVLM-500M-Instruct-bf16"

    def test_mode_defaults(self):
        import streamlit_app

        assert streamlit_app.MODE_DEFAULTS == {
            "Distilled (4 steps)": {"steps": 4, "cfg": 1.0},
            "Base (50 steps)": {"steps": 50, "cfg": 4.0},
        }

    def test_models_maps_to_getters(self):
        import streamlit_app

        assert (
            streamlit_app.MODELS["Distilled (4 steps)"]
            is streamlit_app._get_model_distilled
        )
        assert streamlit_app.MODELS["Base (50 steps)"] is streamlit_app._get_model_base

    def test_mode_defaults_keys_match_models(self):
        import streamlit_app

        assert set(streamlit_app.MODE_DEFAULTS) == set(streamlit_app.MODELS)
```

- [ ] **Step 2: Delete TestDetectDevice entirely**

Remove the entire `TestDetectDevice` class (lines 105–167 of the original test file). No replacement needed.

- [ ] **Step 3: Write TestModelLoading**

Replace `TestPipelineLoading` (lines 169–277 of original test file) with:

```python
class TestModelLoading:
    def test_distilled_model_created_with_correct_config(self):
        mock_model = _make_mock_model()
        streamlit_app, mock_cls = _reload_app(mock_model)
        with (
            patch("streamlit_app.Flux2Klein", return_value=mock_model) as mock_klein,
            patch("streamlit_app.ModelConfig") as mock_config,
        ):
            mock_config.flux2_klein_4b.return_value = "distilled_config"
            streamlit_app._get_model_distilled()
            mock_klein.assert_called_once_with(model_config="distilled_config")

    def test_base_model_created_with_correct_config(self):
        mock_model = _make_mock_model()
        streamlit_app, mock_cls = _reload_app(mock_model)
        with (
            patch("streamlit_app.Flux2Klein", return_value=mock_model) as mock_klein,
            patch("streamlit_app.ModelConfig") as mock_config,
        ):
            mock_config.flux2_klein_base_4b.return_value = "base_config"
            streamlit_app._get_model_base()
            mock_klein.assert_called_once_with(model_config="base_config")
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `uv run pytest tests/test_streamlit_app.py::TestConstants -v 2>&1 | head -30`
Expected: Failures — `streamlit_app` still has old imports/constants.

- [ ] **Step 5: Rewrite app imports and model init**

Replace lines 1–65 of `streamlit_app.py` with:

```python
import os
import random

import streamlit as st
from dotenv import load_dotenv
from mflux.models.common.config import ModelConfig
from mflux.models.flux2.variants import Flux2Klein
from mlx_vlm import generate as vlm_generate
from mlx_vlm import load as load_vlm
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
from PIL import Image

load_dotenv()

MAX_SEED = 2_147_483_647
MAX_IMAGE_SIZE = 1440

VLM_MODEL_ID = "mlx-community/SmolVLM-500M-Instruct-bf16"

MODE_DEFAULTS = {
    "Distilled (4 steps)": {"steps": 4, "cfg": 1.0},
    "Base (50 steps)": {"steps": 50, "cfg": 4.0},
}


@st.cache_resource
def _get_model_distilled():
    return Flux2Klein(model_config=ModelConfig.flux2_klein_4b())


@st.cache_resource
def _get_model_base():
    return Flux2Klein(model_config=ModelConfig.flux2_klein_base_4b())


MODELS = {
    "Distilled (4 steps)": _get_model_distilled,
    "Base (50 steps)": _get_model_base,
}
```

This removes: `torch`, `diffusers`, `transformers`, `functools.lru_cache`, `_detect_device()`, `_load_pipe()`, `hf_token`, `REPO_ID_DISTILLED`, `REPO_ID_BASE`, `PIPES`.

- [ ] **Step 6: Run updated tests**

Run: `uv run pytest tests/test_streamlit_app.py::TestConstants tests/test_streamlit_app.py::TestModelLoading -v`
Expected: All pass.

- [ ] **Step 7: Commit**

```bash
git add streamlit_app.py tests/test_streamlit_app.py
git commit -m "refactor: replace diffusers/torch model init with mflux"
```

---

### Task 4: Rewrite VLM initialization and prompt upsampling

**Files:**
- Modify: `streamlit_app.py` — `_get_vlm()`, `upsample_prompt()`
- Test: `tests/test_streamlit_app.py` — `TestVLMInit`, `TestUpsamplePrompt`

- [ ] **Step 1: Write TestVLMInit**

Replace `TestVLMInit` (lines 780–851 of original test file) with:

```python
class TestVLMInit:
    def test_vlm_loads_correct_model(self):
        mock_model = _make_mock_model()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _ = _reload_app(mock_model, mock_vlm=mock_vlm)
        with (
            patch("streamlit_app.load_vlm") as mock_load,
            patch("streamlit_app.load_config") as mock_lc,
        ):
            mock_vlm_model, mock_vlm_processor, mock_vlm_config = mock_vlm
            mock_load.return_value = (mock_vlm_model, mock_vlm_processor)
            mock_lc.return_value = mock_vlm_config
            streamlit_app._get_vlm()
            mock_load.assert_called_once_with(
                "mlx-community/SmolVLM-500M-Instruct-bf16"
            )
            mock_lc.assert_called_once_with(
                "mlx-community/SmolVLM-500M-Instruct-bf16"
            )

    def test_vlm_returns_triple(self):
        mock_model = _make_mock_model()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _ = _reload_app(mock_model, mock_vlm=mock_vlm)
        with (
            patch("streamlit_app.load_vlm") as mock_load,
            patch("streamlit_app.load_config") as mock_lc,
        ):
            mock_vlm_model, mock_vlm_processor, mock_vlm_config = mock_vlm
            mock_load.return_value = (mock_vlm_model, mock_vlm_processor)
            mock_lc.return_value = mock_vlm_config
            result = streamlit_app._get_vlm()
            assert result == (mock_vlm_model, mock_vlm_processor, mock_vlm_config)
```

- [ ] **Step 2: Write TestUpsamplePrompt**

Replace `TestUpsamplePrompt` (lines 885–1088 of original test file) with:

```python
EXPECTED_SYSTEM_PROMPT = (
    "You are an expert prompt engineer for FLUX.2 by Black Forest Labs. "
    "Rewrite user prompts to be more descriptive while strictly preserving "
    "their core subject and intent. Keep the enhanced prompt under 120 "
    "words.\n\n"
    "Guidelines:\n"
    "- Add concrete visual specifics: textures, materials, lighting, "
    "shadows, and spatial relationships.\n"
    "- Put ALL text that should appear in the image in quotation marks "
    "(signs, labels, screens, etc.) - without quotes, the model generates "
    "gibberish.\n\n"
    "Output only the revised prompt and nothing else."
)

EXPECTED_SYSTEM_PROMPT_WITH_IMAGES = (
    "You are an image-editing expert. Convert the user's editing request "
    "into one concise instruction (50-80 words, ~30 for brief requests).\n\n"
    "Rules:\n"
    "- Single instruction only, no commentary\n"
    "- Use clear, analytical language (avoid vague words like "
    '"whimsical" or "cascading")\n'
    "- Specify what changes AND what stays the same (face, lighting, "
    "composition)\n"
    "- Turn negatives into positives "
    '("don\'t change X" becomes "keep X")\n'
    '- Make abstractions concrete ("futuristic" becomes '
    '"glowing cyan neon, metallic panels")\n\n'
    "Output only the final instruction in plain text and nothing else."
)


class TestUpsamplePrompt:
    def test_chat_message_format_text_only(self):
        mock_model = _make_mock_model()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _ = _reload_app(mock_model, mock_vlm=mock_vlm)
        with (
            patch("streamlit_app.load_vlm") as mock_load,
            patch("streamlit_app.load_config") as mock_lc,
            patch("streamlit_app.apply_chat_template") as mock_chat,
            patch("streamlit_app.vlm_generate") as mock_gen,
        ):
            mock_vlm_model, mock_vlm_processor, mock_vlm_config = mock_vlm
            mock_load.return_value = (mock_vlm_model, mock_vlm_processor)
            mock_lc.return_value = mock_vlm_config
            mock_chat.return_value = "formatted prompt"
            mock_gen.return_value = _MockGenerationResult("enhanced prompt")
            streamlit_app.upsample_prompt("a cat")
            mock_chat.assert_called_once_with(
                mock_vlm_processor,
                mock_vlm_config,
                [
                    {"role": "system", "content": EXPECTED_SYSTEM_PROMPT},
                    {"role": "user", "content": "a cat"},
                ],
                num_images=0,
            )

    def test_chat_message_format_with_images(self):
        mock_model = _make_mock_model()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _ = _reload_app(mock_model, mock_vlm=mock_vlm)
        images = [Image.new("RGB", (64, 64)), Image.new("RGB", (64, 64))]
        with (
            patch("streamlit_app.load_vlm") as mock_load,
            patch("streamlit_app.load_config") as mock_lc,
            patch("streamlit_app.apply_chat_template") as mock_chat,
            patch("streamlit_app.vlm_generate") as mock_gen,
        ):
            mock_vlm_model, mock_vlm_processor, mock_vlm_config = mock_vlm
            mock_load.return_value = (mock_vlm_model, mock_vlm_processor)
            mock_lc.return_value = mock_vlm_config
            mock_chat.return_value = "formatted prompt"
            mock_gen.return_value = _MockGenerationResult("enhanced prompt")
            streamlit_app.upsample_prompt("make it blue", image_list=images)
            mock_chat.assert_called_once_with(
                mock_vlm_processor,
                mock_vlm_config,
                [
                    {"role": "system", "content": EXPECTED_SYSTEM_PROMPT_WITH_IMAGES},
                    {"role": "user", "content": "make it blue"},
                ],
                num_images=2,
            )

    def test_images_passed_to_generate(self):
        mock_model = _make_mock_model()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _ = _reload_app(mock_model, mock_vlm=mock_vlm)
        images = [Image.new("RGB", (64, 64))]
        with (
            patch("streamlit_app.load_vlm") as mock_load,
            patch("streamlit_app.load_config") as mock_lc,
            patch("streamlit_app.apply_chat_template") as mock_chat,
            patch("streamlit_app.vlm_generate") as mock_gen,
        ):
            mock_vlm_model, mock_vlm_processor, mock_vlm_config = mock_vlm
            mock_load.return_value = (mock_vlm_model, mock_vlm_processor)
            mock_lc.return_value = mock_vlm_config
            mock_chat.return_value = "formatted prompt"
            mock_gen.return_value = _MockGenerationResult("enhanced prompt")
            streamlit_app.upsample_prompt("edit", image_list=images)
            call_kwargs = mock_gen.call_args[1]
            assert call_kwargs["image"] is images

    def test_no_images_passed_for_text_only(self):
        mock_model = _make_mock_model()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _ = _reload_app(mock_model, mock_vlm=mock_vlm)
        with (
            patch("streamlit_app.load_vlm") as mock_load,
            patch("streamlit_app.load_config") as mock_lc,
            patch("streamlit_app.apply_chat_template") as mock_chat,
            patch("streamlit_app.vlm_generate") as mock_gen,
        ):
            mock_vlm_model, mock_vlm_processor, mock_vlm_config = mock_vlm
            mock_load.return_value = (mock_vlm_model, mock_vlm_processor)
            mock_lc.return_value = mock_vlm_config
            mock_chat.return_value = "formatted prompt"
            mock_gen.return_value = _MockGenerationResult("enhanced prompt")
            streamlit_app.upsample_prompt("a cat")
            call_kwargs = mock_gen.call_args[1]
            assert call_kwargs["image"] is None

    def test_generation_kwargs(self):
        mock_model = _make_mock_model()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _ = _reload_app(mock_model, mock_vlm=mock_vlm)
        with (
            patch("streamlit_app.load_vlm") as mock_load,
            patch("streamlit_app.load_config") as mock_lc,
            patch("streamlit_app.apply_chat_template") as mock_chat,
            patch("streamlit_app.vlm_generate") as mock_gen,
        ):
            mock_vlm_model, mock_vlm_processor, mock_vlm_config = mock_vlm
            mock_load.return_value = (mock_vlm_model, mock_vlm_processor)
            mock_lc.return_value = mock_vlm_config
            mock_chat.return_value = "formatted prompt"
            mock_gen.return_value = _MockGenerationResult("enhanced prompt")
            streamlit_app.upsample_prompt("a cat")
            call_kwargs = mock_gen.call_args[1]
            assert call_kwargs["max_tokens"] == 150
            assert call_kwargs["temperature"] == 0.7
            assert call_kwargs["top_p"] == 0.9

    def test_extracts_and_strips_output(self):
        mock_model = _make_mock_model()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _ = _reload_app(mock_model, mock_vlm=mock_vlm)
        with (
            patch("streamlit_app.load_vlm") as mock_load,
            patch("streamlit_app.load_config") as mock_lc,
            patch("streamlit_app.apply_chat_template") as mock_chat,
            patch("streamlit_app.vlm_generate") as mock_gen,
        ):
            mock_vlm_model, mock_vlm_processor, mock_vlm_config = mock_vlm
            mock_load.return_value = (mock_vlm_model, mock_vlm_processor)
            mock_lc.return_value = mock_vlm_config
            mock_chat.return_value = "formatted prompt"
            mock_gen.return_value = _MockGenerationResult("  A majestic feline  ")
            result = streamlit_app.upsample_prompt("a cat")
            assert result == "A majestic feline"

    def test_empty_output_returns_original(self):
        mock_model = _make_mock_model()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _ = _reload_app(mock_model, mock_vlm=mock_vlm)
        with (
            patch("streamlit_app.load_vlm") as mock_load,
            patch("streamlit_app.load_config") as mock_lc,
            patch("streamlit_app.apply_chat_template") as mock_chat,
            patch("streamlit_app.vlm_generate") as mock_gen,
        ):
            mock_vlm_model, mock_vlm_processor, mock_vlm_config = mock_vlm
            mock_load.return_value = (mock_vlm_model, mock_vlm_processor)
            mock_lc.return_value = mock_vlm_config
            mock_chat.return_value = "formatted prompt"
            mock_gen.return_value = _MockGenerationResult("")
            result = streamlit_app.upsample_prompt("a cat")
            assert result == "a cat"

    def test_whitespace_only_output_returns_original(self):
        mock_model = _make_mock_model()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _ = _reload_app(mock_model, mock_vlm=mock_vlm)
        with (
            patch("streamlit_app.load_vlm") as mock_load,
            patch("streamlit_app.load_config") as mock_lc,
            patch("streamlit_app.apply_chat_template") as mock_chat,
            patch("streamlit_app.vlm_generate") as mock_gen,
        ):
            mock_vlm_model, mock_vlm_processor, mock_vlm_config = mock_vlm
            mock_load.return_value = (mock_vlm_model, mock_vlm_processor)
            mock_lc.return_value = mock_vlm_config
            mock_chat.return_value = "formatted prompt"
            mock_gen.return_value = _MockGenerationResult("   ")
            result = streamlit_app.upsample_prompt("a cat")
            assert result == "a cat"

    def test_exception_returns_original(self):
        mock_model = _make_mock_model()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _ = _reload_app(mock_model, mock_vlm=mock_vlm)
        with (
            patch("streamlit_app.load_vlm") as mock_load,
            patch("streamlit_app.load_config") as mock_lc,
            patch("streamlit_app.apply_chat_template") as mock_chat,
            patch("streamlit_app.vlm_generate") as mock_gen,
            patch("streamlit_app.st") as mock_st,
        ):
            mock_vlm_model, mock_vlm_processor, mock_vlm_config = mock_vlm
            mock_load.return_value = (mock_vlm_model, mock_vlm_processor)
            mock_lc.return_value = mock_vlm_config
            mock_chat.return_value = "formatted prompt"
            mock_gen.side_effect = RuntimeError("OOM")
            result = streamlit_app.upsample_prompt("a cat")
            assert result == "a cat"
            mock_st.warning.assert_called_once_with(
                "Prompt enhancement failed. Using original prompt."
            )

    def test_empty_image_list_uses_text_only_path(self):
        mock_model = _make_mock_model()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _ = _reload_app(mock_model, mock_vlm=mock_vlm)
        with (
            patch("streamlit_app.load_vlm") as mock_load,
            patch("streamlit_app.load_config") as mock_lc,
            patch("streamlit_app.apply_chat_template") as mock_chat,
            patch("streamlit_app.vlm_generate") as mock_gen,
        ):
            mock_vlm_model, mock_vlm_processor, mock_vlm_config = mock_vlm
            mock_load.return_value = (mock_vlm_model, mock_vlm_processor)
            mock_lc.return_value = mock_vlm_config
            mock_chat.return_value = "formatted prompt"
            mock_gen.return_value = _MockGenerationResult("enhanced prompt")
            streamlit_app.upsample_prompt("a cat", image_list=[])
            mock_chat.assert_called_once_with(
                mock_vlm_processor,
                mock_vlm_config,
                [
                    {"role": "system", "content": EXPECTED_SYSTEM_PROMPT},
                    {"role": "user", "content": "a cat"},
                ],
                num_images=0,
            )
            call_kwargs = mock_gen.call_args[1]
            assert call_kwargs["image"] is None
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_streamlit_app.py::TestVLMInit tests/test_streamlit_app.py::TestUpsamplePrompt -v 2>&1 | head -30`
Expected: Failures — app still has old VLM code.

- [ ] **Step 4: Rewrite _get_vlm() in streamlit_app.py**

Replace the current `_get_vlm()` function (which uses `AutoProcessor` + `AutoModelForImageTextToText`) with:

```python
@st.cache_resource
def _get_vlm():
    model, processor = load_vlm(VLM_MODEL_ID)
    config = load_config(VLM_MODEL_ID)
    return model, processor, config
```

- [ ] **Step 5: Rewrite upsample_prompt() in streamlit_app.py**

Replace the current `upsample_prompt()` function with:

```python
def upsample_prompt(prompt, image_list=None):
    try:
        model, processor, config = _get_vlm()
        system_prompt = (
            UPSAMPLE_PROMPT_WITH_IMAGES if image_list else UPSAMPLE_PROMPT_TEXT_ONLY
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        formatted_prompt = apply_chat_template(
            processor,
            config,
            messages,
            num_images=len(image_list) if image_list else 0,
        )
        result = vlm_generate(
            model,
            processor,
            formatted_prompt,
            image=image_list if image_list else None,
            max_tokens=150,
            temperature=0.7,
            top_p=0.9,
        )
        enhanced = result.text.strip()
        return enhanced or prompt
    except Exception:
        st.warning("Prompt enhancement failed. Using original prompt.")
        return prompt
```

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/test_streamlit_app.py::TestVLMInit tests/test_streamlit_app.py::TestUpsamplePrompt -v`
Expected: All pass.

- [ ] **Step 7: Commit**

```bash
git add streamlit_app.py tests/test_streamlit_app.py
git commit -m "refactor: replace transformers VLM with mlx-vlm"
```

---

### Task 5: Rewrite inference function

**Files:**
- Modify: `streamlit_app.py` — `infer()`
- Test: `tests/test_streamlit_app.py` — `TestInfer`

- [ ] **Step 1: Write TestInfer**

Replace `TestInfer` (lines 279–673 of original test file) with:

```python
class TestInfer:
    def test_returns_image_and_seed(self):
        mock_model = _make_mock_model()
        streamlit_app, _ = _reload_app(mock_model)
        with patch("streamlit_app.Flux2Klein", return_value=mock_model):
            image, seed = streamlit_app.infer("a cat", seed=42)
            assert isinstance(image, Image.Image)
            assert seed == 42

    def test_forwards_args_to_model(self):
        mock_model = _make_mock_model()
        streamlit_app, _ = _reload_app(mock_model)
        with patch("streamlit_app.Flux2Klein", return_value=mock_model):
            streamlit_app.infer(
                "a cat",
                seed=123,
                width=768,
                height=512,
                guidance_scale=3.0,
                num_inference_steps=20,
            )
            mock_model.generate_image.assert_called_once_with(
                seed=123,
                prompt="a cat",
                num_inference_steps=20,
                width=768,
                height=512,
                guidance=3.0,
                image_paths=None,
            )

    def test_fixed_seed(self):
        mock_model = _make_mock_model()
        streamlit_app, _ = _reload_app(mock_model)
        with patch("streamlit_app.Flux2Klein", return_value=mock_model):
            _, seed = streamlit_app.infer("a cat", seed=99, randomize_seed=False)
            assert seed == 99

    def test_randomized_seed(self):
        mock_model = _make_mock_model()
        streamlit_app, _ = _reload_app(mock_model)
        with patch("streamlit_app.Flux2Klein", return_value=mock_model):
            _, seed = streamlit_app.infer("a cat", seed=42, randomize_seed=True)
            assert 0 <= seed <= streamlit_app.MAX_SEED

    def test_default_params(self):
        mock_model = _make_mock_model()
        streamlit_app, _ = _reload_app(mock_model)
        with patch("streamlit_app.Flux2Klein", return_value=mock_model):
            streamlit_app.infer("a cat")
            mock_model.generate_image.assert_called_once_with(
                seed=42,
                prompt="a cat",
                num_inference_steps=4,
                width=1024,
                height=1024,
                guidance=1.0,
                image_paths=None,
            )

    def test_empty_prompt(self):
        mock_model = _make_mock_model()
        streamlit_app, _ = _reload_app(mock_model)
        with patch("streamlit_app.Flux2Klein", return_value=mock_model):
            image, seed = streamlit_app.infer("", seed=42)
            assert isinstance(image, Image.Image)
            mock_model.generate_image.assert_called_once_with(
                seed=42,
                prompt="",
                num_inference_steps=4,
                width=1024,
                height=1024,
                guidance=1.0,
                image_paths=None,
            )

    def test_mode_selects_base_defaults(self):
        mock_model = _make_mock_model()
        streamlit_app, _ = _reload_app(mock_model)
        with patch("streamlit_app.Flux2Klein", return_value=mock_model):
            streamlit_app.infer("a cat", mode="Base (50 steps)")
            mock_model.generate_image.assert_called_once_with(
                seed=42,
                prompt="a cat",
                num_inference_steps=50,
                width=1024,
                height=1024,
                guidance=4.0,
                image_paths=None,
            )

    def test_explicit_params_override_mode_defaults(self):
        mock_model = _make_mock_model()
        streamlit_app, _ = _reload_app(mock_model)
        with patch("streamlit_app.Flux2Klein", return_value=mock_model):
            streamlit_app.infer(
                "a cat",
                mode="Base (50 steps)",
                guidance_scale=2.0,
                num_inference_steps=10,
            )
            mock_model.generate_image.assert_called_once_with(
                seed=42,
                prompt="a cat",
                num_inference_steps=10,
                width=1024,
                height=1024,
                guidance=2.0,
                image_paths=None,
            )

    def test_partial_override_steps_only(self):
        mock_model = _make_mock_model()
        streamlit_app, _ = _reload_app(mock_model)
        with patch("streamlit_app.Flux2Klein", return_value=mock_model):
            streamlit_app.infer(
                "a cat",
                mode="Base (50 steps)",
                num_inference_steps=10,
            )
            call_kwargs = mock_model.generate_image.call_args[1]
            assert call_kwargs["num_inference_steps"] == 10
            assert call_kwargs["guidance"] == 4.0

    def test_image_list_passed_to_model(self):
        mock_model = _make_mock_model()
        streamlit_app, _ = _reload_app(mock_model)
        images = [Image.new("RGB", (64, 64)), Image.new("RGB", (64, 64))]
        with patch("streamlit_app.Flux2Klein", return_value=mock_model):
            streamlit_app.infer("edit this", image_list=images)
            call_kwargs = mock_model.generate_image.call_args[1]
            assert call_kwargs["image_paths"] is images

    def test_no_image_paths_when_none(self):
        mock_model = _make_mock_model()
        streamlit_app, _ = _reload_app(mock_model)
        with patch("streamlit_app.Flux2Klein", return_value=mock_model):
            streamlit_app.infer("a cat")
            call_kwargs = mock_model.generate_image.call_args[1]
            assert call_kwargs["image_paths"] is None

    def test_progress_callback_registered(self):
        mock_model = _make_mock_model()
        streamlit_app, _ = _reload_app(mock_model)
        with patch("streamlit_app.Flux2Klein", return_value=mock_model):
            callback = MagicMock()
            streamlit_app.infer("a cat", progress_callback=callback)
            mock_model.callbacks.register.assert_called_once()

    def test_no_callback_when_progress_callback_none(self):
        mock_model = _make_mock_model()
        streamlit_app, _ = _reload_app(mock_model)
        with patch("streamlit_app.Flux2Klein", return_value=mock_model):
            streamlit_app.infer("a cat")
            mock_model.callbacks.register.assert_not_called()

    def test_progress_callback_invoked_with_step_and_total(self):
        mock_model = _make_mock_model()
        streamlit_app, _ = _reload_app(mock_model)
        with patch("streamlit_app.Flux2Klein", return_value=mock_model):
            callback = MagicMock()
            streamlit_app.infer(
                "a cat", num_inference_steps=4, progress_callback=callback
            )
            registered = mock_model.callbacks.register.call_args[0][0]
            mock_config = MagicMock()
            mock_config.num_inference_steps = 4
            registered.call_in_loop(0, 42, "a cat", None, mock_config, None)
            callback.assert_called_once_with(1, 4)

    def test_progress_callback_step_counts_across_steps(self):
        mock_model = _make_mock_model()
        streamlit_app, _ = _reload_app(mock_model)
        with patch("streamlit_app.Flux2Klein", return_value=mock_model):
            callback = MagicMock()
            streamlit_app.infer(
                "a cat", num_inference_steps=4, progress_callback=callback
            )
            registered = mock_model.callbacks.register.call_args[0][0]
            mock_config = MagicMock()
            mock_config.num_inference_steps = 4
            for step in range(4):
                registered.call_in_loop(step, 42, "a cat", None, mock_config, None)
            assert callback.call_count == 4
            callback.assert_any_call(1, 4)
            callback.assert_any_call(2, 4)
            callback.assert_any_call(3, 4)
            callback.assert_any_call(4, 4)

    def test_progress_callback_with_base_mode(self):
        mock_model = _make_mock_model()
        streamlit_app, _ = _reload_app(mock_model)
        with patch("streamlit_app.Flux2Klein", return_value=mock_model):
            callback = MagicMock()
            streamlit_app.infer(
                "a cat", mode="Base (50 steps)", progress_callback=callback
            )
            registered = mock_model.callbacks.register.call_args[0][0]
            mock_config = MagicMock()
            mock_config.num_inference_steps = 50
            registered.call_in_loop(0, 42, "a cat", None, mock_config, None)
            callback.assert_called_once_with(1, 50)

    def test_progress_callback_with_image_list(self):
        mock_model = _make_mock_model()
        streamlit_app, _ = _reload_app(mock_model)
        images = [Image.new("RGB", (64, 64))]
        with patch("streamlit_app.Flux2Klein", return_value=mock_model):
            callback = MagicMock()
            streamlit_app.infer(
                "edit this", image_list=images, progress_callback=callback
            )
            call_kwargs = mock_model.generate_image.call_args[1]
            assert call_kwargs["image_paths"] is images
            mock_model.callbacks.register.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_streamlit_app.py::TestInfer -v 2>&1 | head -30`
Expected: Failures — app still has old `infer()`.

- [ ] **Step 3: Rewrite infer() in streamlit_app.py**

Replace the current `infer()` function with:

```python
def infer(
    prompt,
    seed=42,
    randomize_seed=False,
    width=1024,
    height=1024,
    guidance_scale=None,
    num_inference_steps=None,
    mode="Distilled (4 steps)",
    image_list=None,
    progress_callback=None,
):
    defaults = MODE_DEFAULTS[mode]
    if guidance_scale is None:
        guidance_scale = defaults["cfg"]
    if num_inference_steps is None:
        num_inference_steps = defaults["steps"]

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    model = MODELS[mode]()

    if progress_callback is not None:

        class _ProgressReporter:
            def call_in_loop(self, t, seed, prompt, latents, config, time_steps):
                progress_callback(t + 1, config.num_inference_steps)

        model.callbacks.register(_ProgressReporter())

    image = model.generate_image(
        seed=seed,
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        width=width,
        height=height,
        guidance=guidance_scale,
        image_paths=image_list if image_list else None,
    )

    return image, seed
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_streamlit_app.py::TestInfer -v`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add streamlit_app.py tests/test_streamlit_app.py
git commit -m "refactor: replace diffusers inference with mflux generate_image"
```

---

### Task 6: Update remaining test classes

**Files:**
- Modify: `tests/test_streamlit_app.py` — `TestResolvePrompt`, `TestClearEnhancement`, `TestStreamlitApp`, `TestExamples`, `TestDimensionsFromImages`

- [ ] **Step 1: Rewrite TestResolvePrompt**

Replace `TestResolvePrompt` (lines 1090–1177 of original test file) with:

```python
class TestResolvePrompt:
    def test_returns_original_when_auto_enhance_off(self):
        mock_model = _make_mock_model()
        streamlit_app, _ = _reload_app(mock_model)
        result, was_enhanced = streamlit_app._resolve_prompt(
            "a cat", None, auto_enhance=False, already_enhanced=False
        )
        assert result == "a cat"
        assert was_enhanced is False

    def test_returns_original_when_already_enhanced(self):
        mock_model = _make_mock_model()
        streamlit_app, _ = _reload_app(mock_model)
        result, was_enhanced = streamlit_app._resolve_prompt(
            "a cat", None, auto_enhance=True, already_enhanced=True
        )
        assert result == "a cat"
        assert was_enhanced is False

    def test_enhances_when_auto_enhance_on_and_not_already_enhanced(self):
        mock_model = _make_mock_model()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _ = _reload_app(mock_model, mock_vlm=mock_vlm)
        with (
            patch("streamlit_app.load_vlm") as mock_load,
            patch("streamlit_app.load_config") as mock_lc,
            patch("streamlit_app.apply_chat_template") as mock_chat,
            patch("streamlit_app.vlm_generate") as mock_gen,
        ):
            mock_vlm_model, mock_vlm_processor, mock_vlm_config = mock_vlm
            mock_load.return_value = (mock_vlm_model, mock_vlm_processor)
            mock_lc.return_value = mock_vlm_config
            mock_chat.return_value = "formatted prompt"
            mock_gen.return_value = _MockGenerationResult("enhanced prompt")
            result, was_enhanced = streamlit_app._resolve_prompt(
                "a cat", None, auto_enhance=True, already_enhanced=False
            )
            assert result == "enhanced prompt"
            assert was_enhanced is True

    def test_enhances_with_images(self):
        mock_model = _make_mock_model()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _ = _reload_app(mock_model, mock_vlm=mock_vlm)
        images = [Image.new("RGB", (64, 64))]
        with (
            patch("streamlit_app.load_vlm") as mock_load,
            patch("streamlit_app.load_config") as mock_lc,
            patch("streamlit_app.apply_chat_template") as mock_chat,
            patch("streamlit_app.vlm_generate") as mock_gen,
        ):
            mock_vlm_model, mock_vlm_processor, mock_vlm_config = mock_vlm
            mock_load.return_value = (mock_vlm_model, mock_vlm_processor)
            mock_lc.return_value = mock_vlm_config
            mock_chat.return_value = "formatted prompt"
            mock_gen.return_value = _MockGenerationResult("enhanced prompt")
            result, was_enhanced = streamlit_app._resolve_prompt(
                "edit this", images, auto_enhance=True, already_enhanced=False
            )
            assert was_enhanced is True
            call_kwargs = mock_gen.call_args[1]
            assert call_kwargs["image"] is images

    def test_both_flags_false(self):
        mock_model = _make_mock_model()
        streamlit_app, _ = _reload_app(mock_model)
        result, was_enhanced = streamlit_app._resolve_prompt(
            "a cat", None, auto_enhance=False, already_enhanced=True
        )
        assert result == "a cat"
        assert was_enhanced is False

    def test_falls_back_on_vlm_error(self):
        mock_model = _make_mock_model()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _ = _reload_app(mock_model, mock_vlm=mock_vlm)
        with (
            patch("streamlit_app.load_vlm") as mock_load,
            patch("streamlit_app.load_config") as mock_lc,
            patch("streamlit_app.apply_chat_template") as mock_chat,
            patch("streamlit_app.vlm_generate") as mock_gen,
            patch("streamlit_app.st"),
        ):
            mock_vlm_model, mock_vlm_processor, mock_vlm_config = mock_vlm
            mock_load.return_value = (mock_vlm_model, mock_vlm_processor)
            mock_lc.return_value = mock_vlm_config
            mock_chat.return_value = "formatted prompt"
            mock_gen.side_effect = RuntimeError("OOM")
            result, was_enhanced = streamlit_app._resolve_prompt(
                "a cat", None, auto_enhance=True, already_enhanced=False
            )
            assert result == "a cat"
            assert was_enhanced is True
```

- [ ] **Step 2: Rewrite TestClearEnhancement**

Replace `TestClearEnhancement` with:

```python
class TestClearEnhancement:
    def test_clears_all_enhancement_keys(self):
        mock_model = _make_mock_model()
        streamlit_app, _ = _reload_app(mock_model)
        with patch("streamlit_app.st") as mock_st:
            mock_st.session_state = {
                "enhanced_prompt": "foo",
                "enhanced_prompt_area": "bar",
                "auto_enhanced_prompt": "baz",
                "other_key": "keep",
            }
            streamlit_app._clear_enhancement()
            assert "enhanced_prompt" not in mock_st.session_state
            assert "enhanced_prompt_area" not in mock_st.session_state
            assert "auto_enhanced_prompt" not in mock_st.session_state
            assert mock_st.session_state["other_key"] == "keep"

    def test_ignores_missing_keys(self):
        mock_model = _make_mock_model()
        streamlit_app, _ = _reload_app(mock_model)
        with patch("streamlit_app.st") as mock_st:
            mock_st.session_state = {"other_key": "keep"}
            streamlit_app._clear_enhancement()
            assert mock_st.session_state == {"other_key": "keep"}
```

- [ ] **Step 3: Rewrite TestStreamlitApp**

Replace `TestStreamlitApp` with:

```python
class TestStreamlitApp:
    def test_get_model_distilled_uses_cache_resource(self):
        """Verify _get_model_distilled is decorated with @st.cache_resource."""
        with (
            patch("mflux.models.flux2.variants.Flux2Klein"),
            patch("mflux.models.common.config.ModelConfig"),
            patch("mlx_vlm.load"),
            patch("mlx_vlm.generate"),
            patch("mlx_vlm.prompt_utils.apply_chat_template"),
            patch("mlx_vlm.utils.load_config"),
        ):
            import streamlit_app

            importlib.reload(streamlit_app)
            assert hasattr(streamlit_app._get_model_distilled, "clear")

    def test_get_model_base_uses_cache_resource(self):
        """Verify _get_model_base is decorated with @st.cache_resource."""
        with (
            patch("mflux.models.flux2.variants.Flux2Klein"),
            patch("mflux.models.common.config.ModelConfig"),
            patch("mlx_vlm.load"),
            patch("mlx_vlm.generate"),
            patch("mlx_vlm.prompt_utils.apply_chat_template"),
            patch("mlx_vlm.utils.load_config"),
        ):
            import streamlit_app

            importlib.reload(streamlit_app)
            assert hasattr(streamlit_app._get_model_base, "clear")

    def test_get_vlm_uses_cache_resource(self):
        """Verify _get_vlm is decorated with @st.cache_resource."""
        with (
            patch("mflux.models.flux2.variants.Flux2Klein"),
            patch("mflux.models.common.config.ModelConfig"),
            patch("mlx_vlm.load"),
            patch("mlx_vlm.generate"),
            patch("mlx_vlm.prompt_utils.apply_chat_template"),
            patch("mlx_vlm.utils.load_config"),
        ):
            import streamlit_app

            importlib.reload(streamlit_app)
            assert hasattr(streamlit_app._get_vlm, "clear")

    def test_ui_not_executed_on_import(self):
        mock_model = _make_mock_model()
        with (
            patch("streamlit.markdown") as mock_markdown,
            patch("streamlit.text_input") as mock_text_input,
            patch("streamlit.button") as mock_button,
        ):
            _reload_app(mock_model)
            mock_markdown.assert_not_called()
            mock_text_input.assert_not_called()
            mock_button.assert_not_called()
```

- [ ] **Step 4: Rewrite TestDimensionsFromImages**

Replace `TestDimensionsFromImages` — the logic is pure math and unchanged, but the setup uses `_make_mock_model` instead of `_make_mock_pipe`:

```python
class TestDimensionsFromImages:
    def test_square_image(self):
        mock_model = _make_mock_model()
        streamlit_app, _ = _reload_app(mock_model)
        images = [Image.new("RGB", (800, 800))]
        w, h = streamlit_app._dimensions_from_images(images)
        assert w == 1024
        assert h == 1024

    def test_landscape_image(self):
        mock_model = _make_mock_model()
        streamlit_app, _ = _reload_app(mock_model)
        images = [Image.new("RGB", (1600, 800))]
        w, h = streamlit_app._dimensions_from_images(images)
        assert w == 1024
        assert h == 512

    def test_portrait_image(self):
        mock_model = _make_mock_model()
        streamlit_app, _ = _reload_app(mock_model)
        images = [Image.new("RGB", (800, 1600))]
        w, h = streamlit_app._dimensions_from_images(images)
        assert w == 512
        assert h == 1024

    def test_rounds_to_multiple_of_32(self):
        mock_model = _make_mock_model()
        streamlit_app, _ = _reload_app(mock_model)
        images = [Image.new("RGB", (1000, 700))]
        w, h = streamlit_app._dimensions_from_images(images)
        assert w % 32 == 0
        assert h % 32 == 0

    def test_clamps_min_to_512(self):
        mock_model = _make_mock_model()
        streamlit_app, _ = _reload_app(mock_model)
        images = [Image.new("RGB", (3000, 500))]
        _, h = streamlit_app._dimensions_from_images(images)
        assert h >= 512

    def test_zero_height_returns_default(self):
        mock_model = _make_mock_model()
        streamlit_app, _ = _reload_app(mock_model)
        images = [Image.new("RGB", (100, 0))]
        w, h = streamlit_app._dimensions_from_images(images)
        assert w == 1024
        assert h == 1024

    def test_zero_width_returns_default(self):
        mock_model = _make_mock_model()
        streamlit_app, _ = _reload_app(mock_model)
        images = [Image.new("RGB", (0, 100))]
        w, h = streamlit_app._dimensions_from_images(images)
        assert w == 1024
        assert h == 1024

    def test_uses_first_image_only(self):
        mock_model = _make_mock_model()
        streamlit_app, _ = _reload_app(mock_model)
        images = [Image.new("RGB", (1600, 800)), Image.new("RGB", (800, 1600))]
        w, h = streamlit_app._dimensions_from_images(images)
        assert w == 1024
        assert h == 512

    def test_4_3_aspect_ratio(self):
        mock_model = _make_mock_model()
        streamlit_app, _ = _reload_app(mock_model)
        images = [Image.new("RGB", (1200, 900))]
        w, h = streamlit_app._dimensions_from_images(images)
        assert w == 1024
        assert h == 768

    def test_16_9_aspect_ratio(self):
        mock_model = _make_mock_model()
        streamlit_app, _ = _reload_app(mock_model)
        images = [Image.new("RGB", (1920, 1080))]
        w, h = streamlit_app._dimensions_from_images(images)
        assert w == 1024
        assert h == 576

    def test_portrait_3_4_aspect_ratio(self):
        mock_model = _make_mock_model()
        streamlit_app, _ = _reload_app(mock_model)
        images = [Image.new("RGB", (900, 1200))]
        w, h = streamlit_app._dimensions_from_images(images)
        assert w == 768
        assert h == 1024

    def test_extreme_panoramic_clamps_height(self):
        mock_model = _make_mock_model()
        streamlit_app, _ = _reload_app(mock_model)
        images = [Image.new("RGB", (5000, 500))]
        w, h = streamlit_app._dimensions_from_images(images)
        assert w == 1024
        assert h == 512

    def test_extreme_tall_clamps_width(self):
        mock_model = _make_mock_model()
        streamlit_app, _ = _reload_app(mock_model)
        images = [Image.new("RGB", (500, 5000))]
        w, h = streamlit_app._dimensions_from_images(images)
        assert w == 512
        assert h == 1024
```

- [ ] **Step 5: Rewrite TestExamples**

Replace `TestExamples` with:

```python
class TestExamples:
    def test_examples_list_structure(self):
        mock_model = _make_mock_model()
        streamlit_app, _ = _reload_app(mock_model)
        assert isinstance(streamlit_app.EXAMPLES, list)
        assert len(streamlit_app.EXAMPLES) == 5
        for example in streamlit_app.EXAMPLES:
            assert "label" in example
            assert "prompt" in example
            assert "images" in example

    def test_text_only_examples_have_no_images(self):
        mock_model = _make_mock_model()
        streamlit_app, _ = _reload_app(mock_model)
        for example in streamlit_app.EXAMPLES[:4]:
            assert example["images"] is None

    def test_image_example_has_valid_paths(self):
        mock_model = _make_mock_model()
        streamlit_app, _ = _reload_app(mock_model)
        image_example = streamlit_app.EXAMPLES[4]
        assert image_example["images"] is not None
        assert len(image_example["images"]) == 3

    def test_bundled_images_are_valid(self):
        import os

        mock_model = _make_mock_model()
        streamlit_app, _ = _reload_app(mock_model)
        image_example = streamlit_app.EXAMPLES[4]
        for path in image_example["images"]:
            assert os.path.exists(path), f"Missing: {path}"
            img = Image.open(path)
            assert img.size[0] > 0 and img.size[1] > 0
```

- [ ] **Step 6: Run full test suite**

Run: `uv run pytest tests/test_streamlit_app.py -v`
Expected: All tests pass.

- [ ] **Step 7: Commit**

```bash
git add tests/test_streamlit_app.py
git commit -m "test: update remaining test classes for mflux and mlx-vlm"
```

---

### Task 7: Update UI references and clean up app

**Files:**
- Modify: `streamlit_app.py` — UI section (references to `PIPES` → `MODELS`)

- [ ] **Step 1: Verify no stale references remain in app**

Run: `grep -n "PIPES\|_pipe\|torch\|diffusers\|transformers\|_detect_device\|lru_cache\|hf_token\|REPO_ID" streamlit_app.py`
Expected: No matches. If any remain, fix them.

- [ ] **Step 2: Run lint and format**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: No errors. If there are issues, fix with `uv run ruff check --fix . && uv run ruff format .`

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest tests/test_streamlit_app.py -v`
Expected: All tests pass.

- [ ] **Step 4: Commit (if any changes)**

```bash
git add streamlit_app.py
git commit -m "refactor: clean up stale references after MLX migration"
```

---

### Task 8: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update CLAUDE.md**

Update these sections of `CLAUDE.md`:

**Project Overview** — replace:
> Two model variants: Distilled (4 steps, fast) and Base (50 steps, higher quality). Supports multi-image input for editing workflows. Optional vision-aware prompt upsampling via SmolVLM-500M-Instruct — the VLM can see uploaded images when enhancing editing prompts.

with:
> Two model variants: Distilled (4 steps, fast) and Base (50 steps, higher quality). Runs natively on Apple Silicon via MLX (mflux for diffusion, mlx-vlm for VLM). Supports multi-image input for editing workflows. Optional vision-aware prompt upsampling via SmolVLM-500M-Instruct (mlx-community/SmolVLM-500M-Instruct-bf16) — the VLM can see uploaded images when enhancing editing prompts.

**Architecture section 1 (Model initialization)** — replace with:
> 1. **Model initialization** — `Flux2Klein(model_config=...)` from mflux creates models directly; MLX manages Apple Silicon unified memory automatically. `_get_model_distilled()` and `_get_model_base()` are `@st.cache_resource`-cached getters. `MODELS` maps mode names to getter functions. `MODE_DEFAULTS` holds per-mode defaults (Distilled: 4 steps, CFG 1.0; Base: 50 steps, CFG 4.0).

**Architecture section 3 (Prompt upsampling)** — replace with:
> 3. **Prompt upsampling** — `_get_vlm()` loads SmolVLM-500M-Instruct-bf16 via `mlx_vlm.load()`, cached with `@st.cache_resource`, returns a `(model, processor, config)` triple. Two system prompts: `UPSAMPLE_PROMPT_TEXT_ONLY` (text-to-image, capped at 120 words) and `UPSAMPLE_PROMPT_WITH_IMAGES` (image editing, concrete language, preserve unchanged elements). `upsample_prompt(prompt, image_list=None)` selects the system prompt based on whether images are provided, formats messages via `mlx_vlm.prompt_utils.apply_chat_template`, and generates via `mlx_vlm.generate()`. Loaded lazily on first use. `_resolve_prompt(prompt, image_list, auto_enhance, already_enhanced)` wraps the auto-enhance decision: enhances only when `auto_enhance` is true and `already_enhanced` is false, returns `(prompt, was_enhanced)` tuple.

**Architecture section 4 (Inference)** — replace with:
> 4. **Inference** — `infer()` takes prompt, seed, dimensions (512–1440px), mode, optional `image_list`, and optional `progress_callback`. Defaults resolve from `MODE_DEFAULTS[mode]`. Calls `model.generate_image()` from mflux. When `progress_callback` is provided, registers an `InLoopCallback` via `model.callbacks.register()` for per-step progress reporting. `_dimensions_from_images()` calculates output dimensions from the first uploaded image's aspect ratio (larger side 1024, rounds to 32, clamps 512–1440).

**Gotchas — replace the Diffusers/FLUX.2 Klein section** with:

```markdown
### mflux / FLUX.2 Klein

- **mflux uses `Flux2Klein(model_config=ModelConfig.flux2_klein_4b())`, not `from_pretrained`.** No repo ID strings, no `torch_dtype`, no `token` parameter.
- **MLX manages device placement automatically.** No `pipe.to(device)` or `enable_model_cpu_offload()`. Apple Silicon unified memory is used directly.
- **`generate_image()` returns a PIL Image directly.** Not wrapped in a `.images` list like diffusers.
- **Progress callbacks use `model.callbacks.register()`.** Register an object with a `call_in_loop(self, t, seed, prompt, latents, config, time_steps)` method. The callback must NOT return callback_kwargs (unlike diffusers).
- **`image_paths` accepts PIL Image objects at runtime.** Despite the name and type annotation (`list[Path | str]`), the internal `ImageUtil.load_image()` handles PIL objects via isinstance check.
- **The guidance parameter is `guidance`, not `guidance_scale`.** Different from diffusers naming.
- **FLUX.2 Klein does not support negative prompts.**
- **Base uses different defaults than Distilled.** Base: 50 steps, CFG 4.0. Distilled: 4 steps, CFG 1.0.
```

**Gotchas — replace the Transformers/SmolVLM section** with:

```markdown
### mlx-vlm / SmolVLM

- **Use `mlx_vlm.load()` to get `(model, processor)` and `mlx_vlm.utils.load_config()` for config.** Config is required by `apply_chat_template`.
- **`mlx_vlm.generate()` handles tokenization and decoding internally.** No manual `processor()` call, no `batch_decode`, no output slicing. Access result via `result.text`.
- **`apply_chat_template` takes `num_images` instead of embedding image tokens in messages.** Pass images as a flat list to the `image` parameter of `generate()`.
- **`temperature > 0` implies sampling.** No `do_sample` parameter. Use `temperature=0.0` for greedy decoding.
- **`max_tokens` instead of `max_new_tokens`.** Different parameter name from transformers.
```

**Gotchas — replace the General section** with:

```markdown
### General

- **Apple Silicon is the primary target.** The app uses MLX (mflux + mlx-vlm) which requires Apple Silicon or Linux CUDA. CPU-only and Windows are not supported.
- **All models share memory via MLX unified memory.** FLUX.2 Klein Distilled + Base + SmolVLM in bfloat16. All loaded lazily via `@st.cache_resource`.
- **Do not pin `sentencepiece==0.1.99`.** No longer a dependency — removed in MLX migration.
```

- [ ] **Step 2: Run lint on CLAUDE.md**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: Clean (ruff doesn't check markdown, but ensures no python issues).

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for MLX migration (mflux + mlx-vlm)"
```

---

### Task 9: Final verification

**Files:** None (verification only)

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest tests/test_streamlit_app.py -v`
Expected: All tests pass.

- [ ] **Step 2: Run lint and format checks**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: No errors.

- [ ] **Step 3: Verify no stale imports in test file**

Run: `grep -n "torch\|diffusers\|transformers\|_make_mock_pipe\|_ToableDict\|mps_available\|cuda_available" tests/test_streamlit_app.py`
Expected: No matches.

- [ ] **Step 4: Verify no stale imports in app file**

Run: `grep -n "torch\|diffusers\|transformers\|lru_cache\|hf_token\|REPO_ID\|PIPES\|_pipe\|_detect_device" streamlit_app.py`
Expected: No matches.

- [ ] **Step 5: Verify app file parses**

Run: `python -c "import ast; ast.parse(open('streamlit_app.py').read()); print('OK')"`
Expected: `OK`
