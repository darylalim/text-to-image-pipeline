# Full MLX Migration: Replace PyTorch/diffusers/transformers with mflux + mlx-vlm

**Date:** 2026-04-03
**Status:** Approved

## Summary

Migrate the entire FLUX.2 Klein Streamlit app from a PyTorch-based stack (diffusers, transformers, accelerate) to an MLX-native stack (mflux, mlx-vlm). This targets Apple Silicon as the primary platform, replacing both the diffusion pipeline and the VLM prompt upsampler.

## Motivation

- MLX is optimized for Apple Silicon unified memory, providing better performance than PyTorch's MPS backend
- mflux offers built-in quantization (3-8 bit) for reduced memory footprint
- Eliminates large transitive dependency tree (torch, accelerate, sentencepiece)
- Both mflux and mlx-vlm are actively maintained and support the exact models currently in use

## Approach

Full MLX migration in a single pass. Replace both diffusers+torch (image generation) and transformers (VLM prompt upsampling) simultaneously.

**Trade-off accepted:** Drops CPU-only and Windows support. Apple Silicon is the primary target. mflux also supports Linux CUDA but that is not a priority.

## Dependencies

### Remove

- `torch`
- `diffusers` (git install)
- `transformers`
- `accelerate`
- `sentencepiece`

### Add

- `mflux` — MLX-native FLUX.2 implementation (MIT, v0.17.4+)
- `mlx-vlm` — MLX-native vision-language model inference

### Keep

- `streamlit`
- `python-dotenv`

### New pyproject.toml dependencies

```toml
dependencies = [
    "mflux",
    "mlx-vlm",
    "streamlit",
    "python-dotenv",
]
```

## Model Initialization

### Diffusion models

Replace `Flux2KleinPipeline.from_pretrained()` with `Flux2Klein(model_config=...)`.

```python
from mflux.models.flux2.variants import Flux2Klein, Flux2KleinEdit
from mflux.models.common.config import ModelConfig

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

- `_detect_device()` and all device detection/placement logic removed entirely. MLX manages unified memory automatically.
- `@lru_cache` import removed (was only used for `_detect_device`).
- `PIPES` dict renamed to `MODELS`.
- No `torch_dtype`, `use_safetensors`, `token`, or `hf_token` parameters needed.
- `@st.cache_resource` caching pattern retained for heavy model objects.

### VLM (prompt upsampling)

Replace `AutoProcessor` + `AutoModelForImageTextToText` with `mlx_vlm.load()`.

```python
from mlx_vlm import load as load_vlm, generate as vlm_generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

VLM_MODEL_ID = "mlx-community/SmolVLM-500M-Instruct-bf16"

@st.cache_resource
def _get_vlm():
    model, processor = load_vlm(VLM_MODEL_ID)
    config = load_config(VLM_MODEL_ID)
    return model, processor, config
```

- Returns `(model, processor, config)` triple instead of `(processor, model)` pair.
- `config` is needed by `apply_chat_template()`.

## Inference

Replace `pipe(**pipe_kwargs).images[0]` with `model.generate_image(...)`.

```python
def infer(
    prompt, seed=42, randomize_seed=False, width=1024, height=1024,
    guidance_scale=None, num_inference_steps=None,
    mode="Distilled (4 steps)", image_list=None, progress_callback=None,
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

Key parameter mappings:

| diffusers | mflux |
|---|---|
| `pipe(prompt=..., guidance_scale=..., ...)` | `model.generate_image(prompt=..., guidance=..., ...)` |
| `torch.Generator(device="cpu").manual_seed(seed)` | `seed=seed` (int) |
| `callback_on_step_end=fn` | `model.callbacks.register(obj)` |
| `pipe(**kwargs).images[0]` | `model.generate_image(...)` returns PIL Image |
| `image=image_list` | `image_paths=image_list` (PIL objects accepted at runtime) |
| `torch.inference_mode()` context | Not needed |

**Open question to verify during implementation:** Whether registered callbacks persist across cached model reuses and need clearing/re-registering each call.

## Prompt Upsampling

Replace manual tokenize/generate/decode with `mlx_vlm.generate()`.

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
            processor, config, messages,
            num_images=len(image_list) if image_list else 0,
        )
        result = vlm_generate(
            model, processor, formatted_prompt,
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

Key parameter mappings:

| transformers | mlx-vlm |
|---|---|
| `processor.apply_chat_template(messages, add_generation_prompt=True)` | `apply_chat_template(processor, config, messages, num_images=N)` |
| `processor(text=..., images=..., return_tensors="pt").to(device)` | Handled internally by `vlm_generate()` |
| `model.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=0.7, top_p=0.9)` | `vlm_generate(model, processor, prompt, image=..., max_tokens=150, temperature=0.7, top_p=0.9)` |
| `processor.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)[0].strip()` | `result.text.strip()` |
| Message content as list-of-dicts `[{"type": "text", "text": "..."}]` | Plain strings; image tokens inserted by `apply_chat_template` based on `num_images` |

## Tests

### Deleted

- `TestDetectDevice` — no device detection in MLX stack.

### Updated mock infrastructure

`_reload_app()` patches:
- `mflux.models.flux2.variants.Flux2Klein` (and `Flux2KleinEdit` if needed)
- `mflux.models.common.config.ModelConfig`
- `mlx_vlm.load`
- `mlx_vlm.generate`
- `mlx_vlm.prompt_utils.apply_chat_template`
- `mlx_vlm.utils.load_config`

All `torch.backends.mps/cuda`, `torch.Generator`, `torch.inference_mode` patches removed.

### Updated test helpers

- `_make_mock_pipe()` → `_make_mock_model()`: returns mock with `generate_image()` returning PIL Image and `callbacks` attribute with `register()` method.
- `_make_mock_vlm()`: returns `(model, processor, config)` triple; mock `vlm_generate` returns object with `.text` attribute.

### Test class changes

| Class | Change |
|---|---|
| `TestConstants` | Update for `MODELS` dict, `ModelConfig` refs |
| `TestDetectDevice` | Deleted |
| `TestPipelineLoading` → `TestModelLoading` | Verify `Flux2Klein(model_config=...)`, no device assertions |
| `TestInfer` | Update mocks: `generate_image()`, `callbacks.register()`, int seed, no `torch.inference_mode` |
| `TestDimensionsFromImages` | Unchanged (pure math) |
| `TestVLMInit` | Verify `mlx_vlm.load()` with correct model ID, no device assertions |
| `TestUpsamplePrompt` | Verify `apply_chat_template()` and `vlm_generate()` args, `result.text` |
| `TestResolvePrompt` | Updated mocks only |
| `TestClearEnhancement` | Unchanged |
| `TestStreamlitApp` | Remove torch-specific assertions |
| `TestExamples` | Unchanged |

## UI

No UI changes. All modifications are behind existing function interfaces.

## Files Changed

| File | Change |
|---|---|
| `pyproject.toml` | Replace dependencies |
| `streamlit_app.py` | Rewrite imports, model init, inference, VLM sections. Remove device detection. UI untouched. |
| `tests/test_streamlit_app.py` | Update all mocks. Delete `TestDetectDevice`. Update assertions. |
| `CLAUDE.md` | Update architecture, gotchas, and setup sections |
