# Performance Refactor Design

**Date:** 2026-02-21
**Scope:** Refactor app.py for performance (startup, memory, inference stability) and conciseness
**Constraints:** Single file, Apple Silicon (MPS) primary, cross-platform compatible

## 1. Lazy Model Loading

Replace top-level `pipe = StableDiffusion3Pipeline.from_pretrained(...)` with a `_get_pipe()` function that loads on first call and caches in a module-level variable. Device/dtype detection moves inside this function. `infer()` calls `_get_pipe()`. App launches instantly; model loads on first inference.

## 2. VAE Slicing + Tiling

Add `pipe.enable_vae_slicing()` and `pipe.enable_vae_tiling()` after existing `enable_attention_slicing()`. Reduces peak memory during decode, especially at large image sizes (up to 1440px).

## 3. CPU Generator for MPS

Change `torch.Generator(device=device)` to `torch.Generator(device="cpu")`. Diffusers-recommended pattern — MPS generators have known reliability issues.

## 4. Code Conciseness

- Combine device/dtype detection into a helper within `_get_pipe()`
- Remove unused `progress` parameter from `infer()` (Gradio's `track_tqdm` works without it)

## Test Impact

- Device selection tests: adapt to test `_get_pipe()` internals or the exposed device/dtype
- Pipeline init tests: adapt to lazy loading (call `_get_pipe()` to trigger load, then assert)
- Infer tests: generator device changes from `device` to `"cpu"`
- UI tests: unchanged
