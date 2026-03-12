# Prompt Upsampling Design

## Overview

Add a prompt upsampling feature to the Streamlit app that uses a local LLM to enhance user prompts before image generation. The enhanced prompt is shown in an editable text area for the user to review and modify before generating.

## Model

- **Model**: `HuggingFaceTB/SmolLM2-1.7B-Instruct`
- **Loading**: `transformers.pipeline("text-generation", ...)` cached with `@st.cache_resource`
- **Device**: Reuses `_detect_device()` — same device and dtype as the image model
- **No new dependencies**: `transformers` is already installed

## Architecture

Two new components in `streamlit_app.py` (single-file architecture preserved):

### `_get_llm()`

Cached LLM loader using `@st.cache_resource`. Loads SmolLM2-1.7B-Instruct via `transformers.pipeline("text-generation", model="HuggingFaceTB/SmolLM2-1.7B-Instruct", torch_dtype=dtype, device=device)`. Device and dtype come from `_detect_device()`.

### `upsample_prompt(prompt: str) -> str`

Takes the user's short prompt, sends it to the LLM with a system prompt, and returns the enhanced text.

- **System prompt**: "You are a prompt engineer. Rewrite the user's text into a detailed, vivid image generation prompt. Keep it under 100 words. Output only the enhanced prompt, nothing else."
- **Generation config**: `max_new_tokens=150`, `do_sample=True`, `temperature=0.7`
- **Output parsing**: Strip the response to just the enhanced text

## UI Flow

1. **Prompt input** — existing `st.text_input` unchanged
2. **"Enhance Prompt" button** — new button below the prompt input. Calls `upsample_prompt()` and populates an editable `st.text_area` with the result
3. **Enhanced prompt area** — only appears after enhancement. User can edit freely. Stored in `st.session_state`
4. **"Run" button** — generates the image using the enhanced prompt if it exists, otherwise the original input
5. **Advanced Settings** — unchanged (seed, dimensions, guidance scale, steps)

## Testing

Following existing mock-based test patterns:

### `TestLLMInit`

- Verify `_get_llm()` loads `HuggingFaceTB/SmolLM2-1.7B-Instruct` with correct args
- Verify it uses `@st.cache_resource` (has `.clear` attribute)
- Verify correct device placement (MPS/CUDA/CPU)

### `TestUpsamplePrompt`

- Mock the LLM pipeline
- Verify `upsample_prompt()` passes the prompt with the system message
- Verify it returns the enhanced text (stripped)

### `TestStreamlitApp` updates

- Verify UI doesn't call LLM functions on import
