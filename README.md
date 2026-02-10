# Text to Image Pipeline
Generate images from text prompts with the Stable Diffusion 3.5 Medium model.

![A capybara wearing a suit holding a sign that reads Hello World](images/example.webp)

## Installation
Run the following commands in the terminal.

- Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Install dependencies: `uv sync`
- Accept the license agreement: [Stable Diffusion 3.5 Medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium)
- Generate a Hugging Face token with read access: [settings](https://huggingface.co/settings/tokens)
- Create a `.env` file in the project root: `nano .env`
    - Enter your Hugging Face token:
      ```
      HF_TOKEN='your_token'
      ```
    - ctrl X to Exit
    - Save changes
- Run the application in a web browser: `uv run python app.py`

## Testing

Run the unit tests (no GPU or model download required):

```
uv run pytest
```

## Troubleshooting

- **`add_prefix_space` tokenizer warning** — Comes from the SD3 T5 tokenizer and `sentencepiece>=0.2.0`. Already suppressed in `app.py` during model loading. Safe to ignore if it appears elsewhere.
- **`torch_dtype` is deprecated warning** — A sub-component warns to use `dtype`, but `StableDiffusion3Pipeline.from_pretrained` still requires `torch_dtype`. Already suppressed in `app.py`. Do not rename the parameter.
- **CUDA not available warning** — Appears on machines without an NVIDIA GPU. Harmless; caused by dependency imports, not by the application.
