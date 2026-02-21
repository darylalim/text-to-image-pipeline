# Text to Image Pipeline

Generate images from text prompts with the Stable Diffusion 3.5 Medium model.

![A capybara wearing a suit holding a sign that reads Hello World](images/example.webp)

## Installation

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Install dependencies: `uv sync`
3. Accept the [Stable Diffusion 3.5 Medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) license agreement
4. Generate a Hugging Face [read-access token](https://huggingface.co/settings/tokens)
5. Create a `.env` file in the project root:
   ```
   HF_TOKEN='your_token'
   ```
6. Run the application: `uv run python app.py`

## Testing

Run the unit tests (no GPU or model download required):

```
uv run pytest
```
