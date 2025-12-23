# Text to Image Pipeline

A tool for generating images from text prompts using Stable Diffusion 3.5 Medium. Built with a simple web interface powered by Gradio.

## Features

- Generate high-quality images from text descriptions
- Adjustable image dimensions (512px - 1440px)
- Customizable generation parameters (guidance scale, inference steps)
- Seed control for reproducible results
- Try the model on [Stability AI API](https://platform.stability.ai/docs/api-reference#tag/Generate/paths/~1v2beta~1stable-image~1generate~1sd3/post)

## Setup

1. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Hugging Face access**
   - Accept the license agreement for [Stable Diffusion 3.5 Medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium)
   - Generate a Hugging Face token with read access from your [settings page](https://huggingface.co/settings/tokens)
   - Create a `.env` file in the project root:
     ```
     HF_TOKEN=your_token_here
     ```

4. **Run the application**
   ```bash
   python app.py
   ```

The application will launch in your default web browser.

## Model Information

This project uses [Stable Diffusion 3.5 Medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium), a 2.6B parameter model from Stability AI. Learn more about the [Stable Diffusion 3.5 series](https://stability.ai/news/introducing-stable-diffusion-3-5).
