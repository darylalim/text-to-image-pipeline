# Text to Image Pipeline
Generate images from text prompts with the Stable Diffusion 3.5 Medium model.
   
## Installation
Run the following commands in the terminal.

- Set up a Python virtual environment: `python3.12 -m venv gradio_env`
- Activate the virtual environment: `source gradio_env/bin/activate`
- Install the required Python packages: `pip install -r requirements.txt`
- Accept the license agreement: [Stable Diffusion 3.5 Medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium)
- Generate a Hugging Face token with read access: [settings](https://huggingface.co/settings/tokens)
- Create a `.env` file in the project root: `nano .env`
    - Enter your Hugging Face token:
      ```
      HF_TOKEN='your_token'
      ```
    - ctrl X to Exit
    - Save changes
- Run the application in a web browser: `gradio app.py`
