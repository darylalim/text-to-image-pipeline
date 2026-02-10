import os
import random
import warnings

import gradio as gr
import torch
from diffusers import StableDiffusion3Pipeline
from dotenv import load_dotenv

load_dotenv()

hf_token = os.getenv("HF_TOKEN")

if torch.backends.mps.is_available():
    device = "mps"
    torch_dtype = torch.float16
elif torch.cuda.is_available():
    device = "cuda"
    torch_dtype = torch.bfloat16
else:
    device = "cpu"
    torch_dtype = torch.float16

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*add_prefix_space.*")
    warnings.filterwarnings("ignore", message=".*torch_dtype.*is deprecated.*")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium",
        torch_dtype=torch_dtype,
        use_safetensors=True,
        token=hf_token,
    )
pipe.to(device)
pipe.enable_attention_slicing()

MAX_SEED = 2_147_483_647
MAX_IMAGE_SIZE = 1440


def infer(
    prompt,
    negative_prompt="",
    seed=42,
    randomize_seed=False,
    width=1024,
    height=1024,
    guidance_scale=4.5,
    num_inference_steps=40,
    progress=gr.Progress(track_tqdm=True),
):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    generator = torch.Generator(device=device).manual_seed(seed)

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


with gr.Blocks() as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown(
            " # [Stable Diffusion 3.5 Medium (2.6B)](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium)"
        )
        with gr.Row():
            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=1,
                placeholder="Enter your prompt",
                container=False,
            )

            run_button = gr.Button("Run", scale=0, variant="primary")

        result = gr.Image(label="Result", show_label=False)

        with gr.Accordion("Advanced Settings", open=False):
            negative_prompt = gr.Text(
                label="Negative prompt",
                max_lines=1,
                placeholder="Enter a negative prompt",
                visible=False,
            )

            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
            )

            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

            with gr.Row():
                width = gr.Slider(
                    label="Width",
                    minimum=512,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=1024,
                )

                height = gr.Slider(
                    label="Height",
                    minimum=512,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=1024,
                )

            with gr.Row():
                guidance_scale = gr.Slider(
                    label="Guidance scale",
                    minimum=0.0,
                    maximum=7.5,
                    step=0.1,
                    value=4.5,
                )

                num_inference_steps = gr.Slider(
                    label="Number of inference steps",
                    minimum=1,
                    maximum=50,
                    step=1,
                    value=40,
                )

        gr.Examples(
            examples=[
                "A capybara wearing a suit holding a sign that reads Hello World"
            ],
            inputs=[prompt],
            outputs=[result, seed],
            fn=infer,
            cache_examples=True,
            cache_mode="lazy",
        )
    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn=infer,
        inputs=[
            prompt,
            negative_prompt,
            seed,
            randomize_seed,
            width,
            height,
            guidance_scale,
            num_inference_steps,
        ],
        outputs=[result, seed],
    )

if __name__ == "__main__":
    demo.launch(css="#col-container { margin: 0 auto; max-width: 640px; }")
