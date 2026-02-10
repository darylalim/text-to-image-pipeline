import importlib
from unittest.mock import ANY, MagicMock, patch

import gradio as gr
import pytest
import torch
from PIL import Image


def _make_mock_pipe():
    """Create a mock diffusion pipeline that returns a dummy image."""
    pipe = MagicMock()
    pipe.return_value.images = [Image.new("RGB", (64, 64))]
    return pipe


def _reload_app(mock_pipe, *, mps_available=False, cuda_available=False):
    """Reload app module with mocked heavy dependencies."""
    with (
        patch("diffusers.StableDiffusion3Pipeline") as mock_cls,
        patch("torch.backends.mps.is_available", return_value=mps_available),
        patch("torch.cuda.is_available", return_value=cuda_available),
    ):
        mock_cls.from_pretrained.return_value = mock_pipe
        import app

        importlib.reload(app)
        return app, mock_cls


@pytest.fixture(scope="class")
def loaded_app():
    """Load app once with CPU defaults for tests that share identical mock params."""
    mock_pipe = _make_mock_pipe()
    with (
        patch("diffusers.StableDiffusion3Pipeline") as mock_cls,
        patch("torch.backends.mps.is_available", return_value=False),
        patch("torch.cuda.is_available", return_value=False),
    ):
        mock_cls.from_pretrained.return_value = mock_pipe
        import app

        importlib.reload(app)
        yield app, mock_cls, mock_pipe


@pytest.fixture
def mock_pipe():
    """Create a fresh mock pipeline for per-test isolation."""
    return _make_mock_pipe()


class TestConstants:
    def test_max_seed(self, loaded_app):
        app, _, _ = loaded_app
        assert app.MAX_SEED == 2_147_483_647

    def test_max_image_size(self, loaded_app):
        app, _, _ = loaded_app
        assert app.MAX_IMAGE_SIZE == 1440


class TestDeviceSelection:
    def test_mps_when_available(self, mock_pipe):
        app, _ = _reload_app(mock_pipe, mps_available=True)
        assert app.device == "mps"
        assert app.torch_dtype is torch.float16

    def test_cuda_when_mps_unavailable(self, mock_pipe):
        app, _ = _reload_app(mock_pipe, cuda_available=True)
        assert app.device == "cuda"
        assert app.torch_dtype is torch.bfloat16

    def test_cpu_fallback(self, mock_pipe):
        app, _ = _reload_app(mock_pipe)
        assert app.device == "cpu"
        assert app.torch_dtype is torch.float16

    def test_mps_priority_over_cuda(self, mock_pipe):
        app, _ = _reload_app(mock_pipe, mps_available=True, cuda_available=True)
        assert app.device == "mps"


class TestPipelineInit:
    def test_from_pretrained_args(self, loaded_app):
        app, mock_cls, _ = loaded_app
        mock_cls.from_pretrained.assert_called_once_with(
            "stabilityai/stable-diffusion-3.5-medium",
            torch_dtype=torch.float16,
            use_safetensors=True,
            token=app.hf_token,
        )

    def test_pipeline_moved_to_device(self, loaded_app):
        _, _, mock_pipe = loaded_app
        mock_pipe.to.assert_called_once_with("cpu")

    def test_attention_slicing_enabled(self, loaded_app):
        _, _, mock_pipe = loaded_app
        mock_pipe.enable_attention_slicing.assert_called_once()


class TestInfer:
    def test_returns_image_and_seed(self, mock_pipe):
        app, _ = _reload_app(mock_pipe)
        image, seed = app.infer("a cat", seed=42)
        assert isinstance(image, Image.Image)
        assert seed == 42

    def test_forwards_args_to_pipeline(self, mock_pipe):
        app, _ = _reload_app(mock_pipe)
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

    def test_fixed_seed(self, mock_pipe):
        app, _ = _reload_app(mock_pipe)
        _, seed = app.infer("a cat", seed=99, randomize_seed=False)
        assert seed == 99

    def test_randomized_seed(self, mock_pipe):
        app, _ = _reload_app(mock_pipe)
        _, seed = app.infer("a cat", seed=42, randomize_seed=True)
        assert 0 <= seed <= app.MAX_SEED

    def test_generator_created_on_correct_device(self, mock_pipe):
        app, _ = _reload_app(mock_pipe)
        with patch("torch.Generator") as mock_gen_cls:
            mock_gen = MagicMock()
            mock_gen.manual_seed.return_value = mock_gen
            mock_gen_cls.return_value = mock_gen
            app.infer("a cat", seed=42)
            mock_gen_cls.assert_called_once_with(device="cpu")
            mock_gen.manual_seed.assert_called_once_with(42)

    def test_default_params(self, mock_pipe):
        app, _ = _reload_app(mock_pipe)
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

    def test_empty_prompt(self, mock_pipe):
        app, _ = _reload_app(mock_pipe)
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

    def test_uses_inference_mode(self, mock_pipe):
        app, _ = _reload_app(mock_pipe)
        with patch("torch.inference_mode") as mock_inference_mode:
            mock_cm = MagicMock()
            mock_inference_mode.return_value = mock_cm
            app.infer("a cat", seed=42)
            mock_inference_mode.assert_called_once()
            mock_cm.__enter__.assert_called_once()


class TestGradioUI:
    def test_demo_is_blocks_instance(self, loaded_app):
        app, _, _ = loaded_app
        assert isinstance(app.demo, gr.Blocks)
