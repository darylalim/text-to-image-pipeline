import importlib
from unittest.mock import ANY, MagicMock, patch

import gradio as gr
import torch
from PIL import Image


def _make_mock_pipe():
    """Create a mock diffusion pipeline that returns a dummy image."""
    pipe = MagicMock()
    pipe.return_value.images = [Image.new("RGB", (64, 64))]
    return pipe


def _reload_app(mock_pipe, *, mps_available=False, cuda_available=False):
    """Reload app module with mocked heavy dependencies and cleared pipe cache."""
    with (
        patch("diffusers.StableDiffusion3Pipeline") as mock_cls,
        patch("torch.backends.mps.is_available", return_value=mps_available),
        patch("torch.cuda.is_available", return_value=cuda_available),
    ):
        mock_cls.from_pretrained.return_value = mock_pipe
        import app

        importlib.reload(app)
        # Clear the lazy-loaded cache so _get_pipe() re-runs
        app._pipe = None
        return app, mock_cls


class TestConstants:
    def test_max_seed(self):
        import app

        assert app.MAX_SEED == 2_147_483_647

    def test_max_image_size(self):
        import app

        assert app.MAX_IMAGE_SIZE == 1440


class TestDetectDevice:
    def test_mps_when_available(self):
        with (
            patch("torch.backends.mps.is_available", return_value=True),
            patch("torch.cuda.is_available", return_value=False),
        ):
            import app

            importlib.reload(app)
            device, dtype = app._detect_device()
            assert device == "mps"
            assert dtype is torch.float16

    def test_cuda_when_mps_unavailable(self):
        with (
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=True),
        ):
            import app

            importlib.reload(app)
            device, dtype = app._detect_device()
            assert device == "cuda"
            assert dtype is torch.bfloat16

    def test_cpu_fallback(self):
        with (
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            import app

            importlib.reload(app)
            device, dtype = app._detect_device()
            assert device == "cpu"
            assert dtype is torch.float16

    def test_mps_priority_over_cuda(self):
        with (
            patch("torch.backends.mps.is_available", return_value=True),
            patch("torch.cuda.is_available", return_value=True),
        ):
            import app

            importlib.reload(app)
            device, _ = app._detect_device()
            assert device == "mps"


class TestPipelineInit:
    def test_from_pretrained_args(self):
        mock_pipe = _make_mock_pipe()
        app, mock_cls = _reload_app(mock_pipe)
        with (
            patch("app.StableDiffusion3Pipeline") as mock_cls2,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_cls2.from_pretrained.return_value = mock_pipe
            app._pipe = None
            app._get_pipe()
            mock_cls2.from_pretrained.assert_called_once_with(
                "stabilityai/stable-diffusion-3.5-medium",
                torch_dtype=torch.float16,
                use_safetensors=True,
                token=app.hf_token,
            )

    def test_pipeline_moved_to_device(self):
        mock_pipe = _make_mock_pipe()
        app, _ = _reload_app(mock_pipe)
        with (
            patch("app.StableDiffusion3Pipeline") as mock_cls,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_cls.from_pretrained.return_value = mock_pipe
            app._pipe = None
            app._get_pipe()
            mock_pipe.to.assert_called_with("cpu")

    def test_attention_slicing_enabled(self):
        mock_pipe = _make_mock_pipe()
        app, _ = _reload_app(mock_pipe)
        with (
            patch("app.StableDiffusion3Pipeline") as mock_cls,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_cls.from_pretrained.return_value = mock_pipe
            app._pipe = None
            app._get_pipe()
            mock_pipe.enable_attention_slicing.assert_called()

    def test_vae_slicing_enabled(self):
        mock_pipe = _make_mock_pipe()
        app, _ = _reload_app(mock_pipe)
        with (
            patch("app.StableDiffusion3Pipeline") as mock_cls,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_cls.from_pretrained.return_value = mock_pipe
            app._pipe = None
            app._get_pipe()
            mock_pipe.enable_vae_slicing.assert_called()

    def test_vae_tiling_enabled(self):
        mock_pipe = _make_mock_pipe()
        app, _ = _reload_app(mock_pipe)
        with (
            patch("app.StableDiffusion3Pipeline") as mock_cls,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_cls.from_pretrained.return_value = mock_pipe
            app._pipe = None
            app._get_pipe()
            mock_pipe.enable_vae_tiling.assert_called()

    def test_caches_pipe(self):
        mock_pipe = _make_mock_pipe()
        app, _ = _reload_app(mock_pipe)
        with (
            patch("app.StableDiffusion3Pipeline") as mock_cls,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_cls.from_pretrained.return_value = mock_pipe
            app._pipe = None
            pipe1 = app._get_pipe()
            pipe2 = app._get_pipe()
            assert pipe1 is pipe2
            mock_cls.from_pretrained.assert_called_once()


class TestInfer:
    def test_returns_image_and_seed(self):
        mock_pipe = _make_mock_pipe()
        app, _ = _reload_app(mock_pipe)
        with (
            patch("app.StableDiffusion3Pipeline") as mock_cls,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_cls.from_pretrained.return_value = mock_pipe
            image, seed = app.infer("a cat", seed=42)
            assert isinstance(image, Image.Image)
            assert seed == 42

    def test_forwards_args_to_pipeline(self):
        mock_pipe = _make_mock_pipe()
        app, _ = _reload_app(mock_pipe)
        with (
            patch("app.StableDiffusion3Pipeline") as mock_cls,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_cls.from_pretrained.return_value = mock_pipe
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

    def test_fixed_seed(self):
        mock_pipe = _make_mock_pipe()
        app, _ = _reload_app(mock_pipe)
        with (
            patch("app.StableDiffusion3Pipeline") as mock_cls,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_cls.from_pretrained.return_value = mock_pipe
            _, seed = app.infer("a cat", seed=99, randomize_seed=False)
            assert seed == 99

    def test_randomized_seed(self):
        mock_pipe = _make_mock_pipe()
        app, _ = _reload_app(mock_pipe)
        with (
            patch("app.StableDiffusion3Pipeline") as mock_cls,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_cls.from_pretrained.return_value = mock_pipe
            _, seed = app.infer("a cat", seed=42, randomize_seed=True)
            assert 0 <= seed <= app.MAX_SEED

    def test_generator_uses_cpu(self):
        mock_pipe = _make_mock_pipe()
        app, _ = _reload_app(mock_pipe)
        with (
            patch("app.StableDiffusion3Pipeline") as mock_cls,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.Generator") as mock_gen_cls,
        ):
            mock_cls.from_pretrained.return_value = mock_pipe
            mock_gen = MagicMock()
            mock_gen.manual_seed.return_value = mock_gen
            mock_gen_cls.return_value = mock_gen
            app.infer("a cat", seed=42)
            mock_gen_cls.assert_called_once_with(device="cpu")
            mock_gen.manual_seed.assert_called_once_with(42)

    def test_default_params(self):
        mock_pipe = _make_mock_pipe()
        app, _ = _reload_app(mock_pipe)
        with (
            patch("app.StableDiffusion3Pipeline") as mock_cls,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_cls.from_pretrained.return_value = mock_pipe
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

    def test_empty_prompt(self):
        mock_pipe = _make_mock_pipe()
        app, _ = _reload_app(mock_pipe)
        with (
            patch("app.StableDiffusion3Pipeline") as mock_cls,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_cls.from_pretrained.return_value = mock_pipe
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

    def test_uses_inference_mode(self):
        mock_pipe = _make_mock_pipe()
        app, _ = _reload_app(mock_pipe)
        with (
            patch("app.StableDiffusion3Pipeline") as mock_cls,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.inference_mode") as mock_inference_mode,
        ):
            mock_cls.from_pretrained.return_value = mock_pipe
            mock_cm = MagicMock()
            mock_inference_mode.return_value = mock_cm
            app.infer("a cat", seed=42)
            mock_inference_mode.assert_called_once()
            mock_cm.__enter__.assert_called_once()


class TestGradioUI:
    def test_demo_is_blocks_instance(self):
        mock_pipe = _make_mock_pipe()
        app, _ = _reload_app(mock_pipe)
        assert isinstance(app.demo, gr.Blocks)
