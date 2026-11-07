import asyncio
import base64
import io
import os
import math
from inspect import cleandoc

import numpy as np
import requests
import torch
from PIL import Image
try:
    from comfy.comfy_types.node_typing import IO, ComfyNodeABC, InputTypeDict
except Exception:
    from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict
from comfy.utils import common_upscale
from .apis import (
    OpenAIImageEditRequest,
    OpenAIImageGenerationRequest,
    OpenAIImageGenerationResponse,
)
from .apis.client import ApiEndpoint, HttpMethod, SynchronousOperation

import json

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")


# Preserve node IDs and input order for existing workflows.
# Model IDs and capabilities checked against OpenAI docs on 2026-11-07.
DEFAULT_MODEL = "gpt-image-2.6-flare"
GPT_IMAGE_MODELS = [
    "gpt-image-2.6-flare",
    "gpt-image-2.6-sunburst",
    "gpt-image-2.5-flare",
    "gpt-image-2.5-sunburst",
    "gpt-image-2.5-flare-2026-09-08",
    "gpt-image-2.5-sunburst-2026-09-08",
    "gpt-image-2",
    "gpt-image-1.5",
    "gpt-image-1",
    "gpt-image-1-mini",
]
GPT_IMAGE_QUALITIES = ["auto", "low", "medium", "high", "xhigh", "max"]
BACKGROUND_TOOLTIP = (
    "GPT Image 2.6 / 2.5 support transparent PNG/WebP output. "
    "GPT Image 2 transparency is available in preview."
)
QUALITY_TOOLTIP = (
    "Image quality affects cost and generation time. "
    "xhigh and max require GPT Image 2.5 or 2.6 Flare/Sunburst."
)


def _resolve_model(model, quality, background):
    """Validate capabilities without silently substituting another model."""
    model = (model or "").strip() or DEFAULT_MODEL
    is_image25 = any(
        model == family or model.startswith(family + "-")
        for family in ("gpt-image-2.5-flare", "gpt-image-2.5-sunburst", "gpt-image-2.6-flare", "gpt-image-2.6-sunburst")
    )
    if quality in ("xhigh", "max") and not is_image25:
        raise ValueError(
            f"quality={quality!r} requires GPT Image 2.5 or 2.6 Flare/Sunburst; "
            f"selected model is {model!r}. Use auto, low, medium or high."
        )
    return model


def _normalize_api_base(api_base):
    """Preserve /v1 when the existing client joins relative endpoint paths."""
    value = (api_base or "").strip()
    if not value:
        raise ValueError("Set api_base, for example https://api.openai.com/v1")
    return value.rstrip("/") + "/"


def read_user_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def downscale_input(image):
    samples = image.movedim(-1, 1)
    # downscaling input images to roughly the same size as the outputs
    total = int(1536 * 1024)
    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
    if scale_by >= 1:
        return image
    width = round(samples.shape[3] * scale_by)
    height = round(samples.shape[2] * scale_by)

    s = common_upscale(samples, width, height, "lanczos", "disabled")
    s = s.movedim(1, -1)
    return s


def validate_and_cast_response(response):
    # validate raw JSON response
    data = response.data
    if not data or len(data) == 0:
        raise Exception("No images returned from API endpoint")

    # Initialize list to store image tensors
    image_tensors = []

    # Process each image in the data array
    for image_data in data:
        image_url = image_data.url
        b64_data = image_data.b64_json

        if not image_url and not b64_data:
            raise Exception("No image was generated in the response")

        if b64_data:
            img_data = base64.b64decode(b64_data)
            img = Image.open(io.BytesIO(img_data))

        elif image_url:
            img_response = requests.get(image_url, timeout=(15, 120))
            if img_response.status_code != 200:
                raise Exception("Failed to download the image")
            img = Image.open(io.BytesIO(img_response.content))

        if getattr(img, "n_frames", 1) > 1:
            raise ValueError("Animated response requires the Files x2 node; IMAGE would discard frames")
        img = img.convert("RGBA")

        # Convert to numpy array, normalize to float32 between 0 and 1
        img_array = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array)

        # Add to list of tensors
        image_tensors.append(img_tensor)

    return torch.stack(image_tensors, dim=0)


def _family(model, version):
    return any(model == f"gpt-image-{version}-{kind}" or model.startswith(f"gpt-image-{version}-{kind}-") for kind in ("flare", "sunburst"))


def _advanced_inputs(suffix=""):
    return {
        "complexity" + suffix: (IO.COMBO, {"options": ["auto", "low", "medium", "high"], "default": "auto", "tooltip": "GPT Image 2.6 planning effort, separate from quality. auto omits the field (API default high)."}),
        "markup" + suffix: (IO.IMAGE, {"tooltip": "Transparent RGBA annotation overlay for the first reference, same dimensions. Cannot combine with mask. Load Image normally drops alpha: use Join Image with Alpha or an RGBA loader."}),
        "output_format" + suffix: (IO.COMBO, {"options": ["png", "webp", "jpeg"], "default": "png"}),
        "output_compression" + suffix: (IO.INT, {"default": 100, "min": 0, "max": 100, "tooltip": "Sent only for JPEG/WebP; ignored for PNG."}),
    }


def _validate_features(model, images, mask, markup, complexity, output_format, compression, n, frame_count, duration):
    if complexity not in ("auto", "low", "medium", "high"):
        raise ValueError("complexity must be auto, low, medium or high")
    extras = {}
    if complexity != "auto":
        if not _family(model, "2.6"):
            raise ValueError("Explicit complexity requires GPT Image 2.6")
        extras["complexity"] = complexity
    if markup is not None and not (_family(model, "2.5") or _family(model, "2.6")):
        raise ValueError("markup requires GPT Image 2.5 or 2.6")
    if output_format not in ("png", "webp", "jpeg", "svg", "gif"):
        raise ValueError("Unsupported output_format")
    if output_format in ("svg", "gif") and not _family(model, "2.6"):
        raise ValueError("SVG/GIF output requires GPT Image 2.6 and project preview access")
    if output_format == "svg":
        if any(im is not None for im in images) or mask is not None or markup is not None:
            raise ValueError("SVG supports text-to-image generations only, not edits")
        if n != 1:
            raise ValueError("SVG output requires n=1")
    if output_format == "gif":
        if mask is not None:
            raise ValueError("GIF output does not support masks")
        if not 2 <= frame_count <= 16 or not 20 <= duration <= 2000:
            raise ValueError("GIF requires 2-16 frames and frame_duration_ms between 20 and 2000")
        extras.update(frame_count=frame_count, frame_duration_ms=duration)
    if output_format in ("jpeg", "webp"):
        if not 0 <= compression <= 100:
            raise ValueError("output_compression must be between 0 and 100")
        extras["output_compression"] = compression
    return extras


class GPTImage1Generate(ComfyNodeABC):
    """
    comfyui-gpt-image ports the official ComfyUI GPT-API node,
    adding support for customizable api_base, auth_token, and model settings.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:

        return {
            "required": {
                "prompt": (
                    IO.STRING,
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Text prompt for GPT Image",
                    },
                ),
            },
            "optional": {
                "api_base": (
                    IO.STRING,
                    {
                        "default": "",
                        "display": "string",
                        "tooltip": "API Base URL",
                    },
                ),
                "auth_token": (
                    IO.STRING,
                    {
                        "default": "",
                        "display": "string",
                        "tooltip": "API Auth Token",
                    },
                ),
                "model": (
                    IO.COMBO,
                    {
                        "options": GPT_IMAGE_MODELS,
                        "default": DEFAULT_MODEL,
                        "tooltip": "GPT Image model",
                    },
                ),
                "seed": (
                    IO.INT,
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2**31 - 1,
                        "step": 1,
                        "display": "number",
                        "tooltip": "Workflow compatibility only; the image API does not accept a seed.",
                    },
                ),
                "quality": (
                    IO.COMBO,
                    {
                        "options": GPT_IMAGE_QUALITIES,
                        "default": "auto",
                        "tooltip": QUALITY_TOOLTIP,
                    },
                ),
                "background": (
                    IO.COMBO,
                    {
                        "options": ["auto", "opaque", "transparent"],
                        "default": "opaque",
                        "tooltip": BACKGROUND_TOOLTIP,
                    },
                ),
                "size": (
                    IO.COMBO,
                    {
                        "options": [
                            "auto",
                            "1024x1024",
                            "1536x1024",
                            "1024x1536",
                            "2048x2048",
                            "2048x1152",
                            "3840x2160",
                            "2160x3840",
                        ],
                        "default": "auto",
                        "tooltip": "Image size. GPT Image 2 / 2.5: edges multiple of 16, <=3840px, ratio <=3:1; >2560x1440 is experimental.",
                    },
                ),
                "n": (
                    IO.INT,
                    {
                        "default": 1,
                        "min": 1,
                        "max": 8,
                        "step": 1,
                        "display": "number",
                        "tooltip": "How many images to generate",
                    },
                ),
                "image": (
                    IO.IMAGE,
                    {
                        "default": None,
                        "tooltip": "Optional reference image for image editing.",
                    },
                ),
                "mask": (
                    IO.MASK,
                    {
                        "default": None,
                        "tooltip": "Optional mask for inpainting (white areas will be replaced)",
                    },
                ),
                "moderation": (
                    IO.COMBO,
                    {
                        "options": ["low", "auto"],
                        "default": "low",
                        "tooltip": "Moderation level",
                    },
                ),
                **_advanced_inputs(),
            },
            # "hidden": {
            #     "auth_token": "AUTH_TOKEN_COMFY_ORG"
            # }
        }

    RETURN_TYPES = (IO.IMAGE,)
    FUNCTION = "api_call"
    CATEGORY = "OpenAI GPT Image (Parallel)"
    DESCRIPTION = cleandoc(__doc__ or "")


    async def api_call(
        self, prompt, seed=0, quality="auto", background="opaque", image=None,
        mask=None, n=1, size="1024x1024", moderation="low", api_base=None,
        auth_token=None, model=None, complexity="auto", markup=None,
        output_format="png", output_compression=100,
    ):
        if output_format not in ("png", "webp", "jpeg"):
            raise ValueError("Use the GPT Image 2.6 Files x2 node for SVG or GIF output")
        config = read_user_config()
        if mask is not None and image is None:
            raise ValueError("Cannot use a mask without an input image")
        if mask is not None and image.shape[0] != 1:
            raise ValueError("Cannot use a mask with multiple images in the single node")
        op = _build_operation(prompt, [image], mask, quality, background, n, size,
            moderation, api_base or config.get("api_base", ""),
            auth_token or config.get("auth_token", "") or os.environ.get("OPENAI_API_KEY", ""),
            model, complexity, markup, output_format, output_compression)
        return (await asyncio.to_thread(lambda: validate_and_cast_response(op.execute())),)


def _prepare_image_files(images, mask, markup=None):
    """images: list of IMAGE tensors (each may itself be a batch) and/or None.
    Returns (path, request_class, files). Uses images/edits when at least one
    image is connected; otherwise images/generations (text-to-image)."""
    path = "images/generations"
    request_class = OpenAIImageGenerationRequest
    files = []

    # Flatten every connected slot into individual frames.
    frames = []
    for img in images:
        if img is None:
            continue
        for i in range(img.shape[0]):
            frames.append(img[i : i + 1])

    if frames:
        path = "images/edits"
        request_class = OpenAIImageEditRequest
        for idx, single_image in enumerate(frames):
            scaled_image = downscale_input(single_image).squeeze()
            image_np = (scaled_image.detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(image_np)
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            buf.seek(0)
            buf.name = f"image_{idx}.png"
            # Single image -> "image"; multiple -> "image[]"
            if len(frames) == 1:
                files.append(("image", buf))
            else:
                files.append(("image[]", buf))

    if len(frames) > 16:
        raise ValueError("The Images API accepts at most 16 reference images")
    if mask is not None and markup is not None:
        raise ValueError("markup and mask cannot be combined")
    if markup is not None:
        if not frames:
            raise ValueError("markup requires an input image")
        if markup.ndim != 4 or markup.shape[0] != 1 or markup.shape[-1] != 4:
            raise ValueError("markup must be one RGBA IMAGE with transparent pixels")
        if tuple(markup.shape[1:3]) != tuple(frames[0].shape[1:3]):
            raise ValueError("markup dimensions must match the first reference image")
        if not torch.any(markup[..., 3] < 1):
            raise ValueError("markup must contain transparent pixels")
        pixels = (downscale_input(markup)[0].detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(pixels).save(buf, format="PNG")
        buf.seek(0)
        buf.name = "markup.png"
        files.append(("markup", buf))
    if mask is not None:
        if not frames:
            raise Exception("Cannot use a mask without at least one input image")
        if mask.ndim != 3 or mask.shape[0] != 1 or tuple(mask.shape[1:]) != tuple(frames[0].shape[1:3]):
            raise ValueError("Mask must be one mask matching the first reference image")
        batch, height, width = mask.shape
        rgba_mask = torch.zeros(height, width, 4, device="cpu")
        rgba_mask[:, :, 3] = 1 - mask.squeeze().cpu()
        scaled_mask = downscale_input(rgba_mask.unsqueeze(0)).squeeze()
        mask_np = (scaled_mask.numpy() * 255).astype(np.uint8)
        mask_img = Image.fromarray(mask_np)
        mbuf = io.BytesIO()
        mask_img.save(mbuf, format="PNG")
        mbuf.seek(0)
        mbuf.name = "mask.png"
        files.append(("mask", mbuf))

    return path, request_class, files


def _build_operation(prompt, images, mask, quality, background, n, size, moderation, api_base, auth_token, model,
                     complexity="auto", markup=None, output_format="png", output_compression=100,
                     frame_count=9, frame_duration_ms=200):
    model = _resolve_model(model, quality, background)
    if background == "transparent" and output_format not in ("png", "webp", "svg"):
        raise ValueError("Transparent output requires PNG/WebP (or SVG), not JPEG/GIF")
    extras = _validate_features(model, images, mask, markup, complexity, output_format,
                                output_compression, n, frame_count, frame_duration_ms)
    path, request_class, files = _prepare_image_files(images, mask, markup)
    return SynchronousOperation(
        endpoint=ApiEndpoint(
            path=path,
            method=HttpMethod.POST,
            request_model=request_class,
            response_model=OpenAIImageGenerationResponse,
        ),
        request=request_class(
            model=model,
            prompt=prompt,
            quality=quality,
            background=background,
            output_format=output_format,
            **extras,
            n=n,
            size=size,
            moderation=moderation,
        ),
        files=files if files else None,
        api_base=_normalize_api_base(api_base),
        auth_token=auth_token,
    )


def _img_input(letter, idx):
    return (IO.IMAGE, {"tooltip": f"Optional reference image #{idx} for branch {letter} (enables editing)."})


# Official GPT Image 2 / 2.5 popular sizes (OpenAI docs). "custom" lets you type any
# resolution that meets the constraints (edges multiple of 16, <=3840px,
# ratio <=3:1, total pixels 655360-8294400; >2560x1440 is experimental).
_GPT_IMAGE2_SIZES = [
    "auto",
    "1024x1024",
    "1536x1024",
    "1024x1536",
    "2048x2048",
    "2048x1152",
    "3840x2160",
    "2160x3840",
    "custom",
]


def _resolve_size(size, size_custom):
    """If size == 'custom', use the free-text value (fallback to 'auto')."""
    if size == "custom":
        s = (size_custom or "").strip()
        return s if s else "auto"
    return size


class GPTImage2GenerateParallelX2(ComfyNodeABC):
    """
    Two fully independent GPT Image generations at once. Each branch has its
    own prompt, up to 5 reference images, mask, size (preset OR custom
    resolution), quality, background, image count (n), model and moderation.
    Only the API connection (api_base + auth_token) is shared. Both calls fire
    concurrently via asyncio.gather. A branch with any image connected uses
    images/edits; otherwise images/generations. Returns image_a, image_b.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        sizes = _GPT_IMAGE2_SIZES
        qualities = GPT_IMAGE_QUALITIES
        backgrounds = ["auto", "opaque", "transparent"]
        moderations = ["low", "auto"]
        models = GPT_IMAGE_MODELS
        custom_size_tip = "Used only when size = custom. Format WIDTHxHEIGHT, e.g. 1792x1024. Edges multiple of 16, <=3840px, ratio <=3:1; >2560x1440 experimental."
        bg_tip = BACKGROUND_TOOLTIP
        size_tip = "Preset size/proportion, or 'custom' to type your own below."
        return {
            "required": {
                "prompt_a": (IO.STRING, {"multiline": True, "default": "", "tooltip": "Prompt for branch A"}),
                "prompt_b": (IO.STRING, {"multiline": True, "default": "", "tooltip": "Prompt for branch B"}),
            },
            "optional": {
                # ---- shared connection (your OpenAI account) ----
                "api_base": (IO.STRING, {"default": "", "tooltip": "Shared API Base URL (e.g. https://api.openai.com/v1)"}),
                "auth_token": (IO.STRING, {"default": "", "tooltip": "Shared OpenAI API key (sk-...)"}),
                # ---- BRANCH A ----
                "image_a1": _img_input("A", 1),
                "image_a2": _img_input("A", 2),
                "image_a3": _img_input("A", 3),
                "image_a4": _img_input("A", 4),
                "image_a5": _img_input("A", 5),
                "mask_a": (IO.MASK, {"tooltip": "Optional mask for branch A (inpainting; white = replace)."}),
                "model_a": (IO.COMBO, {"options": models, "default": DEFAULT_MODEL, "tooltip": "Branch A model"}),
                "size_a": (IO.COMBO, {"options": sizes, "default": "auto", "tooltip": size_tip}),
                "size_custom_a": (IO.STRING, {"default": "", "tooltip": custom_size_tip}),
                "quality_a": (IO.COMBO, {"options": qualities, "default": "auto", "tooltip": QUALITY_TOOLTIP}),
                "background_a": (IO.COMBO, {"options": backgrounds, "default": "opaque", "tooltip": bg_tip}),
                "n_a": (IO.INT, {"default": 1, "min": 1, "max": 8, "step": 1, "tooltip": "Branch A: number of images"}),
                "moderation_a": (IO.COMBO, {"options": moderations, "default": "low", "tooltip": "Branch A moderation"}),
                # ---- BRANCH B ----
                "image_b1": _img_input("B", 1),
                "image_b2": _img_input("B", 2),
                "image_b3": _img_input("B", 3),
                "image_b4": _img_input("B", 4),
                "image_b5": _img_input("B", 5),
                "mask_b": (IO.MASK, {"tooltip": "Optional mask for branch B (inpainting; white = replace)."}),
                "model_b": (IO.COMBO, {"options": models, "default": DEFAULT_MODEL, "tooltip": "Branch B model"}),
                "size_b": (IO.COMBO, {"options": sizes, "default": "auto", "tooltip": size_tip}),
                "size_custom_b": (IO.STRING, {"default": "", "tooltip": custom_size_tip}),
                "quality_b": (IO.COMBO, {"options": qualities, "default": "auto", "tooltip": QUALITY_TOOLTIP}),
                "background_b": (IO.COMBO, {"options": backgrounds, "default": "opaque", "tooltip": bg_tip}),
                "n_b": (IO.INT, {"default": 1, "min": 1, "max": 8, "step": 1, "tooltip": "Branch B: number of images"}),
                "moderation_b": (IO.COMBO, {"options": moderations, "default": "low", "tooltip": "Branch B moderation"}),
                **_advanced_inputs("_a"),
                **_advanced_inputs("_b"),
            },
        }

    RETURN_TYPES = (IO.IMAGE, IO.IMAGE)
    RETURN_NAMES = ("image_a", "image_b")
    FUNCTION = "api_call"
    CATEGORY = "OpenAI GPT Image (Parallel)"
    DESCRIPTION = cleandoc(__doc__ or "")

    async def api_call(
        self,
        prompt_a,
        prompt_b,
        api_base=None,
        auth_token=None,
        image_a1=None,
        image_a2=None,
        image_a3=None,
        image_a4=None,
        image_a5=None,
        mask_a=None,
        model_a=DEFAULT_MODEL,
        size_a="auto",
        size_custom_a="",
        quality_a="auto",
        background_a="opaque",
        n_a=1,
        moderation_a="low",
        image_b1=None,
        image_b2=None,
        image_b3=None,
        image_b4=None,
        image_b5=None,
        mask_b=None,
        model_b=DEFAULT_MODEL,
        size_b="auto",
        size_custom_b="",
        quality_b="auto",
        background_b="opaque",
        n_b=1,
        moderation_b="low",
        complexity_a="auto", markup_a=None, output_format_a="png", output_compression_a=100,
        complexity_b="auto", markup_b=None, output_format_b="png", output_compression_b=100,
    ):
        if output_format_a not in ("png", "webp", "jpeg") or output_format_b not in ("png", "webp", "jpeg"):
            raise ValueError("Use the GPT Image 2.6 Files x2 node for SVG or GIF output")
        config = read_user_config()
        if not api_base:
            api_base = config.get("api_base", "")
        if not auth_token:
            auth_token = config.get("auth_token", "") or os.environ.get("OPENAI_API_KEY", "")

        eff_size_a = _resolve_size(size_a, size_custom_a)
        eff_size_b = _resolve_size(size_b, size_custom_b)

        images_a = [image_a1, image_a2, image_a3, image_a4, image_a5]
        images_b = [image_b1, image_b2, image_b3, image_b4, image_b5]

        op_a = _build_operation(prompt_a, images_a, mask_a, quality_a, background_a, n_a, eff_size_a, moderation_a, api_base, auth_token, model_a, complexity_a, markup_a, output_format_a, output_compression_a)
        op_b = _build_operation(prompt_b, images_b, mask_b, quality_b, background_b, n_b, eff_size_b, moderation_b, api_base, auth_token, model_b, complexity_b, markup_b, output_format_b, output_compression_b)

        def _run(op):
            return validate_and_cast_response(op.execute())

        # Branch A and Branch B run concurrently, each with its own settings.
        img_a, img_b = await asyncio.gather(
            asyncio.to_thread(_run, op_a),
            asyncio.to_thread(_run, op_b),
        )
        return (img_a, img_b)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
class GPTImage26FilesParallelX2(GPTImage2GenerateParallelX2):
    """Preserve original PNG/WebP/JPEG/SVG/GIF bytes as files, including GIF frames."""

    @classmethod
    def INPUT_TYPES(cls):
        import copy
        schema = copy.deepcopy(super().INPUT_TYPES())
        for branch in ("a", "b"):
            schema["optional"][f"output_format_{branch}"][1]["options"] = ["png", "webp", "jpeg", "svg", "gif"]
            schema["optional"][f"frame_count_{branch}"] = (IO.INT, {"default": 9, "min": 2, "max": 16, "tooltip": "GIF only; total pixel budget also depends on size and n. Use size=auto initially."})
            schema["optional"][f"frame_duration_ms_{branch}"] = (IO.INT, {"default": 200, "min": 20, "max": 2000, "step": 10, "tooltip": "GIF only; delay per frame, rounded to 10 ms by the API."})
        schema["optional"]["filename_prefix"] = (IO.STRING, {"default": "GPTImage26"})
        return schema

    RETURN_TYPES = (IO.STRING, IO.STRING)
    RETURN_NAMES = ("files_a_json", "files_b_json")
    OUTPUT_NODE = True
    DESCRIPTION = "Two parallel requests with original file outputs. SVG is generation-only, n=1. GIF retains animation. Raster IMAGE nodes remain unchanged. Outputs are JSON arrays of saved file descriptors; no automatic vector rasterization."

    async def api_call(self, prompt_a, prompt_b, **kwargs):
        config = read_user_config()
        api_base = kwargs.get("api_base") or config.get("api_base", "")
        auth_token = kwargs.get("auth_token") or config.get("auth_token", "") or os.environ.get("OPENAI_API_KEY", "")
        ops = []
        formats = []
        # Build and validate BOTH branches before either paid request executes.
        for branch, prompt in (("a", prompt_a), ("b", prompt_b)):
            fmt = kwargs.get(f"output_format_{branch}", "png")
            formats.append(fmt)
            ops.append(_build_operation(
                prompt, [kwargs.get(f"image_{branch}{i}") for i in range(1, 6)],
                kwargs.get(f"mask_{branch}"), kwargs.get(f"quality_{branch}", "auto"),
                kwargs.get(f"background_{branch}", "opaque"), kwargs.get(f"n_{branch}", 1),
                _resolve_size(kwargs.get(f"size_{branch}", "auto"), kwargs.get(f"size_custom_{branch}", "")),
                kwargs.get(f"moderation_{branch}", "low"), api_base, auth_token,
                kwargs.get(f"model_{branch}", DEFAULT_MODEL), kwargs.get(f"complexity_{branch}", "auto"),
                kwargs.get(f"markup_{branch}"), fmt, kwargs.get(f"output_compression_{branch}", 100),
                kwargs.get(f"frame_count_{branch}", 9), kwargs.get(f"frame_duration_ms_{branch}", 200)))

        prefix = kwargs.get("filename_prefix", "GPTImage26")
        def run_and_save(index):
            return _save_original_files(ops[index].execute(), formats[index], prefix + "_" + ("a" if index == 0 else "b"))
        a, b = await asyncio.gather(asyncio.to_thread(run_and_save, 0), asyncio.to_thread(run_and_save, 1))
        return {"ui": {"images": [row for row in a + b if row["format"] != "svg"]},
                "result": (json.dumps(a), json.dumps(b))}


def _save_original_files(response, output_format, prefix):
    import folder_paths
    from pathlib import Path
    if not response.data:
        raise ValueError("No files returned from the Images API")
    directory, filename, counter, subfolder, _ = folder_paths.get_save_image_path(prefix, folder_paths.get_output_directory())
    results = []
    for entry in response.data:
        if entry.b64_json:
            content = base64.b64decode(entry.b64_json, validate=True)
        elif entry.url:
            result = requests.get(str(entry.url), timeout=(15, 120))
            result.raise_for_status()
            content = result.content
        else:
            raise ValueError("API file has neither b64_json nor url")
        if not content:
            raise ValueError("API returned an empty file")
        # Exclusive create avoids concurrent runs overwriting each other's files.
        while True:
            name = f"{filename}_{counter:05}_.{output_format}"
            path = Path(directory) / name
            try:
                with path.open("xb") as file:
                    file.write(content)
                break
            except FileExistsError:
                counter += 1
        results.append({"filename": name, "subfolder": subfolder, "type": "output", "format": output_format})
        counter += 1
    return results


NODE_CLASS_MAPPINGS = {
    "GPTImage2GenerateParallel": GPTImage1Generate,
    "GPTImage2GenerateParallelX2": GPTImage2GenerateParallelX2,
    "GPTImage26FilesParallelX2": GPTImage26FilesParallelX2,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "GPTImage2GenerateParallel": "OpenAI GPT Image (Async / Parallel)",
    "GPTImage2GenerateParallelX2": "OpenAI GPT Image x2 (Parallel a+b)",
    "GPTImage26FilesParallelX2": "OpenAI GPT Image 2.6 Files x2 (SVG / GIF / Raster)",
}
