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
# Model IDs and capabilities checked against OpenAI docs on 2026-09-08.
DEFAULT_MODEL = "gpt-image-2.5-flare"
GPT_IMAGE_MODELS = [
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
    "GPT Image 2.5 supports transparent backgrounds (PNG output). "
    "GPT Image 2 does not support transparent backgrounds."
)
QUALITY_TOOLTIP = (
    "Image quality affects cost and generation time. "
    "xhigh and max require GPT Image 2.5 Flare or Sunburst."
)


def _resolve_model(model, quality, background):
    """Validate capabilities without silently substituting another model."""
    model = (model or "").strip() or DEFAULT_MODEL
    is_image25 = any(
        model == family or model.startswith(family + "-")
        for family in ("gpt-image-2.5-flare", "gpt-image-2.5-sunburst")
    )
    if quality in ("xhigh", "max") and not is_image25:
        raise ValueError(
            f"quality={quality!r} requires GPT Image 2.5 Flare or Sunburst; "
            f"selected model is {model!r}. Use auto, low, medium or high."
        )
    if background == "transparent" and (
        model == "gpt-image-2" or model.startswith("gpt-image-2-")
    ):
        raise ValueError(
            "GPT Image 2 does not support transparent backgrounds. "
            "Select GPT Image 2.5, or use opaque/auto."
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
            img_response = requests.get(image_url)
            if img_response.status_code != 200:
                raise Exception("Failed to download the image")
            img = Image.open(io.BytesIO(img_response.content))

        img = img.convert("RGBA")

        # Convert to numpy array, normalize to float32 between 0 and 1
        img_array = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array)

        # Add to list of tensors
        image_tensors.append(img_tensor)

    return torch.stack(image_tensors, dim=0)


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
        self,
        prompt,
        seed=0,
        quality="auto",
        background="opaque",
        image=None,
        mask=None,
        n=1,
        size="1024x1024",
        moderation="low",
        api_base=None,
        auth_token=None,
        model=None,
    ):

        config = read_user_config()
        API_BASE = config.get("api_base", "")
        AUTH_TOKEN = config.get("auth_token", "")


        model = _resolve_model(model, quality, background)
        path = "images/generations"
        request_class = OpenAIImageGenerationRequest
        img_binaries = []
        mask_binary = None
        files = []

        if api_base is None or api_base == "":
            # 打印 未找到api_base，尝试使用用户在设置中配置的值
            print(f"No api_base found, trying to get it from settings.")
            api_base = API_BASE

        if auth_token is None or auth_token == "":
            print(f"No auth_token found, trying to get it from settings.")
            auth_token = AUTH_TOKEN

        if image is not None:
            path = "images/edits"
            request_class = OpenAIImageEditRequest

            batch_size = image.shape[0]

            for i in range(batch_size):
                single_image = image[i : i + 1]
                scaled_image = downscale_input(single_image).squeeze()

                image_np = (scaled_image.numpy() * 255).astype(np.uint8)
                img = Image.fromarray(image_np)
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format="PNG")
                img_byte_arr.seek(0)
                img_binary = img_byte_arr
                img_binary.name = f"image_{i}.png"

                img_binaries.append(img_binary)
                if batch_size == 1:
                    files.append(("image", img_binary))
                else:
                    files.append(("image[]", img_binary))

        if mask is not None:
            if image is None:
                raise Exception("Cannot use a mask without an input image")
            if image.shape[0] != 1:
                raise Exception("Cannot use a mask with multiple image")
            if mask.shape[1:] != image.shape[1:-1]:
                raise Exception("Mask and Image must be the same size")
            batch, height, width = mask.shape
            rgba_mask = torch.zeros(height, width, 4, device="cpu")
            rgba_mask[:, :, 3] = 1 - mask.squeeze().cpu()

            scaled_mask = downscale_input(rgba_mask.unsqueeze(0)).squeeze()

            mask_np = (scaled_mask.numpy() * 255).astype(np.uint8)
            mask_img = Image.fromarray(mask_np)
            mask_img_byte_arr = io.BytesIO()
            mask_img.save(mask_img_byte_arr, format="PNG")
            mask_img_byte_arr.seek(0)
            mask_binary = mask_img_byte_arr
            mask_binary.name = "mask.png"
            files.append(("mask", mask_binary))

        # Build the operation
        operation = SynchronousOperation(
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
                output_format="png",
                n=n,
                size=size,
                moderation=moderation,
            ),
            files=files if files else None,
            api_base=_normalize_api_base(api_base),
            auth_token=auth_token,
        )

        # Offload the blocking HTTP call to a thread so ComfyUI's event loop
        # stays free (lets independent async nodes overlap).
        def _blocking_call():
            return validate_and_cast_response(operation.execute())

        img_tensor = await asyncio.to_thread(_blocking_call)
        return (img_tensor,)


def _prepare_image_files(images, mask):
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
            image_np = (scaled_image.numpy() * 255).astype(np.uint8)
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

    if mask is not None:
        if not frames:
            raise Exception("Cannot use a mask without at least one input image")
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


def _build_operation(prompt, images, mask, quality, background, n, size, moderation, api_base, auth_token, model):
    model = _resolve_model(model, quality, background)
    path, request_class, files = _prepare_image_files(images, mask)
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
            output_format="png",
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
    ):
        config = read_user_config()
        if not api_base:
            api_base = config.get("api_base", "")
        if not auth_token:
            auth_token = config.get("auth_token", "")

        eff_size_a = _resolve_size(size_a, size_custom_a)
        eff_size_b = _resolve_size(size_b, size_custom_b)

        images_a = [image_a1, image_a2, image_a3, image_a4, image_a5]
        images_b = [image_b1, image_b2, image_b3, image_b4, image_b5]

        op_a = _build_operation(prompt_a, images_a, mask_a, quality_a, background_a, n_a, eff_size_a, moderation_a, api_base, auth_token, model_a)
        op_b = _build_operation(prompt_b, images_b, mask_b, quality_b, background_b, n_b, eff_size_b, moderation_b, api_base, auth_token, model_b)

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
NODE_CLASS_MAPPINGS = {
    "GPTImage2GenerateParallel": GPTImage1Generate,
    "GPTImage2GenerateParallelX2": GPTImage2GenerateParallelX2,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "GPTImage2GenerateParallel": "OpenAI GPT Image (Async / Parallel)",
    "GPTImage2GenerateParallelX2": "OpenAI GPT Image x2 (Parallel a+b)",
}
