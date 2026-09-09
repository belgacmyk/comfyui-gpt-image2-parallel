# comfyui-gpt-image2-parallel

Async/parallel fork of [`lceric/comfyui-gpt-image`](https://github.com/lceric/comfyui-gpt-image).
Calls the **OpenAI Images API (GPT Image 2.5 and earlier) with your own key** (no Comfy Org login). Runs **two fully independent generations in parallel**.

## Version 2.1.0: GPT Image 2.5

Both nodes support the exact API IDs `gpt-image-2.5-flare` and `gpt-image-2.5-sunburst`, plus their pinned snapshots `gpt-image-2.5-flare-2026-09-08` and `gpt-image-2.5-sunburst-2026-09-08`. New nodes default to Flare with quality `auto`.

The new quality levels `xhigh` and `max` are available only for GPT Image 2.5. Incompatible older-model selections raise a local error before execution. Both request schemas accept the new quality values. GPT Image 2.5 supports transparent output; the nodes explicitly request PNG and preserve alpha. GPT Image 2 still rejects `transparent`.

Existing node IDs, input order, connections and saved model selections are preserved. **Existing workflows keep their old model until you change `model`, `model_a` or `model_b`.** Model access failures do not silently switch to an older model. The legacy `seed` widget remains for compatibility but is not sent to the image API.

This update also normalizes the API base URL so that `/v1` is retained with or without a trailing slash, and fixes the single node's error when a mask is supplied without an image. It does not change credentials, configuration files, image input resizing or the parallel execution architecture.

For an existing installation, back up local changes, apply the update files and restart ComfyUI. Refresh the browser and select the new model in your saved workflow. A ComfyDeploy installation needs a rebuild after the updated code has actually been published to its repository.

The API project and configured provider must have access to the selected model. A dropdown entry does not grant access. A third-party API base must independently support the model IDs and parameters. Workflow metadata records the requested model; it is not independent proof of the provider's backend model.

Official references checked on September 8, 2026:
[Flare](https://developers.openai.com/api/docs/models/gpt-image-2.5-flare),
[Sunburst](https://developers.openai.com/api/docs/models/gpt-image-2.5-sunburst),
[Image generation guide](https://developers.openai.com/api/docs/guides/image-generation).

## Main node — OpenAI GPT Image x2 (Parallel a+b)

Two independent generations at once. **Each branch (A and B) has its own settings**; only the API connection is shared.

Per branch (A and B):
- `prompt_a` / `prompt_b`
- `image_a1`..`image_a5` / `image_b1`..`image_b5` — up to 5 reference images each (sent to images/edits)
- `mask_a` / `mask_b` — optional inpainting mask (white = replace; applied to the first image)
- `size_a` / `size_b` — preset **or** `custom`. Presets (GPT Image 2 / 2.5 sizes): `auto`, `1024x1024`, `1536x1024`, `1024x1536`, `2048x2048` (2K), `2048x1152` (2K), `3840x2160` (4K), `2160x3840` (4K)
- `size_custom_a` / `size_custom_b` — used only when size = `custom`. Type any `WIDTHxHEIGHT` (e.g. `1792x1024`)
- `quality_a` / `quality_b` — `auto` / `low` / `medium` / `high` / `xhigh` / `max` (last two require 2.5)
- `background_a` / `background_b` — `auto` / `opaque` / `transparent`
- `n_a` / `n_b` — number of images
- `model_a` / `model_b` — dropdown: `gpt-image-2.5-flare` (default), `gpt-image-2.5-sunburst`, dated 2.5 snapshots, `gpt-image-2`, `gpt-image-1.5`, `gpt-image-1`, `gpt-image-1-mini`
- `moderation_a` / `moderation_b` — `low` / `auto`

Shared: `api_base` (e.g. `https://api.openai.com/v1`) and `auth_token` (your OpenAI key `sk-...`).

Outputs: `image_a`, `image_b`.

A branch with any image connected uses **images/edits**; with no image it uses **images/generations**. Both calls run concurrently via `asyncio.gather`.

### GPT Image 2 / 2.5 size rules (from OpenAI docs)
`size` accepts any resolution that satisfies: max edge ≤ `3840px`; both edges multiples of `16px`; long:short edge ratio ≤ `3:1`; total pixels between `655,360` and `8,294,400`. Outputs above `2560x1440` (2K) are considered experimental. `gpt-image-2` does **not** support `transparent` backgrounds.

## Second node — OpenAI GPT Image (Async / Parallel)
One request per node, optional `image` (which can be a batch) and `mask`. A mask requires exactly one image. Same model/quality/background options and preset sizes; the custom-size field belongs to the x2 node only.

## Install — local
1. Unzip.
2. Move the folder `comfyui-gpt-image2-parallel` into `ComfyUI/custom_nodes/` (must be `custom_nodes/comfyui-gpt-image2-parallel/nodes_api.py`, no nested folder).
3. Use the Python environment and dependencies belonging to ComfyUI.
4. Restart ComfyUI.

## Install — ComfyDeploy
Add the repo Git URL as a custom node, then rebuild.

## Configure
- `api_base` = `https://api.openai.com/v1`
- `auth_token` = your OpenAI key (`sk-...`)
- `model_a` / `model_b` = pick from the dropdown (default `gpt-image-2.5-flare`)

API organization verification may be required for your project.

## Notes
- If the node does not appear, the console log shows a line starting with `[comfyui-gpt-image2-parallel]`.
- If a parameter errors (e.g. `moderation`), the API returns a clear 400; remove that option.
- Mask works most predictably with a single reference image.

## Offline tests

Run `python -m unittest discover -s tests -v` in the node folder using its Python environment. The tests use real request schemas, Pydantic, Torch, NumPy and Pillow. ComfyUI types and the operation transport are stubbed. They test model/quality selection, serialization, editing inputs, alpha preservation, URL normalization, compatibility and concurrent branches. They do not call the paid API, exercise the real HTTP client or launch the ComfyUI frontend. A live test in your own environment is still needed.

## Credits & license
Fork of [`lceric/comfyui-gpt-image`](https://github.com/lceric/comfyui-gpt-image). MIT License preserved (see `LICENSE`).
