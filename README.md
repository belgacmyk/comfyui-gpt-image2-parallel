# comfyui-gpt-image2-parallel

Async/parallel fork of [`lceric/comfyui-gpt-image`](https://github.com/lceric/comfyui-gpt-image).
Calls the **OpenAI `gpt-image-2` API with your own key** (no Comfy Org login). Runs **two fully independent generations in parallel**.

## Main node — OpenAI GPT Image x2 (Parallel a+b)

Two independent generations at once. **Each branch (A and B) has its own settings**; only the API connection is shared.

Per branch (A and B):
- `prompt_a` / `prompt_b`
- `image_a1`..`image_a5` / `image_b1`..`image_b5` — up to 5 reference images each (sent to images/edits)
- `mask_a` / `mask_b` — optional inpainting mask (white = replace; applied to the first image)
- `size_a` / `size_b` — preset **or** `custom`. Presets (official gpt-image-2 sizes): `auto`, `1024x1024`, `1536x1024`, `1024x1536`, `2048x2048` (2K), `2048x1152` (2K), `3840x2160` (4K), `2160x3840` (4K)
- `size_custom_a` / `size_custom_b` — used only when size = `custom`. Type any `WIDTHxHEIGHT` (e.g. `1792x1024`)
- `quality_a` / `quality_b` — `auto` / `low` / `medium` / `high`
- `background_a` / `background_b` — `auto` / `opaque` / `transparent`
- `n_a` / `n_b` — number of images
- `model_a` / `model_b` — dropdown: `gpt-image-2` (default), `gpt-image-1.5`, `gpt-image-1`, `gpt-image-1-mini`
- `moderation_a` / `moderation_b` — `low` / `auto`

Shared: `api_base` (e.g. `https://api.openai.com/v1`) and `auth_token` (your OpenAI key `sk-...`).

Outputs: `image_a`, `image_b`.

A branch with any image connected uses **images/edits**; with no image it uses **images/generations**. Both calls run concurrently via `asyncio.gather`.

### gpt-image-2 size rules (from OpenAI docs)
`size` accepts any resolution that satisfies: max edge ≤ `3840px`; both edges multiples of `16px`; long:short edge ratio ≤ `3:1`; total pixels between `655,360` and `8,294,400`. Outputs above `2560x1440` (2K) are considered experimental. `gpt-image-2` does **not** support `transparent` backgrounds.

## Second node — OpenAI GPT Image (Async / Parallel)
Single generation per node (one `image` + one `mask`), same size/quality/background options. Async; drop in as many as you want.

## Install — local
1. Unzip.
2. Move the folder `comfyui-gpt-image2-parallel` into `ComfyUI/custom_nodes/` (must be `custom_nodes/comfyui-gpt-image2-parallel/nodes_api.py`, no nested folder).
3. No pip install needed.
4. Restart ComfyUI.

## Install — ComfyDeploy
Add the repo Git URL as a custom node, then rebuild.

## Configure
- `api_base` = `https://api.openai.com/v1`
- `auth_token` = your OpenAI key (`sk-...`)
- `model_a` / `model_b` = pick from the dropdown (default `gpt-image-2`)

OpenAI gates GPT-Image behind **API Organization Verification** — complete it in your developer console first.

## Notes
- Cannot crash ComfyUI on startup; if the node doesn't appear, the console log shows a line starting with `[comfyui-gpt-image2-parallel]`.
- If a parameter errors (e.g. `moderation`), the API returns a clear 400; remove that option.
- Mask works most predictably with a single reference image.

## Credits & license
Fork of [`lceric/comfyui-gpt-image`](https://github.com/lceric/comfyui-gpt-image). MIT License preserved (see `LICENSE`).
