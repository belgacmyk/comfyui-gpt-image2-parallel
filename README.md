# comfyui-gpt-image2-parallel

Parallel OpenAI Images API nodes for ComfyUI and ComfyDeploy, using your own key. Fork of [lceric/comfyui-gpt-image](https://github.com/lceric/comfyui-gpt-image).

## Version 2.2.0 — GPT Image 2.6

Adds `gpt-image-2.6-flare` and `gpt-image-2.6-sunburst`. New nodes default to 2.6 Flare; **saved explicit model selections remain unchanged**. Previous 2.5 aliases/snapshots and older entries remain for workflow compatibility. OpenAI retired `gpt-image-1` on October 23, 2026; retaining that entry does not restore access. No automatic model fallback occurs.

Existing node IDs, output types, connections and previous widget positions are preserved. New optional controls are appended. API-format workflows can omit them. Open and save frontend workflows again after updating to serialize the appended widgets.

### Existing image nodes

- `GPTImage2GenerateParallel`: **OpenAI GPT Image (Async / Parallel)**, one `IMAGE` output.
- `GPTImage2GenerateParallelX2`: **OpenAI GPT Image x2 (Parallel a+b)**, independent `image_a` and `image_b` outputs. Calls overlap through `asyncio.gather` and worker threads.

Each x2 branch retains its prompt, five image sockets (batches flatten to at most 16 references), mask, model, size/custom size, quality, background, image count and moderation. The single node accepts an image batch, but a mask requires exactly one image.

New controls, per branch (`_a` / `_b`) or unsuffixed in the single node:

| Input | Behavior |
| --- | --- |
| `complexity` | `auto`, `low`, `medium`, `high`. Explicit control requires 2.6. `auto` omits the field (current API default: high). Planning effort is separate from quality. |
| `markup` | One transparent RGBA annotation overlay matching the first reference dimensions. Supported on 2.5/2.6 edits; cannot combine with mask. |
| `output_format` | `png` (default), `webp`, `jpeg`. Tensors preserve alpha when present. Use Files x2 for SVG/GIF. |
| `output_compression` | 0–100, sent only for JPEG/WebP. |

Quality remains `auto`, `low`, `medium`, `high`, `xhigh`, `max`; the last two require 2.5/2.6. Transparent output supports PNG/WebP. GPT Image 2 transparency is now available in preview. Availability depends on project/provider access.

**Markup:** standard Load Image returns RGB and a separate inverted mask. Use Join Image with Alpha or an RGBA loader before connecting `markup`. Flattened overlays are rejected. Reference and overlay use the same existing input-downscaling policy. Reference socket/batch order is retained.

The x2 node retains `size=custom` and `size_custom`. Existing presets through 3840×2160 / 2160×3840 remain; verify current model size limits. GIF has a separate total pixel budget. The legacy single-node `seed` is retained but never sent to the API.

### New file node — SVG and animated GIF

`GPTImage26FilesParallelX2`: **OpenAI GPT Image 2.6 Files x2 (SVG / GIF / Raster)**.

Same two independent branches, with SVG/GIF added to `output_format_a/b`. Saves original bytes in the ComfyUI output folder under `filename_prefix` (default `GPTImage26`) plus a branch suffix. `files_a_json` and `files_b_json` contain JSON arrays of `filename`, `subfolder`, `type`, `format`. These are local file descriptors, **not public URLs**. SVG is not executed or rasterized; GIF preserves all animation frames. Raster/GIF artifacts appear in UI image outputs; SVG is available in the output folder and file descriptors.

| Format | Rules |
| --- | --- |
| SVG | 2.6, text-to-image only, n=1; no reference images, masks, markup or compression. Saves original XML bytes. |
| GIF | 2.6, generation or still-reference editing. `frame_count_a/b`: 2–16, default 9. `frame_duration_ms_a/b`: 20–2000, default 200, rounded by the API to 10 ms. No mask, custom compression or transparent-background setting. Start with size=auto; size/frame count/n must fit provider limits. |

SVG/GIF can require project preview access. Both branches are validated before either paid request starts. Raster nodes reject SVG/GIF to avoid losing vector content or animation frames. Format-specific fields are omitted for other formats.

## Install and configure

1. Update the repository under `ComfyUI/custom_nodes/comfyui-gpt-image2-parallel`, preserving local edits and private config.
2. Use the ComfyUI Python environment: Pydantic 2, Torch, NumPy, Pillow, requests.
3. Restart ComfyUI and refresh the browser. Select 2.6 explicitly in existing workflows.
4. Set `api_base=https://api.openai.com/v1` and your key. Existing `auth_token`/private `config.json` behavior is retained; `OPENAI_API_KEY` is also a fallback. Prefer server secrets to credentials embedded in workflows/PNG metadata. Authorization headers are redacted in debug logs.

**ComfyDeploy:** publish/merge the new repository revision, rebuild the machine/deployment that installs it, refresh the schema and save the selected model in a new workflow/deployment version. GitHub changes alone do not update running workers. For SVG/GIF, configure the deployment's file-output/upload contract: local file descriptors cannot connect to `IMAGE`-only outputs. Existing PNG workflows retain their old x2 output contract. No production deployment was rebuilt for this update.

Third-party API bases must independently support these exact model IDs, parameters and formats. Dropdowns do not grant model access or prove a provider's upstream model. No paid generation or live frontend rendering was performed in this update.

## Validation

Run `python -B -m unittest discover -s tests -v`. Tests cover prior input order/output contracts, model/quality/complexity selection, RGBA markup, JSON/multipart serialization through the real HTTP adapter with mocked transport, preflight of both branches, concurrent settings, PNG alpha, and byte-exact SVG/GIF saving with multiple frames. Tests use ComfyUI type shims and make no paid calls. Live frontend behavior and account-specific model/preview access still require verification.

Official sources checked November 7, 2026:

- [Image generation guide](https://developers.openai.com/api/docs/guides/image-generation)
- [Create image](https://developers.openai.com/api/reference/resources/images/methods/generate)
- [Create image edit](https://developers.openai.com/api/reference/resources/images/methods/edit)
- [API changelog](https://developers.openai.com/api/docs/changelog)

MIT License preserved; see LICENSE.
