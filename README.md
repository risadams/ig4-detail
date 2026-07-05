# ig4-detail — Ideogram 4 Detail Enhancer for ComfyUI

A drop-in ComfyUI node pack that adds photographic micro-detail to **Ideogram 4** outputs (or any generated image): realistic skin texture, crisp eyes, hair-strand detail, midtone clarity, fine sharpening and film grain.

Ideogram 4 (like most generative models) tends to produce slightly waxy skin, soft eyes and mushy hair. This node rebuilds plausible micro-detail with fast, deterministic image processing — **no second diffusion pass, no model downloads, no VRAM cost**. It takes an `IMAGE` and returns an `IMAGE`, so it chains directly between your Ideogram node and `Save Image`.

```
[Ideogram 4] ──IMAGE──▶ [IG4 Detail Enhancer] ──IMAGE──▶ [Save Image]
```

## Installation

Clone into your ComfyUI `custom_nodes` folder and install the (tiny) requirements:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/risadams/ig4-detail.git
pip install -r ig4-detail/requirements.txt   # numpy + opencv (usually already present)
```

Restart ComfyUI. The nodes appear under **image → ideogram4**.

## Nodes

### IG4 Detail Enhancer (Ideogram 4)

The full node. Inputs an `IMAGE`, outputs the enhanced `IMAGE` plus the auto-detected `skin_mask` and `eye_mask` (handy for previewing what the node is targeting).

| Knob | Default | Range | What it does |
|---|---|---|---|
| `overall_strength` | 1.0 | 0–2 | Master wet/dry blend. 0 = bypass, 1 = full effect, >1 exaggerates everything. |
| `skin_smooth` | 0.25 | 0–1 | Bilateral pass inside skin only — removes the waxy AI sheen/banding *before* texture is added, so pores read as pores instead of noise-on-noise. |
| `skin_texture` | 1.0 | 0–2 | Pore-scale band-passed texture added inside detected skin areas, weighted toward midtones. |
| `eye_enhance` | 1.0 | 0–2 | Iris sharpening + local contrast + slight saturation inside detected eyes. Makes catchlights pop. |
| `hair_detail` | 1.0 | 0–2 | High-frequency boost weighted toward strand-dense non-skin regions (hair, fabric, foliage). |
| `clarity` | 0.5 | 0–2 | Large-radius, midtone-weighted local contrast (Lightroom-style "clarity"), halo-resistant. |
| `fine_sharpen` | 0.6 | 0–2 | Global unsharp mask applied after the detail passes. |
| `sharpen_radius` | 1.2 | 0.4–3 | Unsharp-mask radius in pixels. |
| `grain_amount` | 0.12 | 0–1 | Photographic luminance grain — the cheapest realism win there is; ties everything together. |
| `grain_size` | 1.0 | 0.5–3 | Grain scale in pixels (bigger = coarser, more "film"). |
| `grain_seed` | 0 | — | Seed for grain and skin-texture noise. Deterministic: same seed = same grain. |
| `saturation` | 1.0 | 0.5–1.5 | Final saturation trim. |
| `detect_faces` | true | — | Face/eye detection for targeted eye enhancement (see backends below). Skin masking stays active either way. |
| `mask` (optional) | — | — | Limits the whole effect to a region (white = enhance). |

### IG4 Detail Enhancer — Simple

One slider (`strength`) plus a grain seed, using tuned portrait defaults. Use it when you just want "make it look real" without touching knobs.

## Example workflow

Load [`workflows/ig4_detail_enhance.json`](workflows/ig4_detail_enhance.json) in ComfyUI (drag the file onto the canvas). It wires:

```
Load Image ─▶ IG4 Detail Enhancer ─▶ Save Image
                     ├─ skin_mask ─▶ Mask To Image ─▶ Preview
                     └─ eye_mask  ─▶ Mask To Image ─▶ Preview
```

To use it on live Ideogram generations, delete the `Load Image` node and connect the `IMAGE` output of your Ideogram node (e.g. the built-in Ideogram API node) into the enhancer's `image` input. That's the entire integration — the node works on top of any Ideogram model output, or any other image for that matter.

## Recipes

| Look | Settings |
|---|---|
| **Natural portrait** (default) | defaults |
| **Editorial / beauty** | `skin_smooth 0.4`, `skin_texture 1.2`, `eye_enhance 1.4`, `clarity 0.3`, `grain_amount 0.08` |
| **Gritty film** | `clarity 1.0`, `grain_amount 0.35`, `grain_size 1.8`, `fine_sharpen 0.8` |
| **Non-portrait** (product, landscape) | `detect_faces off`, `skin_texture 0`, `hair_detail 1.4`, `clarity 0.8` |
| **Subtle polish** | `overall_strength 0.5` on defaults |

Tips:

- Preview the `skin_mask` / `eye_mask` outputs first — if the skin mask catches skin-toned background, feed a `mask` input to constrain the effect.
- `overall_strength` is the fastest way to A/B: queue at 0, 0.5, 1.0 with everything else fixed.
- Grain is applied after sharpening and is luminance-weighted, so shadows and highlights stay clean.
- The node is fully deterministic for a given seed — safe to use in batch pipelines.

## How it works

1. **Detection** — faces and eye centers are located with the first available backend:
   1. **YuNet** (`cv2.FaceDetectorYN`, OpenCV ≥ 4.5.4 and 5.x) — real eye landmarks. The tiny (~350 KB) ONNX model is auto-downloaded once into `ig4_detail/models/` on first use (set `IG4_DETAIL_NO_DOWNLOAD=1` to forbid this).
   2. **Haar cascades** (OpenCV 4.x) — face boxes + cascade eyes, anthropometric fallback.
   3. **Skin-blob heuristic** — face-shaped skin regions, anthropometric eyes. No downloads, works on any OpenCV build.

   Separately, a YCrCb color model builds a feathered skin mask covering faces, hands and arms.
2. **Skin** — mild bilateral smoothing inside the skin mask kills plastic sheen, then band-passed noise at pore frequency (~1–2 px) is added back, midtone-weighted.
3. **Eyes** — small-radius unsharp + local contrast + saturation inside feathered eye circles.
4. **Hair / fine structure** — existing high frequencies are amplified where high-frequency energy is dense and skin is absent.
5. **Global finish** — midtone clarity, unsharp mask, saturation trim, luminance-weighted film grain, then a master blend against the original.

## License

MIT
