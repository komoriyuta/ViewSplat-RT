# ViewSplat-RT

## Real-Time Version of the Original ViewSplat Repository

This repository makes the original **ViewSplat** paper implementation run as a real-time two-camera inference system. It is not a separate model or a rewrite from scratch; it is the original ViewSplat implementation adapted for live cameras, real-time 3DGS generation, CUDA rendering, and browser/OpenCV visualization.

- Original repository: https://github.com/cvlab-uos/ViewSplat
- Original project page: https://cvlab-uos.github.io/ViewSplat/
- Original paper: https://arxiv.org/abs/2603.25265

ViewSplat-RT keeps the ViewSplat model code and makes the paper implementation usable as a live real-time pipeline: `uv`/`pyproject.toml` setup, optimized SPFSplatV2 inference, live two-camera input, CUDA Gaussian rendering, OpenCV preview, and Viser browser streaming.

On an RTX 4060 Ti 16GB, the optimized real-time path reaches about **20 FPS** with the SPFSplatV2 preset.

<p align="center">
  <video src="demo_video/realtime_demo.mp4" width="90%" controls muted playsinline></video>
</p>
<p align="center">
  <a href="demo_video/realtime_demo.mp4">Open demo video</a>
</p>

## What Was Added for Real Time

- `uv` based environment setup through `pyproject.toml` and `uv.lock`
- Real-time two-camera inference entrypoint: `src.realtime_two_camera`
- Automatic checkpoint download from Hugging Face
- OpenCV raw camera preview with `--show-cameras`
- CUDA-rendered midpoint/virtual camera preview with `--show`
- Viser browser streaming with `--viser`
- Camera sweep between the two estimated cameras with `--sweep`
- Presets for RE10K/ACID and SPFSplatV2/SPFSplatV2-L checkpoints
- Vendored `third_party` checkout removed; `diff_gauss_pose` is prepared from the original upstream dependency by a local script, then installed by `uv` from `pyproject.toml`

## Installation

Clone this fork:

```bash
git clone https://github.com/komoriyuta/ViewSplat-RT.git
cd ViewSplat-RT
```

Prepare the original `diff_gauss_pose` dependency, then create the environment:

```bash
python3 scripts/prepare_diff_gauss_pose.py
CUDA_HOME=/usr/local/cuda-13.2 PATH=/usr/local/cuda-13.2/bin:$PATH uv sync --no-dev
```

The project uses Python 3.11. PyTorch is installed from the CUDA 13.0 wheel index. The CUDA Gaussian rasterizer is the original `diff_gauss_pose` dependency from `slothfulxtx/diff-gaussian-rasterization@pose`; `scripts/prepare_diff_gauss_pose.py` clones the pinned upstream revision into ignored `.uv-local/`, adds the minimal build metadata needed by `uv`, and applies the CUDA 13/C++20 compatibility include. There is no checked-in `third_party` directory.

## Real-Time Two-Camera Inference

List available cameras:

```bash
uv run python -m src.realtime_two_camera --list-cameras
```

Run two-camera inference with browser visualization:

```bash
uv run python -m src.realtime_two_camera \
  --left-camera 1 \
  --right-camera 3 \
  --preset re10k-spfv2 \
  --viser
```

Show the raw left/right camera streams:

```bash
uv run python -m src.realtime_two_camera \
  --left-camera 1 \
  --right-camera 3 \
  --preset re10k-spfv2 \
  --show-cameras
```

Show the CUDA-rendered virtual camera image in OpenCV:

```bash
uv run python -m src.realtime_two_camera \
  --left-camera 1 \
  --right-camera 3 \
  --preset re10k-spfv2 \
  --show
```

Sweep the virtual camera between the two estimated cameras:

```bash
uv run python -m src.realtime_two_camera \
  --left-camera 1 \
  --right-camera 3 \
  --preset re10k-spfv2 \
  --viser \
  --sweep \
  --sweep-period 1.0
```

Use the Large ACID preset:

```bash
uv run python -m src.realtime_two_camera \
  --left-camera 1 \
  --right-camera 3 \
  --preset acid-spfv2l \
  --viser
```

## Presets

| Preset | Checkpoint | Notes |
| :--- | :--- | :--- |
| `re10k-spfv2` | `re10k_spfv2_viewsplat.ckpt` | Default real-time preset |
| `acid-spfv2` | `acid_spfv2_viewsplat.ckpt` | ACID SPFSplatV2 |
| `re10k-spfv2l` | `re10k_spfv2l_viewsplat.ckpt` | Large VGGT-backed model |
| `acid-spfv2l` | `acid_spfv2l_viewsplat.ckpt` | Large ACID model |

Missing checkpoints are downloaded automatically into `pretrained_weights`.

## Useful Options

```bash
--show-cameras              Show raw left/right camera frames
--show                      Show CUDA-rendered virtual camera output
--viser                     Stream CUDA-rendered output to Viser
--viser-port 8080           Viser server port
--viser-jpeg-quality 75     JPEG quality for browser streaming
--viser-stream-scale 2      CPU upscale before Viser JPEG streaming
--sweep                     Sweep virtual camera between the two cameras
--sweep-period 1.0          Sweep period in seconds
--disable-view-dependent-head
                            Optional speed/quality tradeoff for SPFSplatV2-L
```

## Evaluation

The original evaluation path is still available. For example:

```bash
uv run python -m src.main +experiment=spfv2_viewsplat/re10k_eval
```

The default evaluation command uses:

```text
pretrained_weights/re10k_spfv2_viewsplat.ckpt
```

## Camera Conventions

This fork follows the original ViewSplat/pixelSplat camera convention. Intrinsics are normalized by image size. Extrinsics are OpenCV-style camera-to-world matrices: `+X` right, `+Y` down, `+Z` forward.

## Attribution

This is a real-time inference fork of the original ViewSplat implementation by the ViewSplat authors:

- https://github.com/cvlab-uos/ViewSplat
- https://cvlab-uos.github.io/ViewSplat/
- https://arxiv.org/abs/2603.25265

ViewSplat builds on SPFSplatV2:

- https://github.com/ranrhuang/SPFSplatV2

## Citation

If you use the ViewSplat model or paper implementation, cite the original ViewSplat work:

```bibtex
@article{Jeong2026viewsplat,
  title={ViewSplat: View-Adaptive 3D Gaussian Splatting for Feed-Forward Synthesis},
  author={Jeong, Moonyeon and Min, Seunggi and Lee, Suhyeon and Seong, Hongje},
  journal={arXiv preprint arXiv: 2603.25265},
  year={2026}
}
```
