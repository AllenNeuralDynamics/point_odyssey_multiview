# point_odyssey_multiview

Code to generate multi-view synthetic datasets for 3D point tracking: humans and animals rendered from several static cameras, with ground-truth 3D trajectories and per-camera visibility.

This repository is built on the [PointOdyssey](https://github.com/y-zheng18/point_odyssey) data generation pipeline ([paper](https://arxiv.org/abs/2307.15055), [project page](https://pointodyssey.com/)), which in turn builds on [Kubric](https://github.com/google-research/kubric). It started from PointOdyssey commit `a053342`; the first commit in this history is that unmodified snapshot, so `git diff` against it shows exactly what changed.

The resulting datasets are released as [Multi-View Point Odyssey (human)](https://huggingface.co/datasets/ai-hhmi/mv-po-human) and [Multi-View Point Odyssey (animal)](https://huggingface.co/datasets/ai-hhmi/mv-po-animal), and are used to train and evaluate [Tracktail](https://arxiv.org/abs/2611.26779).

## What is different from PointOdyssey

- Human and animal rendering and export scripts extended for multi-camera scenes, with ground-truth 3D trajectories and per-camera visibility (`render_human.py`, `render_animal.py`, `render_single.py`, `export_annotation_*.py`).
- Changes to trajectory generation and EXR output (`utils/gen_tracking.py`, `utils/openexr_utils.py`, `utils/file_io.py`).
- `extract_textures.py` extracts image textures from `.blend` files into per-asset folders.

## Setup

Setup is the same as for PointOdyssey: install Blender 3.1+ and the required Blender add-ons (Rokoko Studio Live, SMPL-X, Stop Motion OBJ), then

```
conda create -n point python=3.9
conda activate point
pip install -r requirements.txt
conda install -c conda-forge openexr-python
```

See the [PointOdyssey README](https://github.com/y-zheng18/point_odyssey#introduction) for details on the add-ons and the data layout expected under `data/`.

## Usage

Example commands for multi-view rendering are in `scripts/`, for example `scripts/render_animal_multiview.sh`.

## License and attribution

This repository is released under the MIT License (see `LICENSE`). It contains code from PointOdyssey (MIT) and code modified from Kubric (Apache 2.0); their licenses are in `third_party/` and the original notice is kept in `LICENSE.md`.
