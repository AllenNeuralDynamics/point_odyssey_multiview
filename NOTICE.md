# Notice

`point_odyssey_multiview` is released under the MIT License (see `LICENSE`). Parts of it are derived from other projects, which keep their original licenses. The license texts are in `third_party/`.

## PointOdyssey (MIT)

Most of this repository is derived from the [PointOdyssey](https://github.com/y-zheng18/point_odyssey) data generation code, Copyright (c) 2024 PointOdyssey Authors, released under the MIT License (`third_party/point_odyssey/LICENSE.md`). The first commit in this repository is an unmodified snapshot of upstream commit `a053342`, so `git diff` against it shows every change made here.

Zheng, Harley, Shen, Wetzstein, Guibas. *PointOdyssey: A Large-Scale Synthetic Dataset for Long-Term Point Tracking.* ICCV 2023.

## Kubric (Apache License 2.0)

The following code is modified from [Kubric](https://github.com/google-research/kubric), Copyright 2024 The Kubric Authors, and is licensed under the Apache License, Version 2.0 (`third_party/kubric/LICENSE.md`):

- `utils/openexr_utils.py`
- `set_up_exr_output_node` in `render_animal.py`, `render_human.py` and `render_single.py`

These files keep their original license headers. They were modified for multi-view rendering.

## Changes in this repository

Changes and additions made here are Copyright (c) 2026 Allen Institute for Neural Dynamics and released under the MIT License:

- multi-camera rendering and export for humans and animals (`render_*.py`, `export_annotation_*.py`)
- trajectory generation and EXR output (`utils/gen_tracking.py`, `utils/openexr_utils.py`, `utils/file_io.py`)
- `extract_textures.py`

## Datasets

The generated datasets are released separately under CC BY 4.0.
