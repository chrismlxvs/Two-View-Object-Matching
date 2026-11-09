#  Two-View Object Matching

This repository contains the project code and report source for detecting objects in two views of a tabletop scene, associating detections that belong to the same object, and counting each object once.

## Repository contents

- `code/baseline/`: detector preparation, training, prediction, and cross-view matching.
- `code/tools/`: evaluation and submission-format validation.
- `model/`: selected detector and scene-matcher checkpoints.
- `report/`: editable CVPR-style report source and its figures.
- `selected_configuration.json`: selected method settings and validation metrics.

The course dataset and assignment guide are not included. Obtain them from the course source and place `train/`, `val/`, and `test/` under `code/` (see `code/data/README.md`). The test split has no labels; keep it out of model selection.

## Setup

Use Python 3.10 or newer. Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r code/requirements.txt
```

PyTorch installation can depend on your operating system and CUDA version. If the command above does not install a suitable build, install PyTorch and torchvision using the instructions at [pytorch.org](https://pytorch.org/get-started/locally/), then install the remaining packages from `code/requirements.txt`.

The appearance matcher can use DINOv2 features through PyTorch Hub. Its first run may need internet access to fetch the pretrained backbone; later runs use the local Torch cache.

## Run inference with the selected checkpoints

From the repository root, after placing the course data under `code/`:

```bash
python code/baseline/scene_matcher.py \
  --predict-test \
  --weights model/yolo26n_amodal_best.pt \
  --checkpoint model/scene_matcher_960.pt \
  --prediction-out predictions.json

python code/tools/validate_submission.py \
  --pred predictions.json \
  --manifest code/test/manifest.json
```

To predict validation scenes instead, use `code/baseline/pipeline.py predict` with `--split val`. Refer to the course assignment guide for the input files and output format. Generated predictions, training runs, caches, and dataset contents are ignored by Git.

## Selected validation results

The selected configuration uses a shared YOLO26n detector at 960-pixel resolution and a scene-level matcher. On the labeled validation split it achieved detection F1 **0.9805**, association F1 **0.9531**, association F1 given correct detections **0.9840**, counting MAE **0.230**, and exact-count accuracy **0.790**. These are validation results; no test accuracy is claimed.

## Report

The editable report source is in `report/`. Build it from that directory with a LaTeX installation that includes BibTeX:

```bash
cd report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Third-party material

`report/cvpr.sty` and `report/ieeenat_fullname.bst` are from the CVPR author kit. See their headers and the [CVPR author-kit repository](https://github.com/cvpr-org/author-kit) for applicable terms. The coursework guide and dataset are provided by the course and are not included or relicensed here.
