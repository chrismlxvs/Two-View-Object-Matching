# Dataset placement

Download the CIE6004 Project 1 dataset from the course source. Keep the dataset out of Git and place its `train/`, `val/`, and `test/` directories directly inside `code/`, alongside `baseline/` and `tools/`:

```text
code/
  train/  # images, manifest.json, gt.json
  val/    # images, manifest.json, gt.json
  test/   # images, manifest.json; no ground truth
```

The training and validation scripts resolve these paths relative to `code/`. Do not add test labels or use the test split to tune the model.
