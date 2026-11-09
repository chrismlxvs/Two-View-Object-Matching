"""Training and inference baseline for CIE6004 Project 1."""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = Path(__file__).resolve().parent


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def image_link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def prepare(_args):
    """Create YOLO-format labels while preserving original amodal boxes."""
    for split in ("train", "val"):
        folder = ROOT / split
        manifest = read_json(folder / "manifest.json")
        annotations = read_json(folder / "gt.json")
        for pair_id, views in manifest["pairs"].items():
            records = annotations[pair_id]
            for view in ("a", "b"):
                image_src = folder / views[view]["image"]
                image_dst = BASE / "yolo" / "images" / split / image_src.name
                label_dst = BASE / "yolo" / "labels" / split / (image_src.stem + ".txt")
                image_link(image_src, image_dst)
                rows = []
                width, height = views[view]["width"], views[view]["height"]
                for record in records:
                    box = record[f"bbox_{view}"]
                    if box is None:
                        continue
                    x1, y1, x2, y2 = map(float, box)
                    xc, yc = (x1 + x2) / (2 * width), (y1 + y2) / (2 * height)
                    bw, bh = (x2 - x1) / width, (y2 - y1) / height
                    rows.append(f"0 {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}")
                label_dst.parent.mkdir(parents=True, exist_ok=True)
                label_dst.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    yaml = BASE / "dataset.yaml"
    yaml.write_text(
        "path: " + str((BASE / "yolo").resolve()).replace("\\", "/") + "\n"
        "train: images/train\nval: images/val\nnames:\n  0: object\n",
        encoding="utf-8",
    )
    print(f"Prepared labels and dataset config at {yaml}")


def train(args):
    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str(BASE / "dataset.yaml"), epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, device=args.device, workers=args.workers,
        project=str(BASE / "runs"), name="cie6004", exist_ok=True,
        seed=42, deterministic=True, patience=15, close_mosaic=10,
    )


def hist_feature(image, box):
    import cv2

    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    x1, x2 = max(0, int(x1)), min(w, int(math.ceil(x2)))
    y1, y2 = max(0, int(y1)), min(h, int(math.ceil(y2)))
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros(48, dtype=np.float32)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [8, 6], [0, 180, 0, 256]).reshape(-1)
    hist = hist.astype(np.float32)
    norm = np.linalg.norm(hist)
    return hist / norm if norm else hist


def load_dino_model():
    """Load the public pretrained DINOv2 small backbone through the official PyTorch Hub entry."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", trust_repo=True)
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, device


def dino_features(image, boxes, model, device, cache_path: Path, preserve_aspect=False):
    """Extract normalized DINOv2 crop descriptors, caching features by rounded box coordinates."""
    import cv2
    import torch
    import torch.nn.functional as F

    def key(box):
        return ",".join(f"{float(value):.1f}" for value in box)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached = {}
    if cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as data:
                cached = {str(k): v for k, v in zip(data["keys"].tolist(), data["features"])}
        except (OSError, ValueError, KeyError):
            cached = {}
    missing = [(i, box) for i, box in enumerate(boxes) if key(box) not in cached]
    h, w = image.shape[:2]
    crops = []
    missing_keys = []
    for _, box in missing:
        x1, y1, x2, y2 = box
        x1, x2 = max(0, int(x1)), min(w, int(math.ceil(x2)))
        y1, y2 = max(0, int(y1)), min(h, int(math.ceil(y2)))
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            crop = np.zeros((224, 224, 3), dtype=np.uint8)
        if preserve_aspect:
            ch, cw = crop.shape[:2]
            scale = min(224 / max(1, cw), 224 / max(1, ch))
            nw, nh = max(1, round(cw * scale)), max(1, round(ch * scale))
            resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
            # ImageNet-mean padding becomes zero after the standard normalization below.
            canvas = np.full((224, 224, 3), (104, 116, 124), dtype=np.uint8)
            x0, y0 = (224 - nw) // 2, (224 - nh) // 2
            canvas[y0:y0 + nh, x0:x0 + nw] = resized
            crop = canvas
        else:
            crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crops.append(crop.astype(np.float32) / 255.0)
        missing_keys.append(key(box))

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device)[None, :, None, None]
    with torch.inference_mode():
        for start in range(0, len(crops), 32):
            batch = np.stack(crops[start:start + 32])
            tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device)
            tensor = (tensor - mean) / std
            features = F.normalize(model(tensor).float(), dim=1).cpu().numpy()
            for crop_key, feature in zip(missing_keys[start:start + 32], features):
                cached[crop_key] = feature
    if missing:
        keys = list(cached)
        np.savez_compressed(cache_path, keys=np.asarray(keys), features=np.stack([cached[k] for k in keys]))
    return [cached[key(box)] for box in boxes]


def project_table_prior(box, source_cam, target_cam):
    """Map the source box bottom-center to z=0 and project to the target view."""
    try:
        src = np.asarray(source_cam["P_world_to_pixel"], dtype=float)
        dst = np.asarray(target_cam["P_world_to_pixel"], dtype=float)
        u, v = (box[0] + box[2]) / 2, box[3]
        affine = src[:2, [0, 1, 3]]
        xy = np.linalg.solve(affine[:, :2], np.array([u, v]) - affine[:, 2])
        pixel = dst @ np.array([xy[0], xy[1], 0.0, 1.0])
        if abs(pixel[2]) < 1e-9:
            return None
        point = pixel[:2] / pixel[2]
        if not np.isfinite(point).all():
            return None
        return point
    except (KeyError, ValueError, np.linalg.LinAlgError):
        return None


def association_costs(boxes_a, boxes_b, feats_a, feats_b, cam_a, cam_b,
                      geometry_weight=0.72, appearance="hist"):
    if not boxes_a or not boxes_b:
        return np.empty((len(boxes_a), len(boxes_b)), dtype=float)
    diagonal = math.sqrt(512**2 + 512**2)
    costs = np.zeros((len(boxes_a), len(boxes_b)), dtype=float)
    for i, a in enumerate(boxes_a):
        pa = project_table_prior(a, cam_a, cam_b)
        for j, b in enumerate(boxes_b):
            pb = project_table_prior(b, cam_b, cam_a)
            residuals = []
            if pa is not None:
                residuals.append(np.linalg.norm(pa - np.array([(b[0] + b[2]) / 2, b[3]])))
            if pb is not None:
                residuals.append(np.linalg.norm(pb - np.array([(a[0] + a[2]) / 2, a[3]])))
            geo = min(float(np.mean(residuals)) / diagonal, 1.0) if residuals else 0.5
            if appearance == "both":
                hist_sim = float(np.dot(feats_a[i]["hist"], feats_b[j]["hist"]))
                dino_sim = float(np.dot(feats_a[i]["dino"], feats_b[j]["dino"]))
                hist_app = max(0.0, 1.0 - hist_sim) / 2
                dino_app = max(0.0, 1.0 - dino_sim) / 2
                app = 0.65 * hist_app + 0.35 * dino_app
            else:
                app = max(0.0, 1.0 - float(np.dot(feats_a[i], feats_b[j]))) / 2
            costs[i, j] = geometry_weight * geo + (1.0 - geometry_weight) * app
    return costs


def predict(args):
    import cv2
    from scipy.optimize import linear_sum_assignment
    from ultralytics import YOLO

    split_dir = ROOT / args.split
    manifest = read_json(split_dir / "manifest.json")
    cameras = read_json(split_dir / "cameras.json")
    model = YOLO(args.weights)
    uses_dino = args.appearance in ("dino", "both")
    dino_model, dino_device = load_dino_model() if uses_dino else (None, None)
    geometry_weight = args.geometry_weight
    if geometry_weight is None:
        geometry_weight = {"hist": 0.72, "dino": 0.60, "both": 0.65}[args.appearance]
    predictions = {}
    for pair_id, views in manifest["pairs"].items():
        images = [cv2.imread(str(split_dir / views[v]["image"])) for v in ("a", "b")]
        if any(image is None for image in images):
            raise FileNotFoundError(f"Could not read both images for {pair_id}")
        boxes, features = [], []
        for view_index, image in enumerate(images):
            result = model.predict(image, imgsz=args.imgsz, conf=args.conf, iou=0.6,
                                   max_det=40, verbose=False)[0]
            bxs = result.boxes.xyxy.cpu().numpy().astype(float).tolist()
            boxes.append(bxs)
            if uses_dino:
                view_name = ("a", "b")[view_index]
                cache_path = BASE / "cache" / "dinov2_vits14" / args.split / f"{pair_id}_{view_name}.npz"
                dino_feat = dino_features(image, bxs, dino_model, dino_device, cache_path)
                if args.appearance == "both":
                    hist_feat = [hist_feature(image, box) for box in bxs]
                    features.append([{"hist": hist_feat[i], "dino": dino_feat[i]} for i in range(len(bxs))])
                else:
                    features.append(dino_feat)
            else:
                features.append([hist_feature(image, box) for box in bxs])
        costs = association_costs(boxes[0], boxes[1], features[0], features[1],
                                  cameras[pair_id]["a"], cameras[pair_id]["b"], geometry_weight,
                                  args.appearance)
        matched_a, matched_b = set(), set()
        records = []
        if costs.size:
            rows, cols = linear_sum_assignment(costs)
            for i, j in zip(rows, cols):
                if costs[i, j] <= args.match_threshold:
                    records.append({"bbox_a": boxes[0][i], "bbox_b": boxes[1][j]})
                    matched_a.add(int(i)); matched_b.add(int(j))
        for i, box in enumerate(boxes[0]):
            if i not in matched_a:
                records.append({"bbox_a": box, "bbox_b": None})
        for j, box in enumerate(boxes[1]):
            if j not in matched_b:
                records.append({"bbox_a": None, "bbox_b": box})
        predictions[pair_id] = records
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(predictions, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {len(predictions)} scene predictions to {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="convert train/val annotations to YOLO labels")
    p.set_defaults(func=prepare)
    p = sub.add_parser("train", help="fine-tune a pretrained Ultralytics detector")
    p.add_argument("--model", default="yolo26n.pt")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="")
    p.add_argument("--workers", type=int, default=4)
    p.set_defaults(func=train)
    p = sub.add_parser("predict", help="predict and associate one dataset split")
    p.add_argument("--weights", required=True)
    p.add_argument("--split", choices=("val", "test"), required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.15)
    p.add_argument("--match-threshold", type=float, default=0.42)
    p.add_argument("--appearance", choices=("hist", "dino", "both"), default="hist",
                   help="crop appearance feature: HSV histogram, pretrained DINOv2, or both")
    p.add_argument("--geometry-weight", type=float, default=None,
                   help="geometry cost weight (default: 0.72 for hist, 0.60 for DINOv2, 0.65 for both)")
    p.set_defaults(func=predict)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
