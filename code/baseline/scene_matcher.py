"""Train and evaluate a compact context-aware matcher for paired YOLO detections."""
from __future__ import annotations
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import pipeline
import pair_matcher as pair

ROOT, BASE = pipeline.ROOT, pipeline.BASE


class SceneNet(nn.Module):
    """Message-passing scorer: each candidate link sees its row and column context."""
    def __init__(self, d=32):
        super().__init__()
        self.encode = nn.Sequential(nn.Linear(10, 64), nn.GELU(), nn.Linear(64, d), nn.GELU())
        self.link = nn.Sequential(nn.Linear(d * 5, 64), nn.GELU(), nn.Linear(64, 1))
        self.unmatched_a = nn.Sequential(nn.Linear(d * 2, 32), nn.GELU(), nn.Linear(32, 1))
        self.unmatched_b = nn.Sequential(nn.Linear(d * 2, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x):
        # x: [detections A, detections B, 10 pair features]
        n, m, _ = x.shape
        if n == 0 or m == 0:
            return x.new_empty((n, m)), x.new_zeros(n), x.new_zeros(m)
        e = self.encode(x)
        row_mean, col_mean = e.mean(1), e.mean(0)
        row_max, col_max = e.max(1).values, e.max(0).values
        row_ctx = torch.cat((row_mean, row_max), -1)
        col_ctx = torch.cat((col_mean, col_max), -1)
        link_ctx = torch.cat((e,
            row_mean[:, None, :].expand(n, m, -1),
            row_max[:, None, :].expand(n, m, -1),
            col_mean[None, :, :].expand(n, m, -1),
            col_max[None, :, :].expand(n, m, -1)), -1)
        return self.link(link_ctx).squeeze(-1), self.unmatched_a(row_ctx).squeeze(-1), self.unmatched_b(col_ctx).squeeze(-1)


def map_detections(boxes, records, view):
    gts = [r[f"bbox_{view}"] for r in records if r[f"bbox_{view}"] is not None]
    candidates = sorted(((pair.iou(box, gb), pi, gi) for pi, box in enumerate(boxes)
                         for gi, gb in enumerate(gts) if pair.iou(box, gb) >= .5), reverse=True)
    pred_to_gt, used_p, used_g = {}, set(), set()
    for ov, pi, gi in candidates:
        if pi not in used_p and gi not in used_g:
            pred_to_gt[pi] = gi
            used_p.add(pi); used_g.add(gi)
    return pred_to_gt


def scene_items(predictions, gt=None):
    items = []
    for pid, data in predictions.items():
        a, b = data["boxes"]
        x = data["x"].reshape((len(a), len(b), 10)).astype(np.float32)
        geo = x[:, :, :2].copy()
        target_a = np.full(len(a), len(b), dtype=np.int64)
        target_b = np.full(len(b), len(a), dtype=np.int64)
        if gt is not None and len(a) and len(b):
            ma = map_detections(a, gt[pid], "a")
            mb = map_detections(b, gt[pid], "b")
            gt_to_b = {g: j for j, g in mb.items()}
            gt_to_a = {g: i for i, g in ma.items()}
            for i, g in ma.items():
                if g in gt_to_b: target_a[i] = gt_to_b[g]
            for j, g in mb.items():
                if g in gt_to_a: target_b[j] = gt_to_a[g]
        items.append({"pid": pid, "x": x, "geo": geo, "a": a, "b": b, "scores": data["scores"],
                      "target_a": target_a, "target_b": target_b})
    return items


def normalize(items):
    valid = [s["x"].reshape(-1, 10) for s in items if s["x"].size]
    all_x = np.concatenate(valid, 0)
    mean, std = all_x.mean(0), all_x.std(0)
    std[std < 1e-8] = 1.
    for s in items:
        if s["x"].size: s["x"] = np.clip((s["x"] - mean) / std, -8, 8).astype(np.float32)
    return mean, std


def scene_loss(model, s, device):
    x = torch.as_tensor(s["x"], device=device)
    n, m, _ = x.shape
    if not n or not m:
        return None
    links, ua, ub = model(x)
    ta = torch.as_tensor(s["target_a"], device=device)
    tb = torch.as_tensor(s["target_b"], device=device)
    row_logits = torch.cat((links, ua[:, None]), dim=1)
    col_logits = torch.cat((links.transpose(0, 1), ub[:, None]), dim=1)
    return (F.cross_entropy(row_logits, ta) + F.cross_entropy(col_logits, tb)) / 2


def train_model(items, epochs, seed, device="cpu"):
    torch.manual_seed(seed)
    model = SceneNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = random.Random(seed)
    batch_size = 16
    for epoch in range(1, epochs + 1):
        order = list(range(len(items))); rng.shuffle(order)
        model.train(); losses = []
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            rows = [scene_loss(model, items[i], device) for i in batch]
            rows = [v for v in rows if v is not None]
            if not rows: continue
            loss = torch.stack(rows).mean()
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step(); losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"epoch {epoch}/{epochs}: loss={np.mean(losses):.4f}", flush=True)
    return model


def edge_utilities(model, items, device="cpu"):
    model.eval(); output = {}
    with torch.no_grad():
        for s in items:
            if not s["x"].size:
                output[s["pid"]] = np.empty((len(s["a"]), len(s["b"])), dtype=np.float64)
                continue
            x = torch.as_tensor(s["x"], device=device)
            links, ua, ub = model(x)
            utility = links - ua[:, None] - ub[None, :]
            output[s["pid"]] = utility.cpu().numpy().astype(np.float64)
    return output


def assemble(items, utilities, threshold):
    out = {}
    for s in items:
        a, b = s["a"], s["b"]
        matrix = utilities[s["pid"]]
        matched = pair.partial_assignment(matrix, threshold) if matrix.size else []
        used_a = {i for i, _ in matched}; used_b = {j for _, j in matched}
        records = [{"bbox_a": a[i], "bbox_b": b[j]} for i, j in matched]
        records.extend({"bbox_a": box, "bbox_b": None} for i, box in enumerate(a) if i not in used_a)
        records.extend({"bbox_a": None, "bbox_b": box} for j, box in enumerate(b) if j not in used_b)
        out[s["pid"]] = records
    return out


def calibrate_threshold(items, utilities, gt, manifest):
    sys.path.insert(0, str(ROOT / "tools"))
    from evaluate import evaluate
    ids = [s["pid"] for s in items]
    held_gt = {pid: gt[pid] for pid in ids}
    held_manifest = {"bbox_type": manifest["bbox_type"], "pairs": {pid: manifest["pairs"][pid] for pid in ids}}
    best = None
    for threshold in np.arange(-8., 8.01, .1):
        prediction = assemble(items, utilities, float(threshold))
        metrics = evaluate(prediction, held_gt, held_manifest)
        key = (metrics["association"]["f1"], -metrics["counting"]["mae"], metrics["counting"]["exact_count_accuracy"])
        if best is None or key > best[0]: best = key, float(threshold), metrics
    return best[1], best[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", required=True)
    p.add_argument("--weights-b", default=None, help="optional camera-B detector checkpoint")
    p.add_argument("--conf", type=float, default=.45)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--calibration-fraction", type=float, default=.2)
    p.add_argument("--out", default=str(BASE / "scene_matcher_960.pt"))
    p.add_argument("--val-out", default=str(ROOT / "val_predictions_scene_matcher960.json"))
    p.add_argument("--predict-test", action="store_true", help="run the saved matcher on test and write a separate predictions file")
    p.add_argument("--checkpoint", default=str(BASE / "scene_matcher_960.pt"))
    p.add_argument("--prediction-out", default=str(ROOT / "predictions_scene_matcher960.json"))
    args = p.parse_args()
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    device = "cpu"
    if args.predict_test:
        bundle = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model = SceneNet().to(device); model.load_state_dict(bundle["state_dict"]); model.eval()
        dino, _ = pipeline.load_dino_model()
        weights_b = args.weights_b or bundle.get("weights_b")
        test_predictions, _, _, _ = pair.load_split("test", args.weights, bundle["conf"],
            bundle["imgsz"], dino, device, dino_aspect="stretch", weights_b=weights_b)
        test_items = scene_items(test_predictions)
        for s in test_items:
            if s["x"].size: s["x"] = np.clip((s["x"] - bundle["mean"]) / bundle["std"], -8, 8).astype(np.float32)
        utilities = edge_utilities(model, test_items, device)
        out = Path(args.prediction_out)
        out.write_text(json.dumps(assemble(test_items, utilities, bundle["threshold"]), indent=1) + "\n", encoding="utf-8")
        print(f"Wrote {len(test_items)} scene predictions to {out} using threshold {bundle['threshold']:.2f}")
        return
    dino, _ = pipeline.load_dino_model()
    x, y, groups, predictions, gt, manifest, _ = pair.load_split("train", args.weights, args.conf,
        args.imgsz, dino, device, dino_aspect="stretch", weights_b=args.weights_b)
    train_scenes = scene_items(predictions, gt)
    rng = np.random.default_rng(20261019)
    order = rng.permutation(len(train_scenes))
    n_cal = max(1, int(len(order) * args.calibration_fraction))
    cal_idx = set(order[:n_cal].tolist())
    fit_items = [s for i, s in enumerate(train_scenes) if i not in cal_idx]
    cal_items = [s for i, s in enumerate(train_scenes) if i in cal_idx]
    mean, std = normalize(fit_items)
    for s in cal_items:
        if s["x"].size: s["x"] = np.clip((s["x"] - mean) / std, -8, 8).astype(np.float32)
    print(f"threshold calibration: fit={len(fit_items)} scenes, held out={len(cal_items)} scenes", flush=True)
    calibration_model = train_model(fit_items, args.epochs, 20261019, device)
    cal_util = edge_utilities(calibration_model, cal_items, device)
    threshold, cal_metrics = calibrate_threshold(cal_items, cal_util, gt, manifest)
    print(f"held-out training threshold={threshold:.2f}; association F1={cal_metrics['association']['f1']:.5f}", flush=True)

    # Keep the same model and scaler used to calibrate the threshold. Refitting on
    # calibration scenes changes the score scale and invalidates the selected threshold.
    final_model = calibration_model
    bundle = {"state_dict": final_model.cpu().state_dict(), "mean": mean,
              "std": std, "threshold": threshold, "epochs": args.epochs,
              "imgsz": args.imgsz, "conf": args.conf,
              "weights": args.weights, "weights_b": args.weights_b,
              "calibration_fraction": args.calibration_fraction,
              "training_scene_count": len(fit_items),
              "calibration_metrics": cal_metrics, "architecture": "row-column-context-dustbin-v1"}
    torch.save(bundle, args.out)
    print(f"saved {args.out}", flush=True)

    dino, _ = pipeline.load_dino_model()
    val_predictions, val_gt, val_manifest, _ = pair.load_split("val", args.weights, args.conf,
        args.imgsz, dino, device, dino_aspect="stretch", weights_b=args.weights_b)
    val_items = scene_items(val_predictions)
    for s in val_items:
        if s["x"].size: s["x"] = np.clip((s["x"] - mean) / std, -8, 8).astype(np.float32)
    val_util = edge_utilities(final_model, val_items, device)
    val_out = Path(args.val_out)
    val_out.write_text(json.dumps(assemble(val_items, val_util, threshold), indent=1) + "\n", encoding="utf-8")
    sys.path.insert(0, str(ROOT / "tools"))
    from evaluate import evaluate
    metrics = evaluate(json.loads(val_out.read_text(encoding="utf-8")), val_gt, val_manifest)
    val_metrics = val_out.with_name(val_out.stem + "_metrics.json")
    val_metrics.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"threshold": threshold, "detection_f1": metrics["detection"]["f1"],
        "association_f1": metrics["association"]["f1"],
        "given_detection_f1": metrics["association"]["given_detection"]["f1"],
        "count_mae": metrics["counting"]["mae"],
        "exact_count_accuracy": metrics["counting"]["exact_count_accuracy"],
        "val_predictions": str(val_out), "val_metrics": str(val_metrics)}, indent=2))


if __name__ == "__main__": main()
