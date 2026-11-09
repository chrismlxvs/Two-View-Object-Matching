"""Train and evaluate a lightweight cross-view pair scorer using frozen YOLO detections."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import pipeline

ROOT = pipeline.ROOT
BASE = pipeline.BASE


def iou(a, b):
    inter = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    area_a = max(0., a[2]-a[0]) * max(0., a[3]-a[1])
    area_b = max(0., b[2]-b[0]) * max(0., b[3]-b[1])
    return inter / (area_a + area_b - inter) if area_a + area_b > inter else 0.


def cosine(a, b):
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / den) if den else 0.


def feature_rows(boxes_a, boxes_b, scores_a, scores_b, feats_a, feats_b, cams):
    diag = math.sqrt(512**2 + 512**2)
    rows = []
    for i, a in enumerate(boxes_a):
        pa = pipeline.project_table_prior(a, cams["a"], cams["b"])
        for j, b in enumerate(boxes_b):
            pb = pipeline.project_table_prior(b, cams["b"], cams["a"])
            ca = np.array([(a[0]+a[2])/2, a[3]])
            cb = np.array([(b[0]+b[2])/2, b[3]])
            ga = min(float(np.linalg.norm(pa-cb))/diag, 1.) if pa is not None else .5
            gb = min(float(np.linalg.norm(pb-ca))/diag, 1.) if pb is not None else .5
            ha, wa = max(1e-4, a[3]-a[1]), max(1e-4, a[2]-a[0])
            hb, wb = max(1e-4, b[3]-b[1]), max(1e-4, b[2]-b[0])
            rows.append([ga, gb, abs(ga-gb), cosine(feats_a[i]["hist"], feats_b[j]["hist"]),
                         cosine(feats_a[i]["dino"], feats_b[j]["dino"]),
                         float(scores_a[i]), float(scores_b[j]),
                         abs(math.log(wa/wb)), abs(math.log(ha/hb)),
                         abs(math.log((wa*ha)/(wb*hb)))])
    return np.asarray(rows, dtype=np.float64).reshape((-1, 10))


def load_split(split, weights, conf, imgsz, dino_model, device, cache=True, dino_aspect="stretch", weights_b=None, max_pairs=0):
    import cv2
    from ultralytics import YOLO
    folder = ROOT / split
    manifest = pipeline.read_json(folder / "manifest.json")
    gt = pipeline.read_json(folder / "gt.json") if (folder / "gt.json").exists() else {}
    cameras = pipeline.read_json(folder / "cameras.json")
    detector_paths = {"a": Path(weights), "b": Path(weights_b) if weights_b else Path(weights)}
    detectors = {"a": YOLO(str(detector_paths["a"]))}
    if detector_paths["b"] == detector_paths["a"]:
        detectors["b"] = detectors["a"]
    else:
        detectors["b"] = YOLO(str(detector_paths["b"]))
    detector_caches = {}
    for view, weight_path in detector_paths.items():
        weight_tag = f"{weight_path.stem}_{weight_path.stat().st_size}_{weight_path.stat().st_mtime_ns}"
        detector_caches[view] = BASE / "cache" / "yolo" / weight_tag / split / f"conf{conf:g}_imgsz{imgsz}"
    all_x, all_y, all_groups = [], [], []
    predictions = {}
    pair_items = list(manifest["pairs"].items())
    if max_pairs > 0 and max_pairs < len(pair_items):
        import random
        selected_ids = set(random.Random(42).sample([pid for pid, _ in pair_items], max_pairs))
        pair_items = [(pid, views) for pid, views in pair_items if pid in selected_ids]
    scene_indices = {pid: i for i, pid in enumerate(manifest["pairs"])}
    for n, (pid, views) in enumerate(pair_items, 1):
        imgs, boxes, scores, feats = [], [], [], []
        for v in ("a", "b"):
            image = cv2.imread(str(folder / views[v]["image"]))
            imgs.append(image)
            detection_cache = detector_caches[v] / f"{pid}_{v}.json"
            if cache and detection_cache.exists():
                cached_detection = json.loads(detection_cache.read_text(encoding="utf-8"))
                b, s = cached_detection["boxes"], cached_detection["scores"]
            else:
                result = detectors[v].predict(image, imgsz=imgsz, conf=conf, iou=.6, max_det=40, verbose=False)[0]
                b = result.boxes.xyxy.cpu().numpy().astype(float).tolist()
                s = result.boxes.conf.cpu().numpy().astype(float).tolist()
                if cache:
                    detection_cache.parent.mkdir(parents=True, exist_ok=True)
                    detection_cache.write_text(json.dumps({"boxes": b, "scores": s}), encoding="utf-8")
            boxes.append(b); scores.append(s)
            cache_name = "dinov2_vits14" if dino_aspect == "stretch" else "dinov2_vits14_pad"
            cache_path = BASE / "cache" / cache_name / split / f"{pid}_{v}.npz"
            d = pipeline.dino_features(image, b, dino_model, device, cache_path,
                                       preserve_aspect=(dino_aspect == "pad"))
            h = [pipeline.hist_feature(image, box) for box in b]
            feats.append([{"hist": h[k], "dino": d[k]} for k in range(len(b))])

        x = feature_rows(boxes[0], boxes[1], scores[0], scores[1], feats[0], feats[1], cameras[pid])
        predictions[pid] = {"boxes": boxes, "scores": scores, "features": feats, "x": x}
        if split == "train" and x.size:
            # Map each prediction to its GT object at the course's IoU=0.5 threshold.
            maps = []
            for vi, v in enumerate(("a", "b")):
                gts = [r[f"bbox_{v}"] for r in gt[pid] if r[f"bbox_{v}"] is not None]
                pred_to_gt = {}
                candidates = sorted(((iou(box, gb), pi, gi) for pi, box in enumerate(boxes[vi])
                                     for gi, gb in enumerate(gts) if iou(box, gb) >= .5), reverse=True)
                used_p, used_g = set(), set()
                for ov, pi, gi in candidates:
                    if pi not in used_p and gi not in used_g:
                        pred_to_gt[pi] = gi; used_p.add(pi); used_g.add(gi)
                maps.append(pred_to_gt)
            y = np.asarray([int(i in maps[0] and j in maps[1] and maps[0][i] == maps[1][j])
                            for i in range(len(boxes[0])) for j in range(len(boxes[1]))], dtype=np.int8)
            all_x.append(x); all_y.append(y)
            all_groups.append(np.full(len(y), scene_indices[pid], dtype=np.int32))
        if n % 50 == 0:
            print(f"{split}: processed {n}/{len(pair_items)} pairs", flush=True)
    if split == "train":
        return (np.vstack(all_x), np.concatenate(all_y), np.concatenate(all_groups),
                predictions, gt, manifest, cameras)
    return predictions, gt, manifest, cameras


def fit_logistic(x, y, steps=2500, lr=.08, l2=.02):
    # Balance classes, then optimize a small regularized logistic scorer with NumPy.
    mean, std = x.mean(0), x.std(0)
    std[std < 1e-8] = 1.
    z = np.clip((x-mean)/std, -8, 8)
    pos, neg = max(1, int(y.sum())), max(1, int((1-y).sum()))
    weights = np.where(y == 1, .5/pos, .5/neg)
    coef = np.zeros(z.shape[1]); bias = 0.
    for step in range(steps):
        logits = np.clip(z @ coef + bias, -30, 30)
        p = 1/(1+np.exp(-logits))
        err = (p-y)*weights
        grad = z.T @ err + l2*coef
        gb = err.sum()
        rate = lr / math.sqrt(1 + step/250)
        coef -= rate*grad; bias -= rate*gb
    return {"mean": mean.tolist(), "std": std.tolist(), "coef": coef.tolist(), "bias": float(bias),
            "features": ["geo_a_to_b", "geo_b_to_a", "geo_disagreement", "hist_cosine", "dinov2_cosine",
                         "confidence_a", "confidence_b", "log_width_ratio", "log_height_ratio", "log_area_ratio"]}


def expand_pair_features(x):
    """Add squared and selected geometry/appearance/confidence interactions."""
    x = np.asarray(x, dtype=np.float64).reshape((-1, 10))
    g_a, g_b, g_diff, hist, dino, conf_a, conf_b = [x[:, i] for i in range(7)]
    extra = np.column_stack((
        x * x,
        g_a * hist, g_b * hist, g_a * dino, g_b * dino,
        g_a * conf_a, g_b * conf_b,
        g_diff * hist, g_diff * dino,
        hist * dino,
        hist * conf_a, hist * conf_b, dino * conf_a, dino * conf_b,
    ))
    return np.concatenate((x, extra), axis=1)


def fit_logistic_hard(x, y, groups, hard_per_scene=40, easy_per_scene=20):
    """Mine high-scoring false pairs per scene, retain some easy negatives, then refit."""
    first = fit_logistic(x, y)
    scores = logits(first, x)
    keep = np.flatnonzero(y == 1).tolist()
    rng = np.random.default_rng(42)
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        negatives = indices[y[indices] == 0]
        positives = int(np.sum(y[indices] == 1))
        hard_n = min(len(negatives), max(hard_per_scene, 4 * positives))
        if hard_n:
            hard = negatives[np.argsort(scores[negatives])[-hard_n:]]
            keep.extend(hard.tolist())
            remaining = np.setdiff1d(negatives, hard, assume_unique=False)
            easy_n = min(len(remaining), easy_per_scene)
            if easy_n:
                keep.extend(rng.choice(remaining, size=easy_n, replace=False).tolist())
    keep = np.asarray(sorted(set(keep)), dtype=np.int64)
    model = fit_logistic(x[keep], y[keep])
    model["hard_negative_training"] = True
    model["training_rows"] = int(len(keep))
    model["training_positive_rows"] = int(np.sum(y[keep] == 1))
    model["training_negative_rows"] = int(np.sum(y[keep] == 0))
    model["hard_negatives_per_scene_minimum"] = int(hard_per_scene)
    return model


def cross_validate_recovery(x, y, groups, predictions, gt, manifest, cameras,
                            folds=5, base_conf=.45, radius_px=85):
    """Choose a recovery threshold from scene-held-out matcher predictions."""
    import sys
    sys.path.insert(0, str(ROOT / "tools"))
    from evaluate import evaluate
    unique = np.unique(groups)
    rng = np.random.default_rng(20261019)
    unique = rng.permutation(unique)
    fold_ids = np.array_split(unique, min(folds, len(unique)))
    oof_predictions = {}
    oof_gt = {}
    oof_pairs = {}
    for fold_number, heldout in enumerate(fold_ids, 1):
        is_val = np.isin(groups, heldout)
        train_rows = ~is_val
        print(f"matcher cross-validation fold {fold_number}/{len(fold_ids)}: training on {int(train_rows.sum())} pair rows", flush=True)
        fold_model = fit_logistic_hard(x[train_rows], y[train_rows], groups[train_rows])
        for group in heldout:
            pid = list(manifest["pairs"])[int(group)]
            oof_predictions[pid] = predictions[pid]
            oof_gt[pid] = gt[pid]
            oof_pairs[pid] = manifest["pairs"][pid]
            # Scores depend only on this model's standardized coefficients.
            oof_predictions[pid] = dict(predictions[pid])
            oof_predictions[pid]["pair_model"] = fold_model
    # Each held-out scene has its own fold model; assemble them independently per threshold.
    best = None
    for threshold in np.arange(-20, 10.01, .25):
        merged = {}
        for pid, row in oof_predictions.items():
            pred = assemble_camera_recovery({pid: row}, row["pair_model"], float(threshold), base_conf, radius_px)
            merged.update(pred)
        m = evaluate(merged, oof_gt, {"bbox_type": manifest["bbox_type"], "pairs": oof_pairs})
        key = (m["association"]["f1"], -m["counting"]["mae"], m["counting"]["exact_count_accuracy"])
        if best is None or key > best[0]: best = (key, float(threshold), m)
    return {"matcher_oof_folds": int(len(fold_ids)), "matcher_oof_threshold": best[1],
            "matcher_oof_metrics": best[2],
            "matcher_oof_note": "Detector was trained on all train scenes; only pair matcher was scene-held-out."}


def cross_validate_association(x, y, groups, predictions, gt, manifest, folds=5):
    """Calibrate the pair-acceptance threshold on out-of-fold training scenes."""
    import sys
    sys.path.insert(0, str(ROOT / "tools"))
    from evaluate import evaluate
    unique = np.unique(groups)
    fold_ids = np.array_split(np.random.default_rng(20261019).permutation(unique), min(folds, len(unique)))
    oof = {}
    ids = list(manifest["pairs"])
    for fold_number, heldout in enumerate(fold_ids, 1):
        held = np.isin(groups, heldout)
        model = fit_logistic_hard(x[~held], y[~held], groups[~held])
        print(f"association calibration fold {fold_number}/{len(fold_ids)} complete", flush=True)
        for group in heldout:
            pid = ids[int(group)]
            oof[pid] = dict(predictions[pid])
            oof[pid]["pair_model"] = model
    held_gt = {pid: gt[pid] for pid in oof}
    held_manifest = {"bbox_type": manifest["bbox_type"], "pairs": {pid: manifest["pairs"][pid] for pid in oof}}
    thresholds = np.arange(-10., 10.01, .5)
    best = None
    for threshold in thresholds:
        assembled = {pid: assemble({pid: row}, row["pair_model"], float(threshold))[pid] for pid, row in oof.items()}
        metrics = evaluate(assembled, held_gt, held_manifest)
        key = (metrics["association"]["f1"], -metrics["counting"]["mae"], metrics["counting"]["exact_count_accuracy"])
        if best is None or key > best[0]: best = (key, float(threshold), metrics)
    center = best[1]
    for threshold in np.arange(center - .45, center + .451, .05):
        assembled = {pid: assemble({pid: row}, row["pair_model"], float(threshold))[pid] for pid, row in oof.items()}
        metrics = evaluate(assembled, held_gt, held_manifest)
        key = (metrics["association"]["f1"], -metrics["counting"]["mae"], metrics["counting"]["exact_count_accuracy"])
        if key > best[0]: best = (key, float(threshold), metrics)
    return {"association_oof_folds": len(fold_ids), "association_oof_threshold": best[1],
            "association_oof_metrics": best[2],
            "calibration_note": "Threshold selected on out-of-fold training scenes; detector was trained on all training scenes."}


def logits(model, x):
    if len(x) == 0: return np.empty(0)
    if model.get("feature_transform") == "quadratic_interactions_v1":
        x = expand_pair_features(x)
    z = np.clip((x-np.array(model["mean"]))/np.array(model["std"]), -8, 8)
    return z @ np.array(model["coef"]) + model["bias"]


def partial_assignment(scores, threshold, allowed=None):
    """Maximum-weight bipartite matching with an explicit unmatched option on both sides."""
    from scipy.optimize import linear_sum_assignment
    n, m = scores.shape
    if n == 0 or m == 0:
        return []
    # Each real edge competes against a zero-utility dummy assignment. This prevents
    # a rejected edge from occupying a row/column and blocking a better accepted match.
    utility = np.zeros((n + m, n + m), dtype=np.float64)
    real = scores - float(threshold)
    if allowed is not None:
        real = np.where(allowed, real, -1e6)
    utility[:n, :m] = real
    rows, cols = linear_sum_assignment(-utility)
    return [(int(i), int(j)) for i, j in zip(rows, cols)
            if i < n and j < m and real[i, j] > 0]


def assemble(predictions, model, threshold):
    out = {}
    for pid, data in predictions.items():
        a, b = data["boxes"]
        matrix = logits(model, data["x"]).reshape((len(a), len(b))) if len(a)*len(b) else np.empty((len(a),len(b)))
        records, used_a, used_b = [], set(), set()
        if matrix.size:
            for i, j in partial_assignment(matrix, threshold):
                records.append({"bbox_a":a[i],"bbox_b":b[j]}); used_a.add(i); used_b.add(j)
        records.extend({"bbox_a":box,"bbox_b":None} for i,box in enumerate(a) if i not in used_a)
        records.extend({"bbox_a":None,"bbox_b":box} for j,box in enumerate(b) if j not in used_b)
        out[pid]=records
    return out


def calibrated_radius(box, source_view, calibration, fallback):
    if not calibration:
        return float(fallback)
    view = calibration["views"][source_view]
    grid = int(calibration["grid_size"])
    u, v = (float(box[0]) + float(box[2])) / 2, float(box[3])
    col = min(grid - 1, max(0, int(u / 512 * grid)))
    row = min(grid - 1, max(0, int(v / 512 * grid)))
    return float(view["cells"][row][col]["radius_px"])


def assemble_camera_recovery(predictions, model, threshold, base_conf=.45, radius_px=85., radius_calibration=None):
    """Match normal-confidence boxes first, then recover a lower-confidence counterpart near its camera projection.

    Low-confidence boxes are emitted only when linked to a normal-confidence detection. A learned
    score threshold and a train-derived camera reprojection radius guard each recovery.
    """
    out = {}
    diagonal = math.sqrt(512**2 + 512**2)
    for pid, data in predictions.items():
        boxes_a, boxes_b = data["boxes"]
        scores_a, scores_b = [np.asarray(s) for s in data["scores"]]
        na, nb = len(boxes_a), len(boxes_b)
        all_logits = logits(model, data["x"]).reshape((na, nb)) if na*nb else np.empty((na, nb))
        high_a = set(np.flatnonzero(scores_a >= base_conf).tolist())
        high_b = set(np.flatnonzero(scores_b >= base_conf).tolist())
        used_a, used_b = set(), set()
        pairs = []

        # Preserve standard high-confidence links first.
        ia, ib = sorted(high_a), sorted(high_b)
        if ia and ib:
            for r, c in partial_assignment(all_logits[np.ix_(ia, ib)], threshold):
                i, j = ia[r], ib[c]
                pairs.append((i, j)); used_a.add(i); used_b.add(j)

        # A high-confidence A detection may rescue a lower-confidence B detection near its
        # calibrated ground-plane projection. The second pass is symmetric.
        high_a_left = sorted(high_a - used_a)
        low_b = sorted(np.flatnonzero((scores_b > 0) & (scores_b < base_conf)).tolist())
        if high_a_left and low_b:
            sub = all_logits[np.ix_(high_a_left, low_b)].copy()
            geo = data["x"].reshape((na, nb, -1))[np.ix_(high_a_left, low_b, [0])][:, :, 0]
            limits = np.asarray([calibrated_radius(boxes_a[i], "a", radius_calibration, radius_px)
                                 for i in high_a_left], dtype=float)[:, None]
            valid = geo * diagonal <= limits
            for r, c in partial_assignment(sub, threshold, valid):
                i, j = high_a_left[r], low_b[c]
                if valid[r, c]:
                    pairs.append((i, j)); used_a.add(i); used_b.add(j)

        low_a = sorted(np.flatnonzero((scores_a > 0) & (scores_a < base_conf)).tolist())
        high_b_left = sorted(high_b - used_b)
        if low_a and high_b_left:
            sub = all_logits[np.ix_(low_a, high_b_left)].copy()
            geo = data["x"].reshape((na, nb, -1))[np.ix_(low_a, high_b_left, [1])][:, :, 0]
            limits = np.asarray([calibrated_radius(boxes_b[j], "b", radius_calibration, radius_px)
                                 for j in high_b_left], dtype=float)[None, :]
            valid = geo * diagonal <= limits
            for r, c in partial_assignment(sub, threshold, valid):
                i, j = low_a[r], high_b_left[c]
                if valid[r, c]:
                    pairs.append((i, j)); used_a.add(i); used_b.add(j)

        records = [{"bbox_a": boxes_a[i], "bbox_b": boxes_b[j]} for i, j in pairs]
        records.extend({"bbox_a": boxes_a[i], "bbox_b": None} for i in sorted(high_a - used_a))
        records.extend({"bbox_a": None, "bbox_b": boxes_b[j]} for j in sorted(high_b - used_b))
        out[pid] = records
    return out


def evaluate_sweep(args):
    import sys
    sys.path.insert(0, str(ROOT / "tools"))
    from evaluate import evaluate
    dino_model, device = pipeline.load_dino_model()
    predictions, gt, manifest, _ = load_split("val", args.weights, args.conf, args.imgsz, dino_model, device,
                                               dino_aspect=args.dino_aspect, weights_b=args.weights_b)
    model = json.loads(Path(args.matcher).read_text(encoding="utf-8"))
    best = None
    thresholds = [float(args.threshold)] if args.threshold is not None else np.arange(-20, 10.01, .25)
    for threshold in thresholds:
        pred = assemble(predictions, model, float(threshold))
        m = evaluate(pred, gt, manifest)
        score = (m["association"]["f1"], -m["counting"]["mae"], m["counting"]["exact_count_accuracy"])
        if best is None or score > best[0]: best = (score, float(threshold), m, pred)
    score, threshold, metrics, pred = best
    out = Path(args.out); out.write_text(json.dumps(pred, indent=1)+"\n", encoding="utf-8")
    met = out.with_name(out.stem+"_metrics.json"); met.write_text(json.dumps(metrics,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"threshold":threshold,"detection_f1":metrics["detection"]["f1"],
                      "association_f1":metrics["association"]["f1"],
                      "given_detection_f1":metrics["association"]["given_detection"]["f1"],
                      "count_mae":metrics["counting"]["mae"],
                      "exact_count_accuracy":metrics["counting"]["exact_count_accuracy"],
                      "predictions":str(out),"metrics":str(met)},indent=2))


def predict_test(args):
    dino_model, device = pipeline.load_dino_model()
    predictions, _, _, _ = load_split("test", args.weights, args.conf, args.imgsz, dino_model, device,
                                      dino_aspect=args.dino_aspect, weights_b=args.weights_b)
    model = json.loads(Path(args.matcher).read_text(encoding="utf-8"))
    pred = assemble(predictions, model, args.threshold)
    out = Path(args.out); out.write_text(json.dumps(pred, indent=1)+"\n", encoding="utf-8")
    print(f"Wrote {len(pred)} test scene predictions to {out}; scorer threshold {args.threshold:g}")


def evaluate_recovery(args):
    import sys
    sys.path.insert(0, str(ROOT / "tools"))
    from evaluate import evaluate
    dino_model, device = pipeline.load_dino_model()
    # Low detector cutoff supplies candidate boxes; only boxes linked to a >=base_conf box survive.
    predictions, gt, manifest, _ = load_split("val", args.weights, args.candidate_conf,
                                               args.imgsz, dino_model, device, dino_aspect=args.dino_aspect,
                                               weights_b=args.weights_b)
    model = json.loads(Path(args.matcher).read_text(encoding="utf-8"))
    radius_calibration = json.loads(Path(args.radius_calibration).read_text(encoding="utf-8")) if args.radius_calibration else None
    best = None
    thresholds = [float(args.threshold)] if args.threshold is not None else np.arange(-20, 10.01, .25)
    for threshold in thresholds:
        pred = assemble_camera_recovery(predictions, model, float(threshold), args.base_conf, args.radius_px, radius_calibration)
        metrics = evaluate(pred, gt, manifest)
        score = (metrics["association"]["f1"], -metrics["counting"]["mae"],
                 metrics["counting"]["exact_count_accuracy"])
        if best is None or score > best[0]: best = (score, float(threshold), metrics, pred)
    _, threshold, metrics, pred = best
    out = Path(args.out); out.write_text(json.dumps(pred, indent=1)+"\n", encoding="utf-8")
    met = out.with_name(out.stem+"_metrics.json")
    met.write_text(json.dumps(metrics, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"candidate_conf":args.candidate_conf,"base_conf":args.base_conf,
                      "radius_px":args.radius_px,"match_threshold":threshold,
                      "detection_f1":metrics["detection"]["f1"],
                      "association_f1":metrics["association"]["f1"],
                      "given_detection_f1":metrics["association"]["given_detection"]["f1"],
                      "count_mae":metrics["counting"]["mae"],
                      "exact_count_accuracy":metrics["counting"]["exact_count_accuracy"],
                      "predictions":str(out),"metrics":str(met)}, indent=2))


def predict_test_recovery(args):
    dino_model, device = pipeline.load_dino_model()
    predictions, _, _, _ = load_split("test", args.weights, args.candidate_conf, args.imgsz, dino_model, device,
                                      dino_aspect=args.dino_aspect, weights_b=args.weights_b)
    model = json.loads(Path(args.matcher).read_text(encoding="utf-8"))
    radius_calibration = json.loads(Path(args.radius_calibration).read_text(encoding="utf-8")) if args.radius_calibration else None
    pred = assemble_camera_recovery(predictions, model, args.threshold, args.base_conf, args.radius_px, radius_calibration)
    out=Path(args.out); out.write_text(json.dumps(pred, indent=1)+"\n", encoding="utf-8")
    print(f"Wrote {len(pred)} recovered test scene predictions to {out}")


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest="cmd",required=True)
    t=sub.add_parser("train"); t.add_argument("--weights",required=True); t.add_argument("--weights-b",default=None); t.add_argument("--max-pairs",type=int,default=0,help="optional deterministic scene sample for fast experiments"); t.add_argument("--conf",type=float,default=.45); t.add_argument("--imgsz",type=int,default=640); t.add_argument("--dino-aspect",choices=["stretch","pad"],default="stretch"); t.add_argument("--quadratic",action="store_true",help="add geometry/appearance interaction features"); t.add_argument("--out",default=str(BASE/"pair_matcher.json")); t.add_argument("--hard-negative",action="store_true"); t.add_argument("--hard-per-scene",type=int,default=40); t.add_argument("--easy-per-scene",type=int,default=20); t.add_argument("--crossval-folds",type=int,default=0); t.add_argument("--calibrate-threshold-folds",type=int,default=0,help="select association threshold on scene-held-out training folds"); t.add_argument("--base-conf",type=float,default=.45); t.add_argument("--radius-px",type=float,default=85)
    e=sub.add_parser("evaluate"); e.add_argument("--weights",required=True); e.add_argument("--weights-b",default=None); e.add_argument("--matcher",default=str(BASE/"pair_matcher.json")); e.add_argument("--conf",type=float,default=.45); e.add_argument("--imgsz",type=int,default=640); e.add_argument("--dino-aspect",choices=["stretch","pad"],default="stretch"); e.add_argument("--threshold",type=float,default=None); e.add_argument("--out",default=str(ROOT/"val_predictions_pair_scorer.json"))
    q=sub.add_parser("predict-test"); q.add_argument("--weights",required=True); q.add_argument("--weights-b",default=None); q.add_argument("--matcher",default=str(BASE/"pair_matcher.json")); q.add_argument("--conf",type=float,default=.45); q.add_argument("--imgsz",type=int,default=640); q.add_argument("--dino-aspect",choices=["stretch","pad"],default="stretch"); q.add_argument("--threshold",type=float,default=-.7); q.add_argument("--out",default=str(ROOT/"predictions_pair_scorer.json"))
    r=sub.add_parser("evaluate-recovery", help="validate camera-guided low-confidence counterpart recovery")
    r.add_argument("--weights",required=True); r.add_argument("--weights-b",default=None); r.add_argument("--matcher",default=str(BASE/"pair_matcher.json")); r.add_argument("--candidate-conf",type=float,default=.05); r.add_argument("--base-conf",type=float,default=.45); r.add_argument("--radius-px",type=float,default=85); r.add_argument("--radius-calibration",default=None); r.add_argument("--imgsz",type=int,default=640); r.add_argument("--dino-aspect",choices=["stretch","pad"],default="stretch"); r.add_argument("--threshold",type=float,default=None); r.add_argument("--out",default=str(ROOT/"val_predictions_camera_recovery.json"))
    rt=sub.add_parser("predict-test-recovery", help="predict test using validated camera-guided recovery")
    rt.add_argument("--weights",required=True); rt.add_argument("--weights-b",default=None); rt.add_argument("--matcher",default=str(BASE/"pair_matcher.json")); rt.add_argument("--candidate-conf",type=float,default=.05); rt.add_argument("--base-conf",type=float,default=.45); rt.add_argument("--radius-px",type=float,default=85); rt.add_argument("--radius-calibration",default=None); rt.add_argument("--imgsz",type=int,default=640); rt.add_argument("--dino-aspect",choices=["stretch","pad"],default="stretch"); rt.add_argument("--threshold",type=float,default=-1.0); rt.add_argument("--out",default=str(ROOT/"predictions_camera_recovery.json"))
    args=p.parse_args()
    if args.cmd=="train":
        dino,device=pipeline.load_dino_model(); x,y,groups,predictions,gt,manifest,cameras=load_split("train",args.weights,args.conf,args.imgsz,dino,device,dino_aspect=args.dino_aspect,weights_b=args.weights_b,max_pairs=args.max_pairs)
        fit_x=expand_pair_features(x) if args.quadratic else x
        model=(fit_logistic_hard(fit_x,y,groups,args.hard_per_scene,args.easy_per_scene) if args.hard_negative else fit_logistic(fit_x,y))
        model["candidate_rows"]=int(len(y)); model["positive_candidates"]=int(y.sum())
        if not args.hard_negative:
            model["training_rows"]=int(len(y)); model["training_positive_rows"]=int(y.sum())
        if args.quadratic: model["feature_transform"]="quadratic_interactions_v1"
        model["dino_aspect"]=args.dino_aspect
        if args.calibrate_threshold_folds > 1:
            model["association_calibration"] = cross_validate_association(fit_x,y,groups,predictions,gt,manifest,args.calibrate_threshold_folds)
        if args.crossval_folds > 1:
            model["recovery_calibration"]=cross_validate_recovery(x,y,groups,predictions,gt,manifest,cameras,args.crossval_folds,args.base_conf,args.radius_px)
        Path(args.out).write_text(json.dumps(model,indent=2)+"\n",encoding="utf-8")
        print(f"Saved {args.out}; {len(y)} candidates, {int(y.sum())} positive links, {len(y)-int(y.sum())} negatives")
    elif args.cmd=="evaluate": evaluate_sweep(args)
    elif args.cmd=="predict-test": predict_test(args)
    elif args.cmd=="evaluate-recovery": evaluate_recovery(args)
    elif args.cmd=="predict-test-recovery": predict_test_recovery(args)
    else: predict_test(args)

if __name__=="__main__": main()
