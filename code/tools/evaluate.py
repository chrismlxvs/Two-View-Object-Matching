"""Course evaluator: detection P/R/F1@IoU, cross-view association F1 and unique-object counting.

Every submitted record counts, and record order is irrelevant. Usage:

    python3 evaluate.py --pred predictions.json --gt gt.json --manifest manifest.json [--out metrics.json]
"""
import argparse
import json
from protocol import VIEWS, load_json, save_json, validate_document

DEFAULT_IOU = 0.5


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union else 0.0


def prf(tp, fp, fn):
    # Nothing to evaluate is "not applicable", never an automatic 100%.
    if tp + fp + fn == 0:
        return {"tp": 0, "fp": 0, "fn": 0, "precision": None, "recall": None, "f1": None}
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn)}


def collect(document):
    """Flatten {pair_id: [records]} into one box per (pair, view, record index)."""
    boxes = []
    for pid in sorted(document):
        for index, record in enumerate(document[pid]):
            for view in VIEWS:
                bbox = record[f"bbox_{view}"]
                if bbox is not None:
                    boxes.append({"pair_id": pid, "view": view, "index": index, "bbox": bbox})
    return boxes


def match_detections(predictions, ground_truth, threshold):
    """Greedy one-to-one matching per (pair, view): all (pred, GT) pairs with IoU >= threshold are
    taken in order of decreasing IoU. Ties are broken by box coordinates, so the result does not
    depend on record order (except between exactly identical predicted boxes).

    Returns {(pair, view, pred index): gt index}.
    """
    buckets = {}
    for g in ground_truth:
        buckets.setdefault((g["pair_id"], g["view"]), []).append(g)
    candidates = []
    for p in predictions:
        for g in buckets.get((p["pair_id"], p["view"]), []):
            overlap = iou(p["bbox"], g["bbox"])
            if overlap >= threshold:
                candidates.append((-overlap, list(p["bbox"]), list(g["bbox"]), p["index"], g["index"], p, g))
    candidates.sort(key=lambda c: c[:5])
    used_pred, used_gt, matches = set(), set(), {}
    for *_, p, g in candidates:
        pid = (p["pair_id"], p["view"], p["index"])
        gid = (g["pair_id"], g["view"], g["index"])
        if pid in used_pred or gid in used_gt:
            continue
        used_pred.add(pid)
        used_gt.add(gid)
        matches[pid] = g["index"]
    return matches


def evaluate(prediction, gt, manifest, iou_threshold=DEFAULT_IOU):
    for name, document in (("prediction", prediction), ("GT", gt)):
        errors = validate_document(document, manifest)
        if errors:
            raise ValueError(name + ": " + "; ".join(errors))
    pred_boxes, gt_boxes = collect(prediction), collect(gt)
    mapping = match_detections(pred_boxes, gt_boxes, iou_threshold)
    det_tp = len(mapping)
    per_view = {}
    for view in VIEWS:
        hits = sum(key[1] == view for key in mapping)
        total = sum(p["view"] == view for p in pred_boxes)
        per_view[view] = prf(hits, total - hits, sum(g["view"] == view for g in gt_boxes) - hits)

    # Which GT instance did each GT box belong to, and which GT boxes were detected at all.
    detected = {}  # (pair, view, gt index) -> True
    for key, gt_index in mapping.items():
        detected[(key[0], key[1], gt_index)] = True

    assoc_tp = assoc_pred = assoc_gt = 0
    cond_tp = cond_pred = cond_gt = 0  # association quality given that the boxes were detected
    details = []
    for pid in sorted(gt):
        records, truth = prediction[pid], gt[pid]
        pair_pred = pair_tp = pair_cond_pred = pair_cond_tp = 0
        for index, record in enumerate(records):
            if record["bbox_a"] is None or record["bbox_b"] is None:
                continue
            pair_pred += 1
            ga, gb = mapping.get((pid, "a", index)), mapping.get((pid, "b", index))
            correct = ga is not None and ga == gb
            pair_tp += correct
            if ga is not None and gb is not None:
                pair_cond_pred += 1
                pair_cond_tp += correct
        pair_gt = sum(t["bbox_a"] is not None and t["bbox_b"] is not None for t in truth)
        pair_cond_gt = sum(t["bbox_a"] is not None and t["bbox_b"] is not None
                           and (pid, "a", i) in detected and (pid, "b", i) in detected
                           for i, t in enumerate(truth))
        assoc_tp += pair_tp
        assoc_pred += pair_pred
        assoc_gt += pair_gt
        cond_tp += pair_cond_tp
        cond_pred += pair_cond_pred
        cond_gt += pair_cond_gt
        details.append({"pair_id": pid, "gt_count": len(truth), "predicted_count": len(records),
                        "count_error": len(records) - len(truth),
                        "association": prf(pair_tp, pair_pred - pair_tp, pair_gt - pair_tp)})
    errors = [abs(d["count_error"]) for d in details]
    return {
        "evaluation_protocol": "course-dualview-3.0",
        "iou_threshold": iou_threshold,
        "detection": {
            **prf(det_tp, len(pred_boxes) - det_tp, len(gt_boxes) - det_tp),
            "per_view": per_view,
        },
        "association": {
            **prf(assoc_tp, assoc_pred - assoc_tp, assoc_gt - assoc_tp),
            "given_detection": {
                "definition": "precision: two-sided records whose boxes both matched some GT and point to the same object; "
                              "recall: GT objects whose two boxes were both detected and were linked correctly",
                **prf(cond_tp, cond_pred - cond_tp, cond_gt - cond_tp),
            },
        },
        "counting": {
            "mae": sum(errors) / len(details),
            "rmse": (sum(e * e for e in errors) / len(details)) ** 0.5,
            "exact_count_accuracy": sum(e == 0 for e in errors) / len(details),
        },
        "per_pair": details,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred", required=True, help="predictions.json")
    parser.add_argument("--gt", required=True, help="gt.json of the same split")
    parser.add_argument("--manifest", required=True, help="manifest.json of the same split")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU, help=f"IoU threshold (default {DEFAULT_IOU})")
    parser.add_argument("--out", help="also write metrics to this JSON file")
    parser.add_argument("--quiet", action="store_true", help="print only the summary line")
    args = parser.parse_args()
    try:
        result = evaluate(load_json(args.pred), load_json(args.gt), load_json(args.manifest), args.iou)
    except (OSError, ValueError) as error:
        print(json.dumps({"valid": False, "errors": [str(error)]}, ensure_ascii=False, indent=2))
        return 2
    if args.out:
        save_json(args.out, result)
    if args.quiet:
        d, a, c = result["detection"], result["association"], result["counting"]
        print(f"det_F1@{args.iou:g}={d['f1']}  assoc_F1={a['f1']}  count_MAE={c['mae']:.3f}")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
