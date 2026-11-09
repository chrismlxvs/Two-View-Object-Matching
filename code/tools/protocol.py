"""JSON protocol shared by the student format checker, the evaluator and the dataset builder.

Formats (all coordinates are pixel xyxy in the original image, continuous edges, no +1):

manifest.json   {"bbox_type": "amodal", "pairs": {pair_id: {"a": {"image", "width", "height"}, "b": {...}}}}
gt.json         {pair_id: [{"bbox_a": [x1,y1,x2,y2] | null, "bbox_b": [...] | null}, ...]}
predictions.json  same format as gt.json

One record = one physical object. Both boxes present = the student claims they are the same object.
"""
import json
import math
from pathlib import Path

BBOX_TYPE = "amodal"
VIEWS = ("a", "b")


def load_json(path):
    def pairs_hook(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"Non-finite JSON number: {value}")

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs_hook, parse_constant=reject_constant)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1, allow_nan=False) + "\n", encoding="utf-8")


def number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_manifest(manifest):
    """Return {pair_id: {"a": view, "b": view}}."""
    require(isinstance(manifest, dict), "manifest: must be an object")
    require(manifest.get("bbox_type") == BBOX_TYPE, f"manifest: bbox_type must be {BBOX_TYPE!r}")
    pairs = manifest.get("pairs")
    require(isinstance(pairs, dict) and pairs, "manifest: pairs must be a nonempty object")
    for pid, views in pairs.items():
        require(isinstance(pid, str) and pid, "manifest: invalid pair_id")
        require(isinstance(views, dict) and set(views) == set(VIEWS), f"manifest {pid}: views must be a and b")
        for view in views.values():
            require(isinstance(view, dict) and set(view) == {"image", "width", "height"},
                    f"manifest {pid}: view must have image, width, height")
            require(isinstance(view["image"], str) and view["image"], f"manifest {pid}: invalid image path")
            for dimension in ("width", "height"):
                require(type(view[dimension]) is int and view[dimension] > 0, f"manifest {pid}: invalid image size")
    return pairs


def _check_record(record, views, where):
    fields = {"bbox_a", "bbox_b"}
    require(isinstance(record, dict), f"{where}: must be an object")
    require(set(record) == fields, f"{where}: expected fields {sorted(fields)}, got {sorted(record)}")
    require(record["bbox_a"] is not None or record["bbox_b"] is not None, f"{where}: both bboxes are null")
    for view in VIEWS:
        bbox = record[f"bbox_{view}"]
        if bbox is None:
            continue
        require(isinstance(bbox, list) and len(bbox) == 4 and all(number(x) for x in bbox),
                f"{where}.bbox_{view}: expected four finite numbers [xmin, ymin, xmax, ymax]")
        x1, y1, x2, y2 = bbox
        size = views[view]
        require(0 <= x1 < x2 <= size["width"] and 0 <= y1 < y2 <= size["height"],
                f"{where}.bbox_{view}: invalid xyxy order or outside the {size['width']}x{size['height']} image")


def validate_document(document, manifest):
    """Return a list of error strings (empty = valid). Predictions and GT share one format; no GT is needed."""
    try:
        pairs = validate_manifest(manifest)
        require(isinstance(document, dict), "document: top level must be an object {pair_id: [records]}")
        errors = []
        for pid, records in document.items():
            if pid not in pairs:
                errors.append(f"unknown pair_id {pid}")
                continue
            if not isinstance(records, list):
                errors.append(f"{pid}: must be a list of records")
                continue
            for index, record in enumerate(records):
                try:
                    _check_record(record, pairs[pid], f"{pid}[{index}]")
                except ValueError as error:
                    errors.append(str(error))
        missing = sorted(set(pairs) - set(document))
        if missing:
            shown = ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
            errors.append(f"missing {len(missing)} pair_id(s): {shown}")
        return errors
    except ValueError as error:
        return [str(error)]
