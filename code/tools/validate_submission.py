"""Check that predictions.json has the right structure. Uses only the public manifest, never GT.

    python3 validate_submission.py --pred predictions.json --manifest manifest.json

Exit code 0 = format OK, 2 = format errors (listed in the JSON output). A valid format says
nothing about accuracy: an empty list for every pair is valid.
"""
import argparse
import json
from protocol import load_json, save_json, validate_document


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out")
    args = parser.parse_args()
    try:
        errors = validate_document(load_json(args.pred), load_json(args.manifest))
    except (OSError, ValueError) as error:
        errors = [str(error)]
    result = {"valid": not errors, "errors": errors}
    if args.out:
        save_json(args.out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
