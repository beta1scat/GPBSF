"""Package GPBSF code and dataset for Kaggle with standard POSIX zip paths."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import zipfile

EXCLUDE_PATTERNS = [
    r"\.git($|[\\/])",
    r"__pycache__",
    r"\.pyc$",
    r"^data($|[\\/])",
    r"^runs($|[\\/])",
    r"kaggle_assets",
    r"\.zip$",
]


def should_exclude(rel_path: str) -> bool:
    for pattern in EXCLUDE_PATTERNS:
        if re.search(pattern, rel_path):
            return True
    return False


def package_code(project_root: Path, output_zip: Path) -> None:
    print(f"[1/2] Packaging code from {project_root} to {output_zip}...")
    count = 0
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(project_root.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(project_root)
            rel_str = str(rel)
            if should_exclude(rel_str):
                continue
            # Force POSIX forward slashes to ensure Kaggle compatibility
            arcname = rel.as_posix()
            zf.write(p, arcname=arcname)
            count += 1
    print(f"Successfully packaged {count} code files into {output_zip}")


def package_dataset(dataset_path: Path, output_zip: Path) -> None:
    print(f"[2/2] Packaging dataset from {dataset_path} to {output_zip}...")
    count = 0
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(dataset_path.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(dataset_path)
            arcname = rel.as_posix()
            zf.write(p, arcname=arcname)
            count += 1
    print(f"Successfully packaged {count} dataset files into {output_zip}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", default="", help="Path to bgspcd_v4_robust folder")
    parser.add_argument("--output-dir", default="kaggle_assets", help="Destination folder for zip packages")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    out_dir = (project_root / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    code_zip = out_dir / "gpbsf_code.zip"
    package_code(project_root, code_zip)

    meta_code = {
        "title": "gpbsf-code",
        "id": "[YOUR_KAGGLE_USERNAME]/gpbsf-code",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (out_dir / "dataset-metadata-code.json").write_text(json.dumps(meta_code, indent=2), encoding="utf-8")

    if args.dataset_path:
        ds_path = Path(args.dataset_path).resolve()
        if ds_path.is_dir():
            ds_zip = out_dir / "gpbsf_dataset_v4.zip"
            package_dataset(ds_path, ds_zip)
            meta_ds = {
                "title": "gpbsf-dataset-v4",
                "id": "[YOUR_KAGGLE_USERNAME]/gpbsf-dataset-v4",
                "licenses": [{"name": "CC0-1.0"}],
            }
            (out_dir / "dataset-metadata-dataset.json").write_text(json.dumps(meta_ds, indent=2), encoding="utf-8")
        else:
            print(f"Warning: Dataset path {ds_path} does not exist.")

    print(f"\nAll assets ready in {out_dir}")


if __name__ == "__main__":
    main()
