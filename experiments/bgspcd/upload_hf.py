"""Upload BGSPCD-v4-robust dataset to Hugging Face Hub using the Dual-Track Hybrid Approach.

Dual-Track Structure:
  1. Root level files (direct web preview, metadata streaming):
     - README.md (Dataset Card with academic metadata and citation)
     - dataset_metadata.json (sensor, camera, and geometry generation configurations)
     - manifest.jsonl (complete sample index with ground-truth parameters)
  2. Compressed archives (fast, atomic single-file download):
     - samples.tar.gz (or samples.zip): train, val, and test point clouds (.npz)
     - legacy_cache.tar.gz (or legacy_cache.zip): PointNet/PointNet2 cache
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tarfile
import time
import zipfile


README_TEMPLATE = """---
license: mit
task_categories:
- robotics
- object-detection
- other
tags:
- 3d-object-detection
- 3d-vision
- point-cloud
- shape-fitting
- robotics
- primitive-representation
- mamba3d
- pointnet2
size_categories:
- 10K<n<100K
pretty_name: "BGSPCD-v4-robust (Bounded Geometric Surface Point Cloud Dataset)"
---

# BGSPCD-v4-robust: Robust Benchmark Geometric Primitive Point Cloud Dataset

## 1. Overview
**BGSPCD-v4-robust** is a standardized point cloud dataset designed for evaluating 3D geometric primitive classification, topology inference, and bounded surface fitting in unstructured robotic grasp planning.

The dataset explicitly evaluates algorithm robustness under simulated single-view observation, optical sensor noise, partial truncation, spatial occlusion, sparse sampling, and arbitrary $SE(3)$ pose transformations.

## 2. Geometric Primitives
All samples belong to three canonical topological classes (strictly 1-to-1 mapped with neural classification logits):
- **Class 0 (`cuboid`)**: Length, width, and height parameterized boxes.
- **Class 1 (`frustum`)**: Bottom radius, top-to-bottom ratio, and axial height parameterized truncated cones and cylinders.
- **Class 2 (`ellipsoid`)**: Tri-axial semi-axes parameterized ellipsoids.

## 3. Observation Perturbation Modes
Each shape family consists of 8 observation variants sharing identical canonical parameters and pose:
1. `full_surface_prior`: Uniform ground-truth surface sampling (32,768 points candidate, 2,048 sampled).
2. `single_view`: Standard synthetic ray-casting single-view depth camera observation.
3. `single_view_clean`: Single-view point cloud without sensor noise.
4. `single_view_occluded`: Realistic synthetic occlusion masking.
5. `single_view_partial`: High-ratio boundary truncation.
6. `single_view_rotated`: Arbitrary $SE(3)$ rotational perturbations.
7. `single_view_sensor`: Realistic depth camera Gaussian noise and lateral quantization.
8. `single_view_sparse`: Low-density subsampled surface points.

## 4. Archive Layout (Dual-Track)
```text
bgspcd-v4-robust/
├── README.md                  # Dataset Card & documentation
├── dataset_metadata.json      # Camera intrinsics, primitive ranges & generator seed
├── manifest.jsonl             # Master sample index (sample_id, pose, dimensions, label)
├── samples.tar.gz             # Compressed core point cloud archives (.npz)
│   └── samples/
│       ├── train/ {cuboid, frustum, ellipsoid}
│       ├── val/   {cuboid, frustum, ellipsoid}
│       └── test/  {cuboid, frustum, ellipsoid}
└── legacy_cache.tar.gz        # Cache for PointNet/PointNet2 data loaders (.npz)
```

## 5. Quick Start (Python)

### Download via Hugging Face Hub:
```python
from huggingface_hub import hf_hub_download
import tarfile

# Download metadata and master index
manifest_path = hf_hub_download(repo_id="{repo_id}", filename="manifest.jsonl", repo_type="dataset")
meta_path = hf_hub_download(repo_id="{repo_id}", filename="dataset_metadata.json", repo_type="dataset")

# Download and extract the point clouds
samples_archive = hf_hub_download(repo_id="{repo_id}", filename="samples.tar.gz", repo_type="dataset")
with tarfile.open(samples_archive, "r:gz") as tar:
    tar.extractall(path="./data/bgspcd_v4_robust")
```

### Read a Sample:
```python
import numpy as np

# Load sample .npz
data = np.load("./data/bgspcd_v4_robust/samples/test/cuboid/cuboid_000008_single_view.npz")
points = data["points"]  # (2048, 3) float32
label = int(data["label"])  # 0, 1, or 2

print(f"Loaded point cloud shape: {points.shape}, label: {label}")
```
"""


def _compress_to_tar_gz(source_dir: Path, output_archive: Path) -> None:
    """Compress a directory into a .tar.gz archive with progress reporting."""
    print(f"[*] Packaging directory {source_dir.name} -> {output_archive.name} ...")
    start_time = time.time()
    total_files = sum(1 for _ in source_dir.rglob("*") if _.is_file())
    count = 0

    with tarfile.open(output_archive, "w:gz") as tar:
        for file_path in source_dir.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(source_dir.parent)
                tar.add(file_path, arcname=str(arcname))
                count += 1
                if count % 2000 == 0 or count == total_files:
                    elapsed = time.time() - start_time
                    print(f"    Progress: {count}/{total_files} files packaged ({elapsed:.1f}s)")

    size_mb = output_archive.stat().st_size / (1024 * 1024)
    print(f"[+] Successfully generated {output_archive.name}: {size_mb:.2f} MB in {time.time() - start_time:.1f}s")


def _compress_to_zip(source_dir: Path, output_archive: Path) -> None:
    """Compress a directory into a .zip archive with progress reporting."""
    print(f"[*] Packaging directory {source_dir.name} -> {output_archive.name} ...")
    start_time = time.time()
    total_files = sum(1 for _ in source_dir.rglob("*") if _.is_file())
    count = 0

    with zipfile.ZipFile(output_archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in source_dir.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(source_dir.parent)
                zf.write(file_path, arcname=str(arcname))
                count += 1
                if count % 2000 == 0 or count == total_files:
                    elapsed = time.time() - start_time
                    print(f"    Progress: {count}/{total_files} files packaged ({elapsed:.1f}s)")

    size_mb = output_archive.stat().st_size / (1024 * 1024)
    print(f"[+] Successfully generated {output_archive.name}: {size_mb:.2f} MB in {time.time() - start_time:.1f}s")


def prepare_and_upload(
    local_dir: str | Path,
    repo_id: str,
    private: bool = False,
    token: str | None = None,
    archive_format: str = "tar.gz",
    skip_pack: bool = False,
    include_legacy: bool = True,
) -> None:
    source_path = Path(local_dir).resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Local dataset path does not exist: {source_path}")

    # 1. Verify necessary files
    meta_file = source_path / "dataset_metadata.json"
    manifest_file = source_path / "manifest.jsonl"
    samples_dir = source_path / "samples"
    legacy_dir = source_path / "legacy_cache"

    if not meta_file.exists():
        raise FileNotFoundError(f"Missing {meta_file}")
    if not manifest_file.exists():
        raise FileNotFoundError(f"Missing {manifest_file}")
    if not samples_dir.exists():
        raise FileNotFoundError(f"Missing {samples_dir}")

    # 2. Write/Update README.md
    readme_path = source_path / "README.md"
    readme_content = README_TEMPLATE.replace("{repo_id}", repo_id)
    readme_path.write_text(readme_content, encoding="utf-8")
    print(f"[+] Prepared Dataset Card: {readme_path}")

    # 3. Package archives
    ext = ".tar.gz" if archive_format == "tar.gz" else ".zip"
    compress_fn = _compress_to_tar_gz if archive_format == "tar.gz" else _compress_to_zip

    samples_archive = source_path / f"samples{ext}"
    if not skip_pack or not samples_archive.exists():
        compress_fn(samples_dir, samples_archive)
    else:
        print(f"[*] Reusing existing archive: {samples_archive} ({samples_archive.stat().st_size / (1024*1024):.2f} MB)")

    legacy_archive = None
    if include_legacy and legacy_dir.exists():
        legacy_archive = source_path / f"legacy_cache{ext}"
        if not skip_pack or not legacy_archive.exists():
            compress_fn(legacy_dir, legacy_archive)
        else:
            print(f"[*] Reusing existing archive: {legacy_archive} ({legacy_archive.stat().st_size / (1024*1024):.2f} MB)")

    # 4. Upload to Hugging Face
    try:
        from huggingface_hub import HfApi, login
    except ImportError:
        print("[!] ERROR: 'huggingface_hub' is not installed.")
        print("    Please run: pip install --upgrade huggingface_hub")
        sys.exit(1)

    if token:
        login(token=token)

    api = HfApi()

    print(f"[*] Ensuring public Hugging Face dataset repository exists: {repo_id}")
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )

    files_to_upload = [
        (readme_path, "README.md"),
        (meta_file, "dataset_metadata.json"),
        (manifest_file, "manifest.jsonl"),
        (samples_archive, f"samples{ext}"),
    ]
    if legacy_archive and legacy_archive.exists():
        files_to_upload.append((legacy_archive, f"legacy_cache{ext}"))

    print(f"[*] Uploading {len(files_to_upload)} files to https://huggingface.co/datasets/{repo_id} ...")
    for local_f, remote_f in files_to_upload:
        f_size_mb = local_f.stat().st_size / (1024 * 1024)
        print(f"    --> Uploading {remote_f} ({f_size_mb:.2f} MB)...")
        api.upload_file(
            path_or_fileobj=str(local_f),
            path_in_repo=remote_f,
            repo_id=repo_id,
            repo_type="dataset",
        )
        print(f"    [OK] Uploaded {remote_f}")

    print("\n" + "=" * 60)
    print(f"[+] All files successfully uploaded to:")
    print(f"    https://huggingface.co/datasets/{repo_id}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Upload BGSPCD-v4-robust to Hugging Face Hub (Dual-Track Mode)")
    parser.add_argument("--dir", default=r"C:\Users\robot\Downloads\bgspcd_v4_robust", help="Path to bgspcd_v4_robust")
    parser.add_argument("--repo", required=True, help="Target Hugging Face repo (e.g. username/bgspcd-v4-robust)")
    parser.add_argument("--token", default=None, help="Hugging Face User Access Token (Write)")
    parser.add_argument("--private", action="store_true", help="Set repository visibility to private (default: public)")
    parser.add_argument("--format", choices=["tar.gz", "zip"], default="tar.gz", help="Archive compression format")
    parser.add_argument("--skip-pack", action="store_true", help="Skip packaging if archive files already exist")
    parser.add_argument("--exclude-legacy", action="store_true", help="Do not package or upload legacy_cache")
    args = parser.parse_args()

    prepare_and_upload(
        local_dir=args.dir,
        repo_id=args.repo,
        private=args.private,
        token=args.token,
        archive_format=args.format,
        skip_pack=args.skip_pack,
        include_legacy=not args.exclude_legacy,
    )


if __name__ == "__main__":
    main()
