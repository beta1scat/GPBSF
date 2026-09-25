Geometric Primitive-Based Shape Fitting for Scene Objects (GPBSF).

The repository contains the interactive RGB-D fitting demo and the reproducible
Chapter 4 experiment pipeline. PointNet++ and Mamba3D are pinned as Git
submodules; shared GPBSF code controls the data split, preprocessing, training
budget, metrics, and aggregation so that the comparison uses one protocol.

---

## 1. Initialize model submodules

```bash
git submodule update --init --recursive
```

The pinned revisions and their roles are documented in
[`submodels/README.md`](submodels/README.md).

## 2. Interactive demo model preparation

Download the SAM model checkpoint from the official link:  
👉 [https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)

Then, place the file into the `models/` directory of this repository:

```bash
models/
└── sam_vit_h_4b8939.pth
```

---

## 3. Run the legacy interactive demo with Docker

### Step 1: Enter Docker Configuration Folder

```bash
cd .docker
```

### Step 2: Launch Container

Run the following command, replacing the path with your actual local path to GPBSF:

```bash
PATH_TO_GPBSF=/absolute/path/to/gpbsf docker compose run gpu
```

> 💡 **Tip:**  
> Make sure Docker and NVIDIA Container Toolkit are properly installed to enable GPU access.

---

## 4. Run the legacy interactive demo

After entering the container (or on your host system if dependencies are installed):

```bash
cd ~/code/gpbsf
python main.py
```

---

## 5. Chapter 4 reproducible experiments

The experiment pipeline is intentionally separate from `main.py`, which opens
interactive windows and loads checkpoints during module import.

The Mamba3D submodule documents its reference environment as Python 3.8,
PyTorch 1.13.1, and CUDA 11.7. Install its pinned requirements and CUDA
extensions in a dedicated environment before running the Mamba3D entries; the
PointNet++ baseline must use the same PyTorch environment and the same GPU.

Launch the dedicated pre-configured Mamba3D container directly (recommended, image: `beta1scat/mamba3d:1.0`):

```bash
cd .docker
PATH_TO_GPBSF=/absolute/path/to/gpbsf docker compose run mamba3d
```

Alternatively, when using the base Docker image (`beta1scat/gpbsf:1.0` via `docker compose run gpu`), keep NumPy below version 2. The image's
PyTorch binary was compiled against the NumPy 1.x C API; NumPy 2 causes the
`_ARRAY_API not found` warning and may make tensor--NumPy conversion fail.
Install dependencies from inside the running container, at the GPBSF root:

```bash
python -m pip install -r submodels/mamba3d/requirements.txt
python -m pip uninstall -y opencv-python-headless
python -m pip install --upgrade --force-reinstall "numpy==1.26.4" "opencv-python==4.10.0.84"
python -m pip install causal-conv1d==1.1.1 mamba-ssm==1.1.1
(cd submodels/mamba3d/extensions/chamfer_dist && python setup.py install --user)
```

The original `unlimblue/KNN_CUDA` release was removed. The experiment adapter
uses the installed compatible `knn_cuda.KNN` package. Record the package source
and CUDA/PyTorch versions with the experiment outputs when reporting latency.

The Mamba3D adapter resolves the upstream embedded `bimamba_ssm/ops` package
from `submodels/mamba3d` at runtime. Do not set a machine-specific `PYTHONPATH`
such as `/home/Mamba3D`.

Generate a family-disjoint synthetic dataset:

The current `BGSPCD-v3-camera` protocol renders only front-facing points that
survive a pinhole-camera z-buffer, then applies optional image-plane occlusion,
depth quantization, measurement noise, and outliers. The complete analytic
surface is retained only as evaluation ground truth.

The default configuration contains 1,800 geometry families. Each family has
three difficulty levels (`easy`, `medium`, and `hard`) and three independent
corruption variants per level, producing 16,200 samples in total. Splitting is
performed at the family level, not the individual point-cloud level, so all
variants of one geometry remain in exactly one set:

| Split | Families | Samples per primitive | Total samples |
| --- | ---: | ---: | ---: |
| Train | 1,260 | 3,780 | 11,340 |
| Validation | 270 | 810 | 2,430 |
| Test | 270 | 810 | 2,430 |

Every saved sample contains exactly 2,048 input points. This is a fixed neural
network input resolution, not a claim about the raw number of depth-camera
pixels: the visible partial cloud is resampled after camera rendering. The
complete surface, dimensions, pose, camera metadata, and actual visible ratio
are retained for ground-truth fitting evaluation only.

```bash
cd /path/to/graduate-thesis/code/GPBSF
python -m experiments.bgspcd generate \
  --config config/experiments/bgspcd.json
```

On Windows PowerShell, explicitly use the GPBSF directory as the working
directory. The Python executable may be supplied from another virtual
environment, but that does not make `GPBSF/experiments` importable by itself:

```powershell
Set-Location D:\0-research\00-papers\thesis\graduate-thesis\code\GPBSF
& D:\0-research\00-papers\thesis\graduate-thesis\code\dh-transform-ik\.venv\Scripts\python.exe `
  -m experiments.bgspcd generate --config config\experiments\bgspcd.json
```

Validate its manifest and split isolation:

```bash
python -m experiments.bgspcd validate \
  --dataset-root data/bgspcd_v3_camera
```

### V4-Robust training corpus

For the final Chapter 4 training run, use `BGSPCD-v4-robust` rather than
overwriting either `BGSPCD-v3-camera` or the public legacy corpus. V4 combines
two explicitly identified sources:

- `legacy_public`: the original public BGSPCD files. The original test list is
  retained as test-only; a fixed, stratified 10% of its original training list
  becomes validation data. The remaining legacy samples are training-only.
- `v4_camera`: 1,800 new family-disjoint geometries (600 per primitive), each
  rendered under seven single-view camera conditions and one complete-surface
  condition. Camera-centred rejection sampling requires at least 4,096 unique
  rendered points before a sample is written. The final 2,048 V4 camera points
  are drawn without replacement, so the new source never duplicates a small
  visible fragment merely to reach network input size. Legacy samples retain
  their historical variable point counts; the cached record exposes any
  necessary deterministic replacement sampling through `duplicate_ratio`.

The generated corpus contains 30,400 samples: 23,040 training, 2,720
validation, and 4,640 test samples. `dataset_metadata.json` records source,
split, and observation-mode counts. The V4 camera renderer uses a
foreground-sphere depth occluder rather than V3's rectangular image mask. The public external
`data/experiment` collection remains evaluation-only and is not consulted by
the generator.

Generate V4 once. The command converts the legacy ASCII XYZ+normal files into
compact XYZ-only NPZ cache files under the new V4 directory; this can take
substantial disk I/O but avoids reparsing 10,000-row text files every epoch.

```bash
cd /path/to/graduate-thesis/code/GPBSF
python -m experiments.bgspcd.robust \
  --config config/experiments/bgspcd_v4_robust.json
```

The command refuses to overwrite an existing output directory. To build a
new, separately named candidate, pass `--output data/bgspcd_v4_robust_r2` and
update the classifier configuration manifest accordingly. `--without-legacy`
builds only the new camera-conditioned source and is not the final V4-Robust protocol.

Validate the complete mixed manifest before training:

```bash
python -m experiments.classification.cli \
  --config config/experiments/classification_v4_robust.json \
  validate-data
```

To inspect only new V4 camera observations rather than the cached legacy rows:

```bash
python -m experiments.bgspcd.visualize \
  --dataset-root data/bgspcd_v4_robust --source v4_camera --split test --index 0 --view both
```

Use the V4 Mamba3D configuration, which matches the historical architecture
regularisation and budget (`drop_path_rate=0.1`, batch size 32, 125 epochs):

```bash
python scripts/run_classification_matrix.py \
  --models mamba3d \
  --seeds 3407 3408 3409 \
  --config config/experiments/classification_v4_robust.json
```

After the Mamba3D results are acceptable, use the same V4 configuration for
PointNet++ so both models have identical inputs, splits, preprocessing, and
training seeds:

```bash
python scripts/run_classification_matrix.py \
  --models pointnet2 \
  --seeds 3407 3408 3409 \
  --config config/experiments/classification_v4_robust.json
```

Aggregate test results and then run the immutable external evaluation with
the same V4 configuration:

```bash
python -m experiments.classification.cli \
  --config config/experiments/classification_v4_robust.json \
  aggregate --runs-dir runs/classification_v4_robust

python scripts/run_external_evaluation.py \
  --models mamba3d \
  --seeds 3407 3408 3409 \
  --config config/experiments/classification_v4_robust.json \
  --manifest data/experiment/external_manifest.jsonl
```

For a Hugging Face release, publish `data/bgspcd_v4_robust/` together with
`config/experiments/bgspcd_v4_robust.json`, `dataset_metadata.json`, and this
repository revision. The cached legacy subset is included so the released V4
manifest is self-contained; document the original BGSPCD Hugging Face dataset
as the source of that subset.

Visualize a generated test example. Blue points are the corrupted partial
observation; green points are the full analytic surface reconstructed from the
stored ground truth:

```bash
python -m experiments.bgspcd.visualize \
  --dataset-root data/bgspcd_v3_camera --split test --index 0 --view both
```

To export a PNG rather than opening an interactive Open3D window, add
`--save runs/figures/bgspcd_test_000.png`.
In the interactive window, press `N` to advance to the next sample and `Q` to
close the viewer. The terminal prints the selected sample's camera-visible
fraction, foreground-occluder fraction, noise level, and outlier ratio.

Run the PointNet++ and Mamba3D three-seed matrix:

```bash
python scripts/run_classification_matrix.py \
  --models pointnet2 mamba3d \
  --seeds 3407 3408 3409 \
  --config config/experiments/classification.json
```

Aggregate the held-out test results across seeds:

```bash
python -m experiments.classification.cli \
  --config config/experiments/classification.json \
  aggregate --runs-dir runs/classification
```

### External simulation/real-data validation

`data/experiment` is an immutable external test set, not a source for training,
early stopping, hyperparameter selection, or checkpoint selection. Build its
manifest once. The builder pairs each `pcd/*.ply` file with the identically
named `classes/*.json` label and uses only `input_type`; it maps `0`/`01` to
cuboid, `1`/`11`/`12`/`13`/`14` to frustum, and `2` to ellipsoid. `pred_type`,
fitted size, and pose are never used by the classifier evaluation.

```bash
python -m experiments.classification.cli \
  --config config/experiments/classification.json \
  build-external-manifest \
  --dataset-root data/experiment \
  --output data/experiment/external_manifest.jsonl
```

After all synthetic-data runs have completed, evaluate their existing
validation-selected `best.pt` checkpoints. This writes independent results for
`real_flat`, `real_clutter`, `sim_flat`, and `sim_clutter`; it does not combine
the four conditions into a model-selection score.

```bash
python scripts/run_external_evaluation.py \
  --models pointnet2 mamba3d \
  --seeds 3407 3408 3409 \
  --config config/experiments/classification.json \
  --manifest data/experiment/external_manifest.jsonl
```

For each model/seed, outputs are stored under
`runs/classification/<model>/seed_<seed>/external/external_manifest/`, with
per-condition JSON summaries, prediction CSVs, and confusion matrices. After
all seeds are evaluated, calculate mean, standard deviation, and a bootstrap
95% interval across seeds for each condition separately:

```bash
python -m experiments.classification.cli \
  --config config/experiments/classification.json \
aggregate-external --runs-dir runs/classification
```

### Archived Mamba3D checkpoint

The original `models/ckpt-best.pth` uses the upstream Mamba3D checkpoint
format (`base_model` with `module.`-prefixed keys), rather than this
experiment pipeline's `best.pt` format. Evaluate it separately; never replace
the three-seed experiment checkpoints with it. The command uses the original
`config/bgspcd.yaml` model configuration, evaluates the synthetic held-out
test set and all four external conditions, and writes a standalone result.

```bash
python -m experiments.classification.cli \
  --config config/experiments/classification.json \
  evaluate-legacy-mamba3d \
  --checkpoint models/ckpt-best.pth \
  --legacy-model-config config/bgspcd.yaml \
  --output-dir runs/classification/mamba3d/legacy_ckpt_best \
  --external-manifest data/experiment/external_manifest.jsonl \
  --evaluation-seed 3407
```

Each `(model, seed)` pair runs in a separate process. This is required because
both upstream repositories use generic top-level Python package names such as
`models`, `tools`, and `utils`.
Training prints one line per epoch and also writes the same messages to
`runs/classification/<model>/seed_<seed>/training.log`; epoch-level structured
metrics are stored in `metrics.jsonl` in that same directory.

Evaluate geometric fitting against the synthetic geometry truth retained in the
manifest. This reports observed-to-fitted residual separately from the
independent fitted-to-truth surface error. The resulting CSV separates
`fit_success` (the fitter returned parameters) from `metrics_valid` (all
ground-truth metric calculations completed), so an evaluation-side exception
cannot be misreported as a fitting failure:

```bash
python -m experiments.fitting.evaluate \
  --manifest data/bgspcd_v3_camera/manifest.jsonl \
  --split test \
  --output runs/fitting/oracle_test.csv
```

To assess the final V4-Robust mixed corpus, replace the manifest and use a
new output path. V4 records are grouped as `not_stratified` in the difficulty
summary because their controlled factors are stored as observation modes:

```bash
python -m experiments.fitting.evaluate \
  --manifest data/bgspcd_v4_robust/manifest.jsonl \
  --split test \
  --output runs/fitting/v4_robust_oracle_test.csv \
  --export-fallback-log runs/fitting/fallback_triggers.csv

python3 experiments/fitting/format_table43.py \
  --summary runs/fitting/v4_robust_oracle_test.summary.json \
  --csv runs/fitting/v4_robust_oracle_test.csv
```

## 6. Notes

- The maintained primitive-fitting implementation is
  [`shape_fitting/`](shape_fitting/), copied from the current LGGPF fitting
  source and kept inside GPBSF. New code should import `FittingByBGS` from
  `shape_fitting`.
- Ensure the `models/` directory exists before running the interactive demo.
- Always use an absolute path for `PATH_TO_GPBSF` when running Docker.
- Generated datasets and run directories are ignored by Git. Preserve the
  resolved configuration, manifest hash, predictions, and aggregate tables when
  archiving a thesis experiment.

---
