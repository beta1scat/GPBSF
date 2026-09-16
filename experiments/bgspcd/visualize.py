"""Visualize observed BGSPCD samples and their retained analytic ground truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

from .dataset import _apply_pose, _sample_surface


def _records(dataset_root: Path, split: Optional[str], source: Optional[str]) -> list:
    manifest = dataset_root / "manifest.jsonl"
    records = []
    with manifest.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                record = json.loads(line)
                if (split is None or record["split"] == split) and (source is None or record.get("source_dataset") == source):
                    records.append(record)
    if not records:
        raise ValueError(f"No records were found in {manifest}")
    return records


def _select_record(records: list, sample_id: Optional[str], index: int) -> tuple:
    if sample_id:
        for record in records:
            if record["sample_id"] == sample_id:
                return records.index(record), record
        raise ValueError(f"Sample {sample_id!r} was not found in the selected manifest records")
    if not 0 <= index < len(records):
        raise IndexError(f"index must be in [0, {len(records) - 1}]")
    return index, records[index]


def _truth_surface(record: dict, count: int) -> np.ndarray:
    rng = np.random.default_rng(int(record["seed"]) + 809)
    local = _sample_surface(record["primitive"], rng, count, record["dimensions"])
    return _apply_pose(local, np.asarray(record["pose"], dtype=np.float64))


def _open3d(dataset_root: Path, records: list, start_index: int, view: str, truth_points: int) -> None:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError("Interactive visualization requires open3d.") from exc
    state = {"index": start_index}
    observed_geometry = o3d.geometry.PointCloud()
    truth_geometry = o3d.geometry.PointCloud()

    def load_current() -> None:
        record = records[state["index"]]
        with np.load(dataset_root / record["path"], allow_pickle=False) as sample:
            observed = np.asarray(sample["points"], dtype=np.float64)
        truth = _truth_surface(record, truth_points)
        observed_geometry.points = o3d.utility.Vector3dVector(observed)
        observed_geometry.paint_uniform_color([0.10, 0.35, 0.95])
        truth_geometry.points = o3d.utility.Vector3dVector(truth)
        truth_geometry.paint_uniform_color([0.10, 0.75, 0.25])
        print(
            f"[{state['index'] + 1}/{len(records)}] {record['sample_id']} | "
            f"{record['primitive']} | {record.get('difficulty', record.get('observation_mode', 'unknown'))} | "
            f"visible={record.get('visible_fraction', float('nan')):.3f} | "
            f"camera_visible={record.get('camera_visible_fraction', float('nan')):.3f} | "
            f"occluder={record.get('occluder_fraction', float('nan')):.3f} | "
            f"noise={record.get('noise_std', float('nan')):.4f} m | "
            f"outliers={record.get('outlier_ratio', float('nan')):.3f}"
        )

    def update(visualizer) -> None:
        load_current()
        if view in ("observed", "both"):
            visualizer.update_geometry(observed_geometry)
        if view in ("truth", "both"):
            visualizer.update_geometry(truth_geometry)
        visualizer.reset_view_point(True)

    visualizer = o3d.visualization.VisualizerWithKeyCallback()
    visualizer.create_window(window_name="BGSPCD: blue=observed, green=ground truth | N=next, Q=quit")
    # Populate the geometry before adding it. Otherwise Open3D computes the
    # initial view bounds from the coordinate frame alone and hides the points.
    load_current()
    if view in ("observed", "both"):
        visualizer.add_geometry(observed_geometry)
    if view in ("truth", "both"):
        visualizer.add_geometry(truth_geometry)
    visualizer.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08))
    visualizer.get_render_option().point_size = 2.0

    def next_sample(window):
        state["index"] = (state["index"] + 1) % len(records)
        update(window)
        return False

    def quit_viewer(window):
        window.close()
        return False

    visualizer.register_key_callback(ord("N"), next_sample)
    visualizer.register_key_callback(ord("n"), next_sample)
    visualizer.register_key_callback(ord("Q"), quit_viewer)
    visualizer.register_key_callback(ord("q"), quit_viewer)
    visualizer.reset_view_point(True)
    visualizer.run()
    visualizer.destroy_window()


def _save_matplotlib(observed: np.ndarray, truth: np.ndarray, view: str, output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("PNG export requires matplotlib.") from exc
    figure = plt.figure(figsize=(7, 6))
    axis = figure.add_subplot(111, projection="3d")
    if view in ("observed", "both"):
        axis.scatter(observed[:, 0], observed[:, 1], observed[:, 2], s=1, c="#1a59f0", label="observed")
    if view in ("truth", "both"):
        axis.scatter(truth[:, 0], truth[:, 1], truth[:, 2], s=1, c="#20a842", alpha=0.55, label="ground truth")
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_zlabel("z (m)")
    axis.legend(loc="best")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="data/bgspcd_v3_camera")
    parser.add_argument("--sample-id", help="sample id from manifest.jsonl")
    parser.add_argument("--index", type=int, default=0, help="zero-based manifest index after optional --split filtering")
    parser.add_argument("--split", choices=("train", "val", "test"))
    parser.add_argument("--source", help="optional source_dataset filter, e.g. v4_camera")
    parser.add_argument("--view", choices=("observed", "truth", "both"), default="both")
    parser.add_argument("--truth-points", type=int, default=4096)
    parser.add_argument("--save", help="optional PNG output path; otherwise opens an Open3D window")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    records = _records(dataset_root, args.split, args.source)
    selected_index, record = _select_record(records, args.sample_id, args.index)
    with np.load(dataset_root / record["path"], allow_pickle=False) as sample:
        observed = np.asarray(sample["points"], dtype=np.float64)
    truth = _truth_surface(record, args.truth_points)
    print(json.dumps(record, ensure_ascii=False, indent=2))
    if args.save:
        _save_matplotlib(observed, truth, args.view, Path(args.save).resolve())
        print(f"Wrote {Path(args.save).resolve()}")
    else:
        _open3d(dataset_root, records, selected_index, args.view, args.truth_points)


if __name__ == "__main__":
    main()
