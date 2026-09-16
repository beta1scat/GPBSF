"""One-protocol training and evaluation engine for PointNet++ and Mamba3D."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .adapters import build_adapter
from .dataset import ManifestPointCloudDataset
from .external import ExternalPointCloudDataset, GROUPS
from .metrics import classification_metrics, write_confusion_matrix, write_external_predictions, write_predictions
from .preprocess import augment_training_points
from .reproducibility import environment_record, file_sha256, set_seed


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_config(path) -> dict:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def resolve_device(requested: str):
    import torch

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def run_directory(config: dict, model: str, seed: int, override=None) -> Path:
    root = repository_root()
    if override:
        return Path(override).resolve()
    return (root / config["output"]["runs_dir"] / model / f"seed_{seed}").resolve()


def _build_loaders(config: dict, seed: int):
    import torch

    root = repository_root()
    dataset_config = config["dataset"]
    manifest = root / dataset_config["manifest"]
    points = int(config["training"]["num_points"])
    train = ManifestPointCloudDataset(manifest, "train", points, True, seed)
    val = ManifestPointCloudDataset(manifest, "val", points, False, seed)
    test = ManifestPointCloudDataset(manifest, "test", points, False, seed)
    generator = torch.Generator().manual_seed(seed)
    loader_args = {
        "batch_size": int(config["training"]["batch_size"]),
        "num_workers": int(config["training"].get("num_workers", 0)),
        "pin_memory": bool(config["training"].get("pin_memory", True)),
    }
    return (
        torch.utils.data.DataLoader(train, shuffle=True, generator=generator, drop_last=False, **loader_args),
        torch.utils.data.DataLoader(val, shuffle=False, drop_last=False, **loader_args),
        torch.utils.data.DataLoader(test, shuffle=False, drop_last=False, **loader_args),
    )


def _evaluate(adapter, loader, device, include_predictions: bool):
    import torch

    adapter.model.eval()
    targets, predictions, rows = [], [], []
    elapsed_seconds, seen = 0.0, 0
    with torch.no_grad():
        for points, labels, sample_ids in loader:
            points, labels = points.to(device), labels.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            scores = adapter.logits(points)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed_seconds += time.perf_counter() - started
            predicted = scores.argmax(dim=1)
            target_values = labels.cpu().tolist()
            predicted_values = predicted.cpu().tolist()
            targets.extend(target_values)
            predictions.extend(predicted_values)
            seen += len(target_values)
            if include_predictions:
                rows.extend(
                    {
                        "sample_id": sample_id,
                        "target": target,
                        "prediction": prediction,
                        "correct": int(target == prediction),
                    }
                    for sample_id, target, prediction in zip(sample_ids, target_values, predicted_values)
                )
    metrics = classification_metrics(targets, predictions)
    metrics["inference_ms_per_sample"] = 1000.0 * elapsed_seconds / seen if seen else 0.0
    return metrics, rows


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _log(destination: Path, message: str) -> None:
    """Emit progress immediately and retain the same text in the run folder."""
    print(message, flush=True)
    with (destination / "training.log").open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(message + "\n")


def train(config: dict, model_name: str, seed: int, output_override=None) -> Path:
    import torch

    set_seed(seed)
    root = repository_root()
    destination = run_directory(config, model_name, seed, output_override)
    if destination.exists():
        raise FileExistsError(f"Run directory already exists: {destination}. Use a new --run-dir.")
    device = resolve_device(config["training"].get("device", "auto"))
    print(f"[{model_name} | seed={seed}] Preparing model on {device} ...", flush=True)
    adapter = build_adapter(model_name, root, device, config.get("models", {}).get("mamba3d"))
    train_loader, val_loader, _ = _build_loaders(config, seed)
    destination.mkdir(parents=True)
    optimizer = torch.optim.AdamW(
        adapter.model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(config["training"]["epochs"]))
    generator = torch.Generator(device=device.type).manual_seed(seed + 31)
    resolved = dict(config)
    manifest = root / config["dataset"]["manifest"]
    resolved.update(
        {
            "model": model_name,
            "seed": seed,
            "device": str(device),
            "parameter_count": adapter.parameter_count(),
            "manifest_sha256": file_sha256(manifest),
        }
    )
    _write_json(destination / "config.resolved.json", resolved)
    _write_json(destination / "environment.json", environment_record(root))
    _log(
        destination,
        f"[{model_name} | seed={seed}] Start: train={len(train_loader.dataset)}, "
        f"val={len(val_loader.dataset)}, parameters={adapter.parameter_count():,}, "
        f"epochs={config['training']['epochs']}",
    )

    best_accuracy, best_epoch = -1.0, -1
    history_path = destination / "metrics.jsonl"
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        adapter.model.train()
        losses, correct, samples = [], 0, 0
        for points, labels, _ in train_loader:
            points, labels = points.to(device), labels.to(device)
            points = augment_training_points(points, generator)
            optimizer.zero_grad(set_to_none=True)
            scores = adapter.logits(points)
            loss = adapter.loss(scores, labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            correct += int((scores.argmax(dim=1) == labels).sum().item())
            samples += int(labels.numel())
        scheduler.step()
        validation, _ = _evaluate(adapter, val_loader, device, include_predictions=False)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "train_accuracy": correct / samples if samples else 0.0,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "val": validation,
        }
        with history_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        if validation["accuracy"] > best_accuracy:
            best_accuracy, best_epoch = validation["accuracy"], epoch
            torch.save(
                {
                    "model": model_name,
                    "seed": seed,
                    "epoch": epoch,
                    "model_state": adapter.model.state_dict(),
                    "config": resolved,
                },
                destination / "best.pt",
            )
        _log(
            destination,
            f"[{model_name} | seed={seed}] Epoch {epoch:03d}/{config['training']['epochs']} | "
            f"loss={row['train_loss']:.5f} | train_acc={row['train_accuracy']:.4f} | "
            f"val_acc={validation['accuracy']:.4f} | val_macro_f1={validation['macro_f1']:.4f} | "
            f"best_val_acc={best_accuracy:.4f}@{best_epoch}",
        )
    _write_json(destination / "train_summary.json", {"best_epoch": best_epoch, "best_val_accuracy": best_accuracy})
    _log(destination, f"[{model_name} | seed={seed}] Completed. Best validation accuracy: {best_accuracy:.4f}@{best_epoch}")
    return destination


def evaluate(config: dict, run_dir, split: str = "test") -> dict:
    import torch

    destination = Path(run_dir).resolve()
    resolved = json.loads((destination / "config.resolved.json").read_text(encoding="utf-8"))
    model_name, seed = resolved["model"], int(resolved["seed"])
    device = resolve_device(resolved.get("device", config["training"].get("device", "auto")))
    root = repository_root()
    adapter = build_adapter(model_name, root, device, config.get("models", {}).get("mamba3d"))
    checkpoint = torch.load(destination / "best.pt", map_location=device)
    adapter.model.load_state_dict(checkpoint["model_state"], strict=True)
    _, val_loader, test_loader = _build_loaders(config, seed)
    loader = test_loader if split == "test" else val_loader
    result, predictions = _evaluate(adapter, loader, device, include_predictions=True)
    result.update({"model": model_name, "seed": seed, "split": split, "parameter_count": adapter.parameter_count()})
    _write_json(destination / f"{split}_summary.json", result)
    write_predictions(destination / f"{split}_predictions.csv", predictions)
    write_confusion_matrix(destination / f"{split}_confusion_matrix.csv", result["confusion_matrix"])
    print(
        f"[{model_name} | seed={seed}] {split}: accuracy={result['accuracy']:.4f}, "
        f"macro_f1={result['macro_f1']:.4f}, latency={result['inference_ms_per_sample']:.3f} ms/sample",
        flush=True,
    )
    return result


def _external_loader(config: dict, manifest, seed: int, group: str):
    import torch

    dataset = ExternalPointCloudDataset(manifest, config["training"]["num_points"], seed, group)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        drop_last=False,
        num_workers=int(config["training"].get("num_workers", 0)),
        pin_memory=bool(config["training"].get("pin_memory", True)),
    )


def evaluate_external(config: dict, run_dir, external_manifest) -> dict:
    """Evaluate a validation-selected checkpoint without altering it or selecting on external data."""
    import torch

    destination = Path(run_dir).resolve()
    manifest = Path(external_manifest).resolve()
    resolved = json.loads((destination / "config.resolved.json").read_text(encoding="utf-8"))
    model_name, seed = resolved["model"], int(resolved["seed"])
    device = resolve_device(resolved.get("device", config["training"].get("device", "auto")))
    root = repository_root()
    adapter = build_adapter(model_name, root, device, config.get("models", {}).get("mamba3d"))
    checkpoint = torch.load(destination / "best.pt", map_location=device)
    adapter.model.load_state_dict(checkpoint["model_state"], strict=True)
    external_root = destination / "external" / manifest.stem
    external_root.mkdir(parents=True, exist_ok=True)
    group_results = {}
    for group in GROUPS:
        loader = _external_loader(config, manifest, seed, group)
        adapter.model.eval()
        targets, predictions, rows = [], [], []
        elapsed_seconds, seen = 0.0, 0
        with torch.no_grad():
            for points, labels, sample_ids, groups in loader:
                points, labels = points.to(device), labels.to(device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                scores = adapter.logits(points)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed_seconds += time.perf_counter() - started
                predicted = scores.argmax(dim=1)
                target_values, predicted_values = labels.cpu().tolist(), predicted.cpu().tolist()
                targets.extend(target_values)
                predictions.extend(predicted_values)
                seen += len(target_values)
                rows.extend(
                    {"sample_id": sample_id, "group": sample_group, "target": target, "prediction": prediction, "correct": int(target == prediction)}
                    for sample_id, sample_group, target, prediction in zip(sample_ids, groups, target_values, predicted_values)
                )
        result = classification_metrics(targets, predictions)
        result["inference_ms_per_sample"] = 1000.0 * elapsed_seconds / seen if seen else 0.0
        result["group"] = group
        group_results[group] = result
        write_external_predictions(external_root / f"{group}_predictions.csv", rows)
        write_confusion_matrix(external_root / f"{group}_confusion_matrix.csv", result["confusion_matrix"])
    summary = {
        "model": model_name,
        "seed": seed,
        "checkpoint": "best.pt",
        "checkpoint_selection": "synthetic_validation_accuracy",
        "external_manifest": str(manifest),
        "external_manifest_sha256": file_sha256(manifest),
        "groups": group_results,
    }
    _write_json(external_root / "external_summary.json", summary)
    print(f"[{model_name} | seed={seed}] external evaluation saved to {external_root}", flush=True)
    return summary


def evaluate_legacy_mamba3d(config: dict, checkpoint_path, legacy_model_config, output_dir, external_manifest, evaluation_seed: int) -> dict:
    """Evaluate an archived native Mamba3D checkpoint without modifying normal runs."""
    import torch
    import yaml

    checkpoint_path = Path(checkpoint_path).resolve()
    legacy_model_config = Path(legacy_model_config).resolve()
    destination = Path(output_dir).resolve()
    manifest = Path(external_manifest).resolve()
    if destination.exists():
        raise FileExistsError(f"Legacy evaluation directory already exists: {destination}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Legacy checkpoint is missing: {checkpoint_path}")
    with legacy_model_config.open("r", encoding="utf-8") as stream:
        model_config = yaml.safe_load(stream)["model"]
    set_seed(evaluation_seed)
    root = repository_root()
    device = resolve_device(config["training"].get("device", "auto"))
    adapter = build_adapter("mamba3d", root, device, model_config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "base_model" not in checkpoint:
        raise KeyError("Expected a native Mamba3D checkpoint with a 'base_model' state dictionary")
    state = {key.removeprefix("module."): value for key, value in checkpoint["base_model"].items()}
    adapter.model.load_state_dict(state, strict=True)
    checkpoint_metrics = {
        key: float(value.item()) if hasattr(value, "item") and getattr(value, "numel", lambda: 1)() == 1 else str(value)
        for key, value in checkpoint.get("best_metrics", {}).items()
    }
    destination.mkdir(parents=True)
    _, _, test_loader = _build_loaders(config, evaluation_seed)
    synthetic, predictions = _evaluate(adapter, test_loader, device, include_predictions=True)
    synthetic.update(
        {
            "model": "mamba3d_legacy",
            "evaluation_seed": evaluation_seed,
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "checkpoint_best_metrics": checkpoint_metrics,
            "parameter_count": adapter.parameter_count(),
        }
    )
    _write_json(destination / "test_summary.json", synthetic)
    write_predictions(destination / "test_predictions.csv", predictions)
    write_confusion_matrix(destination / "test_confusion_matrix.csv", synthetic["confusion_matrix"])
    external_root = destination / "external" / manifest.stem
    external_root.mkdir(parents=True)
    group_results = {}
    for group in GROUPS:
        loader = _external_loader(config, manifest, evaluation_seed, group)
        adapter.model.eval()
        targets, predictions, rows = [], [], []
        elapsed_seconds, seen = 0.0, 0
        with torch.no_grad():
            for points, labels, sample_ids, groups in loader:
                points, labels = points.to(device), labels.to(device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                scores = adapter.logits(points)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed_seconds += time.perf_counter() - started
                predicted = scores.argmax(dim=1)
                target_values, predicted_values = labels.cpu().tolist(), predicted.cpu().tolist()
                targets.extend(target_values)
                predictions.extend(predicted_values)
                seen += len(target_values)
                rows.extend(
                    {"sample_id": sample_id, "group": sample_group, "target": target, "prediction": prediction, "correct": int(target == prediction)}
                    for sample_id, sample_group, target, prediction in zip(sample_ids, groups, target_values, predicted_values)
                )
        result = classification_metrics(targets, predictions)
        result["inference_ms_per_sample"] = 1000.0 * elapsed_seconds / seen if seen else 0.0
        result["group"] = group
        group_results[group] = result
        write_external_predictions(external_root / f"{group}_predictions.csv", rows)
        write_confusion_matrix(external_root / f"{group}_confusion_matrix.csv", result["confusion_matrix"])
    summary = {
        "model": "mamba3d_legacy",
        "evaluation_seed": evaluation_seed,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best_metrics": checkpoint_metrics,
        "legacy_model_config": str(legacy_model_config),
        "legacy_model_config_sha256": file_sha256(legacy_model_config),
        "external_manifest": str(manifest),
        "external_manifest_sha256": file_sha256(manifest),
        "groups": group_results,
    }
    _write_json(external_root / "external_summary.json", summary)
    _write_json(destination / "legacy_evaluation.json", {"synthetic_test": synthetic, "external": summary})
    print(f"[mamba3d legacy] evaluation saved to {destination}", flush=True)
    return {"output_dir": str(destination), "synthetic_test": synthetic, "external": summary}
