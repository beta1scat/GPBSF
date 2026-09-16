"""Metrics and result writers with no sklearn dependency."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .labels import CLASS_NAMES


def confusion_matrix(targets: Sequence[int], predictions: Sequence[int], classes: int = 3) -> np.ndarray:
    matrix = np.zeros((classes, classes), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        matrix[int(target), int(prediction)] += 1
    return matrix


def classification_metrics(targets: Sequence[int], predictions: Sequence[int]) -> dict:
    matrix = confusion_matrix(targets, predictions)
    total = int(matrix.sum())
    recalls, precisions, f1_scores = [], [], []
    for index in range(matrix.shape[0]):
        true_positive = float(matrix[index, index])
        false_negative = float(matrix[index, :].sum() - matrix[index, index])
        false_positive = float(matrix[:, index].sum() - matrix[index, index])
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls.append(recall)
        precisions.append(precision)
        f1_scores.append(f1)
    return {
        "sample_count": total,
        "accuracy": float(np.trace(matrix) / total) if total else 0.0,
        "macro_f1": float(np.mean(f1_scores)),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "per_class_recall": {name: float(recalls[index]) for index, name in enumerate(CLASS_NAMES)},
        "confusion_matrix": matrix.tolist(),
    }


def write_confusion_matrix(path, matrix: Sequence[Sequence[int]]) -> None:
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true\\pred", *CLASS_NAMES])
        for name, row in zip(CLASS_NAMES, matrix):
            writer.writerow([name, *row])


def write_predictions(path, rows: Iterable[dict]) -> None:
    path = Path(path)
    fields = ("sample_id", "target", "prediction", "correct")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_external_predictions(path, rows: Iterable[dict]) -> None:
    path = Path(path)
    fields = ("sample_id", "group", "target", "prediction", "correct")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_mean(values: Sequence[float], repetitions: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return math.nan, math.nan
    if values.size == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        means[index] = rng.choice(values, size=values.size, replace=True).mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))
