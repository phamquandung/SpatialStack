#!/usr/bin/env python3
"""Summarize per-scene VLN metrics from an evaluation JSONL result file."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


METRICS = ("success", "spl", "os", "ne")
RESOURCE_METRICS = (
    "steps",
    "peak_alloc_mb",
    "peak_reserved_mb",
    "peak_vggt_kv_mb",
)
TIMING_METRICS = (
    "mean_step_ms",
    "mean_vggt_ms",
    "episode_time_s",
)
OPTIONAL_METRICS = RESOURCE_METRICS + TIMING_METRICS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute episode-weighted accuracy, step, and memory metrics for each "
            "scene in a SpatialStack VLN result.json JSONL file."
        )
    )
    parser.add_argument("result_path", type=Path, help="Path to result.json")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on an invalid JSON/episode line instead of warning and skipping it.",
    )
    return parser.parse_args()


def _warn_or_raise(message: str, strict: bool) -> None:
    if strict:
        raise ValueError(message)
    print(f"warning: {message}", file=sys.stderr)


def load_episode_rows(
    result_path: Path, strict: bool = False
) -> Tuple[List[Mapping[str, object]], Dict[str, int]]:
    """Load unique episode rows; aggregate and malformed lines are excluded."""
    if not result_path.is_file():
        raise FileNotFoundError(f"result file does not exist: {result_path}")

    unique: Dict[Tuple[str, str, str], Mapping[str, object]] = {}
    stats = {
        "episode_rows": 0,
        "summary_rows": 0,
        "duplicate_rows": 0,
        "invalid_rows": 0,
    }

    with result_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                stats["invalid_rows"] += 1
                _warn_or_raise(
                    f"line {line_number}: invalid JSON ({exc.msg}); skipped", strict
                )
                continue

            if not isinstance(row, dict):
                stats["invalid_rows"] += 1
                _warn_or_raise(f"line {line_number}: expected a JSON object; skipped", strict)
                continue
            if "episode_id" not in row:
                stats["summary_rows"] += 1
                continue

            missing = [name for name in ("scene_id", "episode_id", *METRICS) if name not in row]
            if missing:
                stats["invalid_rows"] += 1
                _warn_or_raise(
                    f"line {line_number}: missing fields {', '.join(missing)}; skipped", strict
                )
                continue

            try:
                metric_values = {name: float(row[name]) for name in METRICS}
            except (TypeError, ValueError):
                stats["invalid_rows"] += 1
                _warn_or_raise(
                    f"line {line_number}: metric fields must be numeric; skipped", strict
                )
                continue
            if not all(math.isfinite(value) for value in metric_values.values()):
                stats["invalid_rows"] += 1
                _warn_or_raise(
                    f"line {line_number}: metric fields must be finite; skipped", strict
                )
                continue

            normalized = dict(row)
            normalized.update(metric_values)
            for name in OPTIONAL_METRICS:
                if name not in row:
                    continue
                try:
                    value = float(row[name])
                except (TypeError, ValueError):
                    _warn_or_raise(
                        f"line {line_number}: {name} must be numeric; resource value ignored",
                        strict,
                    )
                    normalized.pop(name, None)
                    continue
                if not math.isfinite(value):
                    _warn_or_raise(
                        f"line {line_number}: {name} must be finite; resource value ignored",
                        strict,
                    )
                    normalized.pop(name, None)
                    continue
                normalized[name] = value
            key = (
                str(row["scene_id"]),
                str(row["episode_id"]),
                str(row.get("episode_instruction", "")),
            )
            stats["episode_rows"] += 1
            if key in unique:
                stats["duplicate_rows"] += 1
            unique[key] = normalized

    return list(unique.values()), stats


def mean_metrics(rows: Sequence[Mapping[str, object]]) -> Dict[str, float]:
    count = len(rows)
    if count == 0:
        raise ValueError("cannot summarize an empty episode collection")
    return {
        name: sum(float(row[name]) for row in rows) / count for name in METRICS
    }


def format_table(rows: Sequence[Mapping[str, object]]) -> str:
    grouped: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["scene_id"])].append(row)

    output_rows = []
    for scene_id in sorted(grouped):
        scene_rows = grouped[scene_id]
        metrics = mean_metrics(scene_rows)
        output_rows.append(
            (
                scene_id,
                str(len(scene_rows)),
                f"{100.0 * metrics['success']:.2f}",
                f"{100.0 * metrics['spl']:.2f}",
                f"{100.0 * metrics['os']:.2f}",
                f"{metrics['ne']:.3f}",
            )
        )

    overall = mean_metrics(rows)
    output_rows.append(
        (
            "OVERALL",
            str(len(rows)),
            f"{100.0 * overall['success']:.2f}",
            f"{100.0 * overall['spl']:.2f}",
            f"{100.0 * overall['os']:.2f}",
            f"{overall['ne']:.3f}",
        )
    )

    headers = ("Scene", "Episodes", "SR (%)", "SPL (%)", "OS (%)", "NE")
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in output_rows))
        for index in range(len(headers))
    ]

    def render(values: Iterable[str]) -> str:
        values = tuple(values)
        return "  ".join(
            value.ljust(widths[index]) if index == 0 else value.rjust(widths[index])
            for index, value in enumerate(values)
        )

    separator = "  ".join("-" * width for width in widths)
    return "\n".join([render(headers), separator, *(render(row) for row in output_rows)])


def format_resource_table(rows: Sequence[Mapping[str, object]]) -> str:
    """Report means of episode steps and episode-level peak memory measurements."""
    grouped: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["scene_id"])].append(row)

    def summarize(scene_rows: Sequence[Mapping[str, object]]) -> Tuple[str, ...]:
        values = []
        formats = (".3f", ".2f", ".2f", ".2f")
        for name, number_format in zip(RESOURCE_METRICS, formats):
            if not all(name in row for row in scene_rows):
                values.append("n/a")
                continue
            mean = sum(float(row[name]) for row in scene_rows) / len(scene_rows)
            values.append(format(mean, number_format))
        return tuple(values)

    output_rows = [
        (scene_id, str(len(grouped[scene_id])), *summarize(grouped[scene_id]))
        for scene_id in sorted(grouped)
    ]
    output_rows.append(("OVERALL", str(len(rows)), *summarize(rows)))

    headers = (
        "Scene",
        "Episodes",
        "Mean steps",
        "Mean peak CUDA MB",
        "Mean peak reserved MB",
        "Mean peak VGGT KV MB",
    )
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in output_rows))
        for index in range(len(headers))
    ]

    def render(values: Iterable[str]) -> str:
        values = tuple(values)
        return "  ".join(
            value.ljust(widths[index]) if index == 0 else value.rjust(widths[index])
            for index, value in enumerate(values)
        )

    separator = "  ".join("-" * width for width in widths)
    return "\n".join([render(headers), separator, *(render(row) for row in output_rows)])


def _has_step_weights(rows: Sequence[Mapping[str, object]]) -> bool:
    """True when every row carries a positive step count usable as a weight."""
    return all("steps" in row and float(row["steps"]) > 0 for row in rows)


def _mean_per_step(
    rows: Sequence[Mapping[str, object]], field: str, step_weighted: bool
) -> float | None:
    """Mean of a per-episode average.

    ``mean_step_ms``/``mean_vggt_ms`` are already averaged *within* an episode, so a
    plain mean over episodes would weight a 27-step episode the same as a 54-step one.
    Weighting by step count recovers the true per-step latency.
    """
    present = [row for row in rows if field in row]
    if not present or len(present) != len(rows):
        return None
    if not step_weighted:
        return sum(float(row[field]) for row in present) / len(present)
    total = sum(float(row["steps"]) for row in present)
    if total <= 0:
        return None
    return sum(float(row[field]) * float(row["steps"]) for row in present) / total


def _sum_field(rows: Sequence[Mapping[str, object]], field: str) -> float | None:
    present = [row for row in rows if field in row]
    if not present or len(present) != len(rows):
        return None
    return sum(float(row[field]) for row in present)


def format_timing_table(rows: Sequence[Mapping[str, object]]) -> Tuple[str, bool]:
    """Per-scene and overall step latency, VGGT share, and control frequency."""
    grouped: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["scene_id"])].append(row)

    step_weighted = _has_step_weights(rows)

    def summarize(scene_rows: Sequence[Mapping[str, object]]) -> Tuple[str, ...]:
        steps = _sum_field(scene_rows, "steps")
        step_ms = _mean_per_step(scene_rows, "mean_step_ms", step_weighted)
        vggt_ms = _mean_per_step(scene_rows, "mean_vggt_ms", step_weighted)
        wall_s = _sum_field(scene_rows, "episode_time_s")
        episode_s = None if wall_s is None else wall_s / len(scene_rows)
        share = (
            f"{100.0 * vggt_ms / step_ms:.1f}"
            if step_ms and vggt_ms is not None and step_ms > 0
            else "n/a"
        )
        hz = f"{1000.0 / step_ms:.2f}" if step_ms and step_ms > 0 else "n/a"
        return (
            "n/a" if steps is None else f"{steps:.0f}",
            "n/a" if step_ms is None else f"{step_ms:.1f}",
            "n/a" if vggt_ms is None else f"{vggt_ms:.1f}",
            share,
            hz,
            "n/a" if episode_s is None else f"{episode_s:.1f}",
            "n/a" if wall_s is None else f"{wall_s:.1f}",
        )

    output_rows = [
        (scene_id, str(len(grouped[scene_id])), *summarize(grouped[scene_id]))
        for scene_id in sorted(grouped)
    ]
    output_rows.append(("OVERALL", str(len(rows)), *summarize(rows)))

    headers = (
        "Scene",
        "Episodes",
        "Steps",
        "Step ms",
        "VGGT ms",
        "VGGT %",
        "Ctrl Hz",
        "Episode s",
        "Wall s",
    )
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in output_rows))
        for index in range(len(headers))
    ]

    def render(values: Iterable[str]) -> str:
        values = tuple(values)
        return "  ".join(
            value.ljust(widths[index]) if index == 0 else value.rjust(widths[index])
            for index, value in enumerate(values)
        )

    separator = "  ".join("-" * width for width in widths)
    table = "\n".join([render(headers), separator, *(render(row) for row in output_rows)])
    return table, step_weighted


def format_memory_summary(rows: Sequence[Mapping[str, object]]) -> str:
    """Render the overall episode-level peak-memory mean and maximum."""
    metrics = (
        ("VGGT KV MB", "peak_vggt_kv_mb"),
        ("Alloc MB", "peak_alloc_mb"),
        ("Reserved MB", "peak_reserved_mb"),
    )
    output = []
    for label, field_name in metrics:
        values = [float(row[field_name]) for row in rows if field_name in row]
        if not values:
            output.append(f"{label:<12}: mean n/a, max n/a")
            continue
        output.append(
            f"{label:<12}: mean {sum(values) / len(values):.2f}, "
            f"max {max(values):.2f}"
        )
    return "\n".join(output)


def main() -> int:
    args = parse_args()
    try:
        rows, stats = load_episode_rows(args.result_path, strict=args.strict)
        if not rows:
            raise ValueError("no valid episode rows found")
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(format_table(rows))
    timing_table, step_weighted = format_timing_table(rows)
    weighting = (
        "step-weighted, so scenes with longer episodes count proportionally"
        if step_weighted
        else "episode-weighted fallback: some rows have no usable step count"
    )
    print(f"\nStep timing ({weighting}):")
    print(timing_table)
    print("\nEpisode-level efficiency metrics (mean within each scene):")
    print(format_resource_table(rows))
    print("\nOverall memory metrics (episode-level peaks):")
    print(format_memory_summary(rows))
    print(
        "\n"
        f"Unique episodes: {len(rows)} | "
        f"raw episode rows: {stats['episode_rows']} | "
        f"duplicates removed: {stats['duplicate_rows']} | "
        f"summary rows ignored: {stats['summary_rows']} | "
        f"invalid rows skipped: {stats['invalid_rows']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
