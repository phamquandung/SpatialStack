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
            for name in RESOURCE_METRICS:
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
