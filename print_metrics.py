#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path


MAIN_METRICS = (
    ("success", "SR", ".4f"),
    ("spl", "SPL", ".4f"),
    ("os", "OS", ".4f"),
    ("ne", "NE", ".4f"),
    ("steps", "Steps", ".2f"),
)

EXTRA_NUMERIC_FIELDS = (
    ("peak_vggt_kv_mb", "VGGT KV MB", ".2f"),
    ("peak_alloc_mb", "Alloc MB", ".2f"),
    ("peak_reserved_mb", "Reserved MB", ".2f"),
    ("mean_step_ms", "Step ms", ".2f"),
    ("mean_vggt_ms", "VGGT ms", ".2f"),
    ("episode_time_s", "Episode s", ".2f"),
)

AGGREGATE_KEYS = {"sucs_all", "spls_all", "oss_all", "ones_all"}
EPISODE_KEYS = {"scene_id", "episode_id", "success", "spl", "os", "ne"}


def read_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc
    return records


def mean(values):
    return sum(values) / len(values) if values else 0.0


def numeric_values(rows, key):
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def calculate_metrics(rows):
    metrics = {
        "episodes": len(rows),
        "episode_min": min((row["episode_id"] for row in rows), default=None),
        "episode_max": max((row["episode_id"] for row in rows), default=None),
    }

    for key, _, _ in MAIN_METRICS:
        values = numeric_values(rows, key)
        if values:
            metrics[key] = mean(values)

    extras = {}
    for key, _, _ in EXTRA_NUMERIC_FIELDS:
        values = numeric_values(rows, key)
        if values:
            extras[key] = {
                "mean": mean(values),
                "max": max(values),
                "count": len(values),
            }
    metrics["extras"] = extras

    return metrics


def format_value(value, spec):
    return format(value, spec)


def print_metric_block(title, metrics, show_extras):
    print(title)
    print(f"  Episodes : {metrics['episodes']}")
    if metrics["episode_min"] is not None:
        print(f"  Episode  : {metrics['episode_min']}..{metrics['episode_max']}")

    for key, label, spec in MAIN_METRICS:
        if key in metrics:
            print(f"  {label:<8} : {format_value(metrics[key], spec)}")

    if show_extras and metrics["extras"]:
        print("  Extra")
        for key, label, spec in EXTRA_NUMERIC_FIELDS:
            if key not in metrics["extras"]:
                continue
            stat = metrics["extras"][key]
            mean_value = format_value(stat["mean"], spec)
            if key.startswith("peak_"):
                max_value = format_value(stat["max"], spec)
                print(f"    {label:<11}: mean {mean_value}, max {max_value}")
            else:
                print(f"    {label:<11}: mean {mean_value}")


def collect_episode_records(records):
    episodes = []
    aggregate_records = []

    for record in records:
        if EPISODE_KEYS.issubset(record):
            episodes.append(record)
        elif AGGREGATE_KEYS.issubset(record):
            aggregate_records.append(record)

    return episodes, aggregate_records


def print_aggregate_record(record):
    print("Latest aggregate record in file")
    print(f"  Episodes : {record.get('length', 'N/A')}")
    print(f"  SR       : {float(record['sucs_all']):.4f}")
    print(f"  SPL      : {float(record['spls_all']):.4f}")
    print(f"  OS       : {float(record['oss_all']):.4f}")
    print(f"  NE       : {float(record['ones_all']):.4f}")


def print_metrics(episodes, aggregate_records, show_by_scene, show_extras):
    if not episodes:
        print("No episode-level metrics found.")
        if aggregate_records:
            print()
            print_aggregate_record(aggregate_records[-1])
        return

    print_metric_block("Overall metrics", calculate_metrics(episodes), show_extras)

    if aggregate_records:
        print()
        print_aggregate_record(aggregate_records[-1])

    if show_by_scene:
        by_scene = defaultdict(list)
        for row in episodes:
            by_scene[row.get("scene_id", "unknown")].append(row)

        print("\nMetrics by scene")
        for scene_id in sorted(by_scene):
            print_metric_block(
                f"\nScene {scene_id}",
                calculate_metrics(by_scene[scene_id]),
                show_extras,
            )


def main():
    parser = argparse.ArgumentParser(
        description="Read JanusVLN result.json JSONL file and print evaluation metrics."
    )
    parser.add_argument(
        "result_path",
        nargs="?",
        default=Path(__file__).with_name("result.json"),
        type=Path,
        help="Path to result.json. Default: result.json next to this script.",
    )
    parser.add_argument(
        "--overall-only",
        action="store_true",
        help="Only print overall metrics, without grouping by scene_id.",
    )
    parser.add_argument(
        "--no-extra",
        action="store_true",
        help="Hide optional runtime and memory fields such as peak_alloc_mb.",
    )
    args = parser.parse_args()

    records = read_jsonl(args.result_path)
    episodes, aggregate_records = collect_episode_records(records)
    print_metrics(
        episodes,
        aggregate_records,
        show_by_scene=not args.overall_only,
        show_extras=not args.no_extra,
    )


if __name__ == "__main__":
    main()
