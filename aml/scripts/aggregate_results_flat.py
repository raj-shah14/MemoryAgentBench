"""Aggregate bundles produced by the sequential baseline.

Unlike the profile-aware aggregator, this reads a single flat folder of
per-run bundles and infers the profile from each metadata.json.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundles", required=True)
    parser.add_argument("--report_dir", required=True)
    return parser.parse_args()


def discover_bundles(root: Path):
    if not root.exists():
        return []

    bundles = []
    for metadata_path in root.rglob("metadata.json"):
        bundle_root = metadata_path.parent
        summary_path = bundle_root / "summary.json"
        results_path = bundle_root / "results.json"

        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        summary = {}
        if summary_path.exists():
            with summary_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)

        bundles.append(
            {
                "environment_profile": metadata.get("environment_profile", "unknown"),
                "bundle_root": str(bundle_root),
                "metadata": metadata,
                "summary": summary,
                "results_path": str(results_path) if results_path.exists() else None,
            }
        )

    return bundles


def main():
    args = parse_args()
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    bundle_entries = discover_bundles(Path(args.bundles))

    summary_payload = {"total_runs": len(bundle_entries), "runs": bundle_entries}
    (report_dir / "combined_runs.json").write_text(
        json.dumps(summary_payload, indent=2), encoding="utf-8"
    )

    fieldnames = [
        "run_id",
        "environment_profile",
        "agent_name",
        "dataset",
        "sub_dataset",
        "exact_match",
        "f1",
        "total_queries",
        "simulated_runtime_seconds",
        "bundle_root",
    ]
    with (report_dir / "leaderboard.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for entry in bundle_entries:
            metadata = entry["metadata"]
            summary = entry["summary"]
            writer.writerow(
                {
                    "run_id": metadata.get("run_id"),
                    "environment_profile": entry["environment_profile"],
                    "agent_name": metadata.get("agent_name"),
                    "dataset": metadata.get("dataset"),
                    "sub_dataset": metadata.get("sub_dataset"),
                    "exact_match": summary.get("averaged_metrics", {}).get("exact_match"),
                    "f1": summary.get("averaged_metrics", {}).get("f1"),
                    "total_queries": summary.get("total_queries"),
                    "simulated_runtime_seconds": summary.get("simulated_runtime_seconds"),
                    "bundle_root": entry["bundle_root"],
                }
            )


if __name__ == "__main__":
    main()
