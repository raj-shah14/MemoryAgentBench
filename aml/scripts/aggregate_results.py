import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_artifacts", required=True)
    parser.add_argument("--memory_artifacts", required=True)
    parser.add_argument("--hipporag_artifacts", required=True)
    parser.add_argument("--report_dir", required=True)
    return parser.parse_args()


def discover_bundles(root: Path, environment_profile: str):
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
                "environment_profile": environment_profile,
                "bundle_root": str(bundle_root),
                "metadata": metadata,
                "summary": summary,
                "results_path": str(results_path) if results_path.exists() else None,
            }
        )

    return bundles


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_leaderboard(path: Path, bundle_entries):
    fieldnames = [
        "run_id",
        "environment_profile",
        "agent_name",
        "dataset",
        "sub_dataset",
        "exact_match",
        "f1",
        "total_queries",
        "bundle_root",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
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
                    "bundle_root": entry["bundle_root"],
                }
            )


def main():
    args = parse_args()
    report_dir = Path(args.report_dir)

    bundle_entries = []
    bundle_entries.extend(discover_bundles(Path(args.base_artifacts), "base"))
    bundle_entries.extend(discover_bundles(Path(args.memory_artifacts), "memory"))
    bundle_entries.extend(discover_bundles(Path(args.hipporag_artifacts), "hipporag"))

    summary_payload = {
        "total_runs": len(bundle_entries),
        "runs": bundle_entries,
    }

    write_json(report_dir / "combined_runs.json", summary_payload)
    write_leaderboard(report_dir / "leaderboard.csv", bundle_entries)


if __name__ == "__main__":
    main()