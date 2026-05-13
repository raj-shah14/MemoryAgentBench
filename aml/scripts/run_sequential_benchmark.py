"""Sequential baseline that runs every matrix shard in a single AML command job.

This intentionally bypasses the profile-level fan-out used by the parallel
pipeline so we can measure the runtime advantage of parallelizing across
environment profiles.

Each shard is materialized into a per-run bundle under <out_root>/<run_id>/.
Bundles are written flat (no profile sub-folders); the flat aggregator
reconstructs the profile grouping by reading metadata.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Reuse the per-shard bundle writer from the parallel runner so the sequential
# and parallel paths emit byte-identical bundle layouts.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import run_parallel_benchmark as parallel_runner  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix_spec", required=True, help="Raw matrix JSON file (not pre-split).")
    parser.add_argument("--repo_root", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--dry_run", action="store_true", default=False)
    return parser.parse_args()


def load_jobs(matrix_path: Path):
    with matrix_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("jobs"), list):
        return payload["jobs"]
    raise ValueError("Matrix spec must be a list or an object with a 'jobs' list.")


def main() -> int:
    args = parse_args()
    matrix_path = Path(args.matrix_spec)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    jobs = load_jobs(matrix_path)
    print(f"[run_sequential_benchmark] processing {len(jobs)} shard(s) sequentially from {matrix_path}")

    parallel_runner.ARGS = argparse.Namespace(
        repo_root=args.repo_root,
        artifacts_root=str(out_root),
        environment_profile="sequential",
        dry_run=args.dry_run,
        shards_dir=None,
    )

    started = time.time()
    for index, job in enumerate(jobs, start=1):
        run_id = job["run_id"]
        bundle_root = out_root / run_id
        shard_name = f"{index:03d}_{run_id}.json"

        if job.get("skip"):
            print(f"[{index}/{len(jobs)}] skip {run_id}: {job.get('reason', '')}")
            continue

        if args.dry_run or job.get("dry_run"):
            parallel_runner.ARGS.environment_profile = job.get("environment_profile", "sequential")
            parallel_runner.write_dry_run_bundle(bundle_root, shard_name, job)
            print(f"[{index}/{len(jobs)}] dry_run  {run_id}  -> {bundle_root}")
        else:
            raise NotImplementedError(
                "Sequential baseline currently only supports --dry_run mode for runtime "
                "comparisons. Use the parallel pipeline for real workloads."
            )

    elapsed = time.time() - started
    summary = {
        "total_shards": len(jobs),
        "elapsed_seconds": elapsed,
        "out_root": str(out_root),
    }
    (out_root / "_sequential_run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"[run_sequential_benchmark] done in {elapsed:.2f}s ({summary['total_shards']} shards)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
