import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ARGS = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", required=True)
    parser.add_argument("--artifacts_root", required=True)
    parser.add_argument("--environment_profile", required=True)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help="Skip launching main.py and emit a stub bundle for pipeline-shape validation only.",
    )
    parser.add_argument(
        "--shards_dir",
        default=None,
        help="When set, the script runs in standalone CLI mode and iterates every *.json shard in this folder instead of being driven by the AML parallel framework.",
    )
    return parser.parse_args()


def init():
    global ARGS
    ARGS = parse_args()


def load_job_spec(path_like):
    shard_path = Path(path_like)
    with shard_path.open("r", encoding="utf-8") as handle:
        return shard_path, json.load(handle)


def build_command(job, bundle_root: Path):
    command = [
        sys.executable,
        "main.py",
        "--agent_config",
        job["agent_config"],
        "--dataset_config",
        job["dataset_config"],
        "--run_id",
        job["run_id"],
        "--state_root",
        str(bundle_root / "_state"),
        "--artifact_root",
        str(bundle_root),
    ]

    if job.get("chunk_size_ablation"):
        command.extend(["--chunk_size_ablation", str(job["chunk_size_ablation"])])
    if job.get("max_test_queries_ablation"):
        command.extend(["--max_test_queries_ablation", str(job["max_test_queries_ablation"])])
    if job.get("force"):
        command.append("--force")

    return command


def write_child_metadata(bundle_root: Path, shard_name: str, job: dict, completed: subprocess.CompletedProcess):
    bundle_root.mkdir(parents=True, exist_ok=True)
    child_metadata = {
        "run_id": job["run_id"],
        "environment_profile": ARGS.environment_profile,
        "matrix_shard": shard_name,
        "agent_config": job["agent_config"],
        "dataset_config": job["dataset_config"],
        "return_code": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }
    metadata_path = bundle_root / "aml_child_job.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(child_metadata, handle, indent=2)


def write_dry_run_bundle(bundle_root: Path, shard_name: str, job: dict) -> None:
    bundle_root.mkdir(parents=True, exist_ok=True)

    simulated_runtime = float(job.get("simulated_runtime_seconds") or 0.0)
    if simulated_runtime > 0:
        time.sleep(simulated_runtime)

    summary = {
        "run_id": job["run_id"],
        "artifact_root": str(bundle_root),
        "results_path": str(bundle_root / "results.json"),
        "total_queries": 0,
        "time_cost": [simulated_runtime],
        "averaged_metrics": {"exact_match": 0.0, "f1": 0.0},
        "dry_run": True,
        "simulated_runtime_seconds": simulated_runtime,
    }
    metadata = {
        "run_id": job["run_id"],
        "environment_profile": ARGS.environment_profile,
        "matrix_shard": shard_name,
        "agent_config_path": job.get("agent_config"),
        "dataset_config_path": job.get("dataset_config"),
        "agent_name": job.get("backend") or job.get("agent_name") or "dry_run",
        "dataset": job.get("dataset") or "dry_run",
        "sub_dataset": job.get("sub_dataset") or "dry_run",
        "artifact_root": str(bundle_root),
        "state_root": str(bundle_root / "_state"),
        "dry_run": True,
        "hostname": os.environ.get("HOSTNAME", ""),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    results = {"data": [], "averaged_metrics": summary["averaged_metrics"], "dry_run": True}

    (bundle_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (bundle_root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (bundle_root / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


def run(mini_batch):
    if ARGS is None:
        init()

    repo_root = Path(ARGS.repo_root)
    artifacts_root = Path(ARGS.artifacts_root)
    results = []

    for item in mini_batch:
        shard_path, job = load_job_spec(item)

        if job.get("skip"):
            results.append({"status": "skipped", "reason": job.get("reason", "sentinel shard")})
            continue

        run_id = job["run_id"]
        bundle_root = artifacts_root / run_id
        bundle_root.mkdir(parents=True, exist_ok=True)

        if ARGS.dry_run or job.get("dry_run"):
            write_dry_run_bundle(bundle_root, shard_path.name, job)
            results.append({"status": "dry_run", "run_id": run_id, "bundle_root": str(bundle_root)})
            continue

        command = build_command(job, bundle_root)
        completed = subprocess.run(
            command,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        write_child_metadata(bundle_root, shard_path.name, job, completed)

        if completed.returncode != 0:
            raise RuntimeError(
                f"Benchmark child run failed for {run_id}.\nSTDOUT:\n{completed.stdout[-4000:]}\nSTDERR:\n{completed.stderr[-4000:]}"
            )

        results.append({"status": "completed", "run_id": run_id, "bundle_root": str(bundle_root)})

    return results


def main_cli():
    """Standalone CLI mode for use under a plain `command` job.

    Lists every *.json shard under --shards_dir and dispatches each one through
    the same per-shard logic used by the AML parallel runner.
    """
    init()
    if not ARGS.shards_dir:
        raise SystemExit("--shards_dir is required in CLI mode")

    shards_dir = Path(ARGS.shards_dir)
    shard_files = sorted(p for p in shards_dir.glob("*.json"))
    if not shard_files:
        print(f"[run_parallel_benchmark] no shards found in {shards_dir}; nothing to do.")
        return 0

    print(f"[run_parallel_benchmark] processing {len(shard_files)} shard(s) from {shards_dir}")
    outcomes = run([str(p) for p in shard_files])
    for outcome in outcomes:
        print(json.dumps(outcome))
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())