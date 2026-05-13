import argparse
import json
import re
from pathlib import Path


PROFILE_DIR_NAMES = {
    "base": "base",
    "memory": "memory",
    "hipporag": "hipporag",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix_spec", required=True)
    parser.add_argument("--base_output_dir", required=True)
    parser.add_argument("--memory_output_dir", required=True)
    parser.add_argument("--hipporag_output_dir", required=True)
    return parser.parse_args()


def load_jobs(matrix_spec_path: Path):
    with matrix_spec_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("jobs"), list):
        return payload["jobs"]
    raise ValueError("Matrix spec must be a list or an object with a 'jobs' list.")


def infer_profile(job):
    explicit_profile = job.get("environment_profile")
    if explicit_profile:
        return explicit_profile

    backend = (job.get("backend") or "").lower()
    agent_config = (job.get("agent_config") or "").lower()

    if "hippo" in backend or "hippo" in agent_config:
        return "hipporag"
    if any(token in backend for token in ["letta", "cognee", "mem0", "zep"]):
        return "memory"
    if any(token in agent_config for token in ["letta", "cognee", "mem0", "zep"]):
        return "memory"
    return "base"


def sanitize_name(value):
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-")
    return cleaned or "job"


def write_job(output_dir: Path, index: int, job: dict):
    run_id = job.get("run_id") or sanitize_name(job.get("name", f"job-{index:03d}"))
    materialized = dict(job)
    materialized["run_id"] = run_id

    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / f"{index:03d}_{sanitize_name(run_id)}.json"
    with target_path.open("w", encoding="utf-8") as handle:
        json.dump(materialized, handle, indent=2)


def ensure_non_empty(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    sentinel_path = output_dir / "000___empty__.json"
    if any(path.suffix == ".json" for path in output_dir.iterdir()):
        return
    with sentinel_path.open("w", encoding="utf-8") as handle:
        json.dump({"skip": True, "reason": "no jobs for this environment profile"}, handle)


def main():
    args = parse_args()
    jobs = load_jobs(Path(args.matrix_spec))

    output_dirs = {
        "base": Path(args.base_output_dir),
        "memory": Path(args.memory_output_dir),
        "hipporag": Path(args.hipporag_output_dir),
    }

    for index, job in enumerate(jobs, start=1):
        profile = infer_profile(job)
        if profile not in PROFILE_DIR_NAMES:
            raise ValueError(f"Unsupported environment profile: {profile}")
        write_job(output_dirs[profile], index, job)

    for output_dir in output_dirs.values():
        ensure_non_empty(output_dir)


if __name__ == "__main__":
    main()