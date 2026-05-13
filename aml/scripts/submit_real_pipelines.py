"""Register the real-smoke env and submit the real sequential + parallel pipelines.

This wraps `submit_pipeline.py` with two extras the real run needs:

1. **Env-var injection.** The Foundry/AOAI key cannot be written to the
   workspace key vault from this account, so the key is staged in the local
   shell as `AZURE_OPENAI_API_KEY` and this script injects it (plus
   `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_VERSION`, optional deployment
   override) into every command job in the loaded pipeline before submission.
   Nothing is committed.

2. **Submit-and-track both pipelines.** Sequential and parallel are submitted
   back-to-back and their job names are written to a small JSON state file so
   `compare_pipeline_runtimes.py` can be pointed at them.

Usage (from the repo root, with the AOAI envs already exported in the shell):

    python aml/scripts/submit_real_pipelines.py --register --wait
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

from azure.ai.ml import MLClient, load_environment, load_job
from azure.ai.ml.entities import CommandJob
from azure.identity import AzureCliCredential


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SUBSCRIPTION_ID = "c7d74d79-1ca2-4d95-a534-783b00cbf117"
DEFAULT_RESOURCE_GROUP = "shahra-rg"
DEFAULT_WORKSPACE = "shahra-workspace"

ENV_FILE = REPO_ROOT / "aml/environments/mabench-real-smoke.yml"

PIPELINES = {
    "sequential": REPO_ROOT / "aml/pipelines/sequential_benchmark_pipeline_real.yml",
    "parallel": REPO_ROOT / "aml/pipelines/parallel_benchmark_pipeline_real.yml",
}

# Env vars copied from the local shell into every command job at submission
# time. Nothing about the values is logged here -- only the keys.
INJECTED_ENV_VARS = (
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_OPENAI_DEPLOYMENT",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subscription_id", default=DEFAULT_SUBSCRIPTION_ID)
    parser.add_argument("--resource_group", default=DEFAULT_RESOURCE_GROUP)
    parser.add_argument("--workspace_name", default=DEFAULT_WORKSPACE)
    parser.add_argument(
        "--register",
        action="store_true",
        help="Register/refresh azureml:mabench-real-smoke:1 before submitting.",
    )
    parser.add_argument(
        "--only",
        choices=sorted(PIPELINES.keys()),
        default=None,
        help="Submit only one pipeline (sequential or parallel). Default: both.",
    )
    parser.add_argument(
        "--sequential_pipeline",
        default=None,
        help="Override path to the sequential pipeline YAML (defaults to the smoke variant).",
    )
    parser.add_argument(
        "--parallel_pipeline",
        default=None,
        help="Override path to the parallel pipeline YAML (defaults to the smoke variant).",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Stream each submitted job until it reaches a terminal state.",
    )
    parser.add_argument(
        "--state_file",
        default=str(REPO_ROOT / "aml" / "real_pipeline_submissions.json"),
        help="Where to write the submitted job names for follow-up comparison.",
    )
    return parser.parse_args()


def build_client(args: argparse.Namespace) -> MLClient:
    return MLClient(
        credential=AzureCliCredential(),
        subscription_id=args.subscription_id,
        resource_group_name=args.resource_group,
        workspace_name=args.workspace_name,
    )


def collect_env_vars() -> dict[str, str]:
    missing = [name for name in INJECTED_ENV_VARS if name not in os.environ]
    # AZURE_OPENAI_DEPLOYMENT is optional -- the workload script defaults to gpt-5.4-mini.
    required_missing = [name for name in missing if name != "AZURE_OPENAI_DEPLOYMENT"]
    if required_missing:
        raise SystemExit(
            "Missing required environment variables: "
            + ", ".join(required_missing)
            + ". Export them in your shell before running this script."
        )
    return {name: os.environ[name] for name in INJECTED_ENV_VARS if name in os.environ}


def inject_env_vars(pipeline_job, env_vars: dict[str, str]) -> int:
    """Walk every immediate child job of the pipeline and set environment_variables.

    Returns the number of jobs touched.
    """
    touched = 0
    jobs_attr = getattr(pipeline_job, "jobs", None) or {}
    for job_key, job in jobs_attr.items():
        # CommandJob and similar expose `environment_variables` as a writable dict.
        existing = getattr(job, "environment_variables", None) or {}
        merged = dict(existing)
        merged.update(env_vars)
        try:
            job.environment_variables = merged  # type: ignore[attr-defined]
            touched += 1
        except AttributeError:
            print(f"[inject] WARNING: job '{job_key}' ({type(job).__name__}) does not accept environment_variables; skipped.")
    return touched


def register_environment(ml_client: MLClient) -> None:
    if not ENV_FILE.exists():
        raise FileNotFoundError(f"Environment file not found: {ENV_FILE}")
    print(f"[register] {ENV_FILE.relative_to(REPO_ROOT)}")
    env = load_environment(source=str(ENV_FILE))
    result = ml_client.environments.create_or_update(env)
    print(f"  -> {result.name}:{result.version} (id={result.id})")


def submit_one(
    ml_client: MLClient,
    pipeline_path: Path,
    env_vars: dict[str, str],
    wait: bool,
) -> dict:
    print(f"[submit] pipeline <- {pipeline_path.relative_to(REPO_ROOT)}")
    job = load_job(source=str(pipeline_path))
    touched = inject_env_vars(job, env_vars)
    print(f"[submit] injected {len(env_vars)} env var(s) into {touched} child job(s)")
    submitted = ml_client.jobs.create_or_update(job)
    print(f"[submit] job name        : {submitted.name}")
    print(f"[submit] job studio url  : {submitted.studio_url}")

    if wait:
        try:
            ml_client.jobs.stream(submitted.name)
        except Exception as exc:  # noqa: BLE001
            print(f"[submit] stream interrupted: {exc}")
        final = ml_client.jobs.get(submitted.name)
        print(f"[submit] final status    : {final.status}")
        return {"name": submitted.name, "status": final.status, "studio_url": final.studio_url}

    return {"name": submitted.name, "status": "Submitted", "studio_url": submitted.studio_url}


def main() -> int:
    args = parse_args()
    ml_client = build_client(args)
    env_vars = collect_env_vars()
    print(f"[submit_real_pipelines] injecting env keys: {sorted(env_vars.keys())}")

    if args.register:
        register_environment(ml_client)

    targets: Iterable[str] = (args.only,) if args.only else ("sequential", "parallel")

    pipeline_paths = dict(PIPELINES)
    if args.sequential_pipeline:
        pipeline_paths["sequential"] = Path(args.sequential_pipeline).resolve()
    if args.parallel_pipeline:
        pipeline_paths["parallel"] = Path(args.parallel_pipeline).resolve()

    state = {
        "submitted_at": datetime.utcnow().isoformat() + "Z",
        "subscription_id": args.subscription_id,
        "resource_group": args.resource_group,
        "workspace_name": args.workspace_name,
        "jobs": {},
    }
    for key in targets:
        info = submit_one(ml_client, pipeline_paths[key], env_vars, args.wait)
        state["jobs"][key] = info

    state_path = Path(args.state_file).resolve()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    try:
        display_path = state_path.relative_to(REPO_ROOT)
    except ValueError:
        display_path = state_path
    print(f"[submit_real_pipelines] wrote submission state -> {display_path}")

    if "sequential" in state["jobs"] and "parallel" in state["jobs"]:
        print("\nNext step:")
        print(
            f"  python aml/scripts/compare_pipeline_runtimes.py "
            f"--sequential {state['jobs']['sequential']['name']} "
            f"--parallel {state['jobs']['parallel']['name']}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
