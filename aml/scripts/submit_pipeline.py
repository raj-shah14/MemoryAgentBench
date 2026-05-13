"""Register MemoryAgentBench AML environments and submit the parallel benchmark pipeline.

This script uses the Azure ML Python SDK (azure-ai-ml) so it works without the
`az ml` CLI extension.

Usage:
    python aml/scripts/submit_pipeline.py --register-environments \
        --environments dryrun \
        --pipeline aml/pipelines/parallel_benchmark_pipeline_dryrun.yml \
        --experiment memoryagentbench_parallel_dryrun

Default workspace targets the shared shahra-workspace; override via flags.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from azure.ai.ml import MLClient, load_environment, load_job
from azure.identity import AzureCliCredential


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SUBSCRIPTION_ID = "c7d74d79-1ca2-4d95-a534-783b00cbf117"
DEFAULT_RESOURCE_GROUP = "shahra-rg"
DEFAULT_WORKSPACE = "shahra-workspace"

ENVIRONMENT_FILES = {
    "base": REPO_ROOT / "aml/environments/mabench-base.yml",
    "memory": REPO_ROOT / "aml/environments/mabench-memory.yml",
    "hipporag": REPO_ROOT / "aml/environments/mabench-hipporag.yml",
    "dryrun": REPO_ROOT / "aml/environments/mabench-dryrun.yml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subscription_id", default=DEFAULT_SUBSCRIPTION_ID)
    parser.add_argument("--resource_group", default=DEFAULT_RESOURCE_GROUP)
    parser.add_argument("--workspace_name", default=DEFAULT_WORKSPACE)
    parser.add_argument(
        "--environments",
        nargs="*",
        default=[],
        choices=sorted(ENVIRONMENT_FILES.keys()),
        help="Subset of environments to register before submission.",
    )
    parser.add_argument(
        "--register-environments",
        dest="register_environments",
        action="store_true",
        help="Register the selected environments via create_or_update.",
    )
    parser.add_argument(
        "--pipeline",
        default=None,
        help="Path to a pipelineJob YAML to submit. If omitted, submission is skipped.",
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help="Optional experiment name override for the submitted pipeline.",
    )
    parser.add_argument(
        "--display_name",
        default=None,
        help="Optional display name override for the submitted pipeline.",
    )
    parser.add_argument(
        "--no-wait",
        dest="wait",
        action="store_false",
        help="Submit and exit without polling the job for completion.",
    )
    parser.set_defaults(wait=True)
    return parser.parse_args()


def build_client(args: argparse.Namespace) -> MLClient:
    return MLClient(
        credential=AzureCliCredential(),
        subscription_id=args.subscription_id,
        resource_group_name=args.resource_group,
        workspace_name=args.workspace_name,
    )


def register_environments(ml_client: MLClient, environment_keys: list[str]) -> None:
    if not environment_keys:
        print("[register] no environments selected, skipping")
        return

    for key in environment_keys:
        env_path = ENVIRONMENT_FILES[key]
        if not env_path.exists():
            raise FileNotFoundError(f"Environment file not found: {env_path}")

        print(f"[register] {key} <- {env_path.relative_to(REPO_ROOT)}")
        env = load_environment(source=str(env_path))
        result = ml_client.environments.create_or_update(env)
        print(f"  -> {result.name}:{result.version} (id={result.id})")


def submit_pipeline(
    ml_client: MLClient,
    pipeline_path: Path,
    experiment_name: str | None,
    display_name: str | None,
    wait_for_completion: bool,
) -> None:
    print(f"[submit] pipeline <- {pipeline_path.relative_to(REPO_ROOT)}")
    job = load_job(source=str(pipeline_path))

    if experiment_name:
        job.experiment_name = experiment_name
    if display_name:
        job.display_name = display_name

    submitted = ml_client.jobs.create_or_update(job)
    print(f"[submit] job name        : {submitted.name}")
    print(f"[submit] job studio url  : {submitted.studio_url}")

    if not wait_for_completion:
        return

    print("[submit] streaming job until terminal state...")
    try:
        ml_client.jobs.stream(submitted.name)
    except Exception as exc:  # noqa: BLE001 - best-effort streaming
        print(f"[submit] stream interrupted: {exc}")

    final = ml_client.jobs.get(submitted.name)
    print(f"[submit] final status    : {final.status}")
    print(f"[submit] studio url      : {final.studio_url}")


def main() -> int:
    args = parse_args()
    ml_client = build_client(args)

    if args.register_environments:
        register_environments(ml_client, args.environments)

    if args.pipeline:
        submit_pipeline(
            ml_client,
            Path(args.pipeline).resolve(),
            args.experiment,
            args.display_name,
            args.wait,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
