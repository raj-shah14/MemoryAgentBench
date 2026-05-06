"""Submit the per-shard parallel pipeline programmatically.

The YAML pipeline schema doesn't natively support "fan out one command job per
matrix entry" without the parallel job type (which has its own auth quirks),
so this script builds the topology with the SDK's @dsl.pipeline decorator:

    [matrix_spec] --> [shard_00, shard_01, ..., shard_N-1] --> [aggregate]

Each shard_i job runs `real_smoke_workload.py --matrix_spec ... --shard_index i`,
so all N shards execute concurrently (one per cluster node) instead of the
3-way per-profile fan-out used by parallel_benchmark_pipeline_real.yml.

Usage (from repo root, with AOAI envs in shell):

    python aml/scripts/submit_real_pipeline_pershard.py --register --matrix aml/matrix/real_big_matrix.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from azure.ai.ml import Input, MLClient, Output, command, dsl, load_environment
from azure.ai.ml.entities import Environment
from azure.identity import AzureCliCredential


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SUBSCRIPTION_ID = "c7d74d79-1ca2-4d95-a534-783b00cbf117"
DEFAULT_RESOURCE_GROUP = "shahra-rg"
DEFAULT_WORKSPACE = "shahra-workspace"

ENV_FILE = REPO_ROOT / "aml/environments/mabench-real-smoke.yml"
SCRIPTS_DIR = REPO_ROOT / "aml/scripts"
DEFAULT_COMPUTE = "image-builder"
DEFAULT_ENV = "azureml:mabench-real-smoke:1"

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
    parser.add_argument("--matrix", required=True, help="Path to matrix JSON file (one shard per entry).")
    parser.add_argument("--register", action="store_true", help="Register/refresh the real-smoke env first.")
    parser.add_argument(
        "--experiment",
        default="memoryagentbench_runtime_compare_real_big",
        help="Experiment name to attach the submitted pipeline to.",
    )
    parser.add_argument(
        "--display_name",
        default="memoryagentbench_parallel_real_big_pershard",
        help="Display name for the submitted pipeline.",
    )
    parser.add_argument(
        "--state_file",
        default=str(REPO_ROOT / "aml" / "real_big_pipeline_submission.json"),
        help="Where to write the submitted job name.",
    )
    parser.add_argument("--wait", action="store_true", help="Stream the job until terminal state.")
    return parser.parse_args()


def collect_env_vars() -> dict[str, str]:
    missing = [name for name in INJECTED_ENV_VARS if name not in os.environ and name != "AZURE_OPENAI_DEPLOYMENT"]
    if missing:
        raise SystemExit("Missing required env vars: " + ", ".join(missing))
    return {name: os.environ[name] for name in INJECTED_ENV_VARS if name in os.environ}


def load_matrix(matrix_path: Path) -> list[dict]:
    with matrix_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("jobs"), list):
        return payload["jobs"]
    raise ValueError("Matrix must be a list or {jobs: [...]}.")


def build_shard_command(shard_index: int, environment_profile: str, env_vars: dict[str, str]):
    """Build a CommandComponent that runs a single shard from the matrix file."""
    return command(
        name=f"run_shard_{shard_index:02d}",
        display_name=f"shard_{shard_index:02d}_{environment_profile}",
        code=str(SCRIPTS_DIR),
        environment=DEFAULT_ENV,
        compute=DEFAULT_COMPUTE,
        inputs={"matrix_spec": Input(type="uri_file")},
        outputs={"bundles": Output(type="uri_folder")},
        environment_variables=env_vars,
        is_deterministic=False,
        command=(
            "python real_smoke_workload.py "
            "--matrix_spec ${{inputs.matrix_spec}} "
            f"--shard_index {shard_index} "
            f"--environment_profile {environment_profile} "
            "--out_root ${{outputs.bundles}}"
        ),
    )


def build_aggregate_command(num_shards: int, env_vars: dict[str, str]):
    """Build the aggregator that fans in all per-shard bundle folders."""
    inputs = {f"shard_{i:02d}": Input(type="uri_folder") for i in range(num_shards)}
    # The flat aggregator scans a single folder, so we copy each per-shard folder
    # into a unified `--bundles` folder before invoking it.
    cmd_parts = ["mkdir -p ${{outputs.report_dir}}/_inputs"]
    for i in range(num_shards):
        cmd_parts.append(
            f"cp -r ${{{{inputs.shard_{i:02d}}}}}/* ${{{{outputs.report_dir}}}}/_inputs/ 2>/dev/null || true"
        )
    cmd_parts.append(
        "python aggregate_results_flat.py --bundles ${{outputs.report_dir}}/_inputs --report_dir ${{outputs.report_dir}}"
    )
    return command(
        name="aggregate",
        display_name="aggregate",
        code=str(SCRIPTS_DIR),
        environment=DEFAULT_ENV,
        compute=DEFAULT_COMPUTE,
        inputs=inputs,
        outputs={"report_dir": Output(type="uri_folder")},
        environment_variables=env_vars,
        is_deterministic=False,
        command=" && ".join(cmd_parts),
    )


def build_pipeline_definition(matrix_jobs: list[dict], env_vars: dict[str, str]):
    num_shards = len(matrix_jobs)

    shard_components = [
        build_shard_command(
            shard_index=i,
            environment_profile=str(matrix_jobs[i].get("environment_profile", "real")),
            env_vars=env_vars,
        )
        for i in range(num_shards)
    ]
    aggregate_component = build_aggregate_command(num_shards, env_vars)

    @dsl.pipeline(
        name="memoryagentbench_parallel_real_big_pershard",
        description=(
            f"Per-shard parallel fan-out: {num_shards} command jobs, one per matrix entry. "
            "Each shard runs real Azure OpenAI completions and writes its own bundle; "
            "the aggregator fans in all bundles into a single report."
        ),
        default_compute=DEFAULT_COMPUTE,
    )
    def pipeline_func(matrix_spec):
        shard_outputs = {}
        for i in range(num_shards):
            step = shard_components[i](matrix_spec=matrix_spec)
            shard_outputs[f"shard_{i:02d}"] = step.outputs.bundles
        agg = aggregate_component(**shard_outputs)
        return {"aggregate_report": agg.outputs.report_dir}

    return pipeline_func


def main() -> int:
    args = parse_args()
    env_vars = collect_env_vars()

    ml_client = MLClient(
        credential=AzureCliCredential(),
        subscription_id=args.subscription_id,
        resource_group_name=args.resource_group,
        workspace_name=args.workspace_name,
    )

    if args.register:
        env = load_environment(source=str(ENV_FILE))
        result = ml_client.environments.create_or_update(env)
        print(f"[register] {result.name}:{result.version}")

    matrix_path = Path(args.matrix).resolve()
    matrix_jobs = load_matrix(matrix_path)
    print(f"[submit] matrix: {matrix_path.relative_to(REPO_ROOT)}  shards: {len(matrix_jobs)}")
    print(f"[submit] injecting env keys: {sorted(env_vars.keys())}")

    pipeline_func = build_pipeline_definition(matrix_jobs, env_vars)
    pipeline_job = pipeline_func(matrix_spec=Input(type="uri_file", path=str(matrix_path)))
    pipeline_job.experiment_name = args.experiment
    pipeline_job.display_name = args.display_name
    pipeline_job.outputs["aggregate_report"].mode = "upload"  # type: ignore[union-attr]

    submitted = ml_client.jobs.create_or_update(pipeline_job)
    print(f"[submit] job name        : {submitted.name}")
    print(f"[submit] job studio url  : {submitted.studio_url}")

    state = {
        "submitted_at": datetime.utcnow().isoformat() + "Z",
        "matrix": str(matrix_path),
        "num_shards": len(matrix_jobs),
        "job": {
            "name": submitted.name,
            "status": getattr(submitted, "status", "Submitted"),
            "studio_url": submitted.studio_url,
        },
    }
    state_path = Path(args.state_file)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(f"[submit] wrote submission state -> {state_path.relative_to(REPO_ROOT)}")

    if args.wait:
        try:
            ml_client.jobs.stream(submitted.name)
        except Exception as exc:  # noqa: BLE001
            print(f"[submit] stream interrupted: {exc}")
        final = ml_client.jobs.get(submitted.name)
        print(f"[submit] final status    : {final.status}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
