"""Real-LLM-call workload for the sequential vs parallel pipeline smoke benchmark.

This replaces the dry-run `time.sleep` in [run_parallel_benchmark.py] /
[run_sequential_benchmark.py] with a small number of *real* Azure OpenAI chat
completion calls per shard. The point is to measure pipeline orchestration
overhead under realistic per-shard wall times (real network I/O, real auth,
real cluster activity) without the multi-day yak-shave of getting every
memory-system backend (mem0, hipporag, ...) wired against a fresh Foundry
endpoint.

Each shard JSON should look like:
    {
      "name": "real-base-01",
      "run_id": "real-base-01",
      "environment_profile": "base",
      "real_completion_calls": 3,
      "prompt": "optional prompt override"
    }

Per-shard outputs match the existing aggregator's contract:
    <bundle_root>/summary.json
    <bundle_root>/metadata.json
    <bundle_root>/results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_PROMPT = (
    "Summarize the following sentence in one short clause: "
    "The quick brown fox jumps over the lazy dog while the AML pipeline "
    "fans out shards across the cluster."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shards_dir",
        required=False,
        default=None,
        help="Folder of pre-split *.json shards (parallel pipeline path).",
    )
    parser.add_argument(
        "--matrix_spec",
        required=False,
        default=None,
        help="Raw matrix JSON (sequential pipeline path).",
    )
    parser.add_argument(
        "--shard_index",
        type=int,
        default=None,
        help="When set together with --matrix_spec, run only this shard index. Used by per-shard parallel fan-out.",
    )
    parser.add_argument(
        "--out_root",
        required=True,
        help="Where to write per-shard bundle folders.",
    )
    parser.add_argument(
        "--environment_profile",
        default="real",
        help="Profile tag written into per-shard metadata.",
    )
    parser.add_argument(
        "--default_completion_calls",
        type=int,
        default=3,
        help="Number of real chat completions per shard if the shard JSON does not override it.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=64,
        help="Per-call max output tokens (kept small to bound smoke cost).",
    )
    return parser.parse_args()


def _load_jobs_from_matrix(matrix_path: Path) -> list[dict]:
    with matrix_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("jobs"), list):
        return payload["jobs"]
    raise ValueError("Matrix spec must be a list or an object with a 'jobs' list.")


def _load_jobs_from_dir(shards_dir: Path) -> list[dict]:
    jobs: list[dict] = []
    for path in sorted(shards_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            jobs.append(json.load(handle))
    return jobs


def _build_client():
    """Build an Azure OpenAI client from environment variables.

    The submit script is expected to inject these into every command job:
      - AZURE_OPENAI_ENDPOINT
      - AZURE_OPENAI_API_KEY
      - AZURE_OPENAI_API_VERSION
    """
    from openai import AzureOpenAI  # imported lazily so a dry import works without the package

    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION")
    if not endpoint or not api_key or not api_version:
        raise RuntimeError(
            "Missing AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY / AZURE_OPENAI_API_VERSION; "
            "the submit script must inject these into the job environment."
        )
    return AzureOpenAI(api_key=api_key, api_version=api_version, azure_endpoint=endpoint)


def _resolve_deployment() -> str:
    """The chat deployment name on the Foundry account."""
    return os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4-mini")


def run_shard(client: Any, deployment: str, job: dict, bundle_root: Path, args: argparse.Namespace) -> dict:
    bundle_root.mkdir(parents=True, exist_ok=True)
    run_id = job.get("run_id") or job.get("name") or "unnamed-shard"
    profile = job.get("environment_profile", args.environment_profile)
    num_calls = int(job.get("real_completion_calls", args.default_completion_calls))
    prompt = job.get("prompt", DEFAULT_PROMPT)

    # Lazy import so the file can be parsed without `openai` installed.
    from openai import APIStatusError, BadRequestError  # noqa: WPS433

    started = time.time()
    per_call_seconds: list[float] = []
    sample_outputs: list[str] = []
    call_errors: list[dict] = []
    failed_calls = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for index in range(num_calls):
        call_started = time.time()
        try:
            response = client.chat.completions.create(
                model=deployment,
                messages=[
                    {"role": "system", "content": "You are a concise smoke-test responder."},
                    {"role": "user", "content": f"[shard={run_id} call={index + 1}/{num_calls}] {prompt}"},
                ],
                max_completion_tokens=args.max_tokens,
            )
        except (BadRequestError, APIStatusError) as exc:
            # Treat individual call failures (content-filter false positives, transient 4xx/5xx)
            # as recorded-but-non-fatal so a single hiccup does not invalidate the shard run.
            elapsed = time.time() - call_started
            per_call_seconds.append(elapsed)
            sample_outputs.append("")
            failed_calls += 1
            err_record = {
                "call_index": index,
                "error_type": type(exc).__name__,
                "status_code": getattr(exc, "status_code", None),
                "code": getattr(exc, "code", None),
                "message": str(exc)[:500],
            }
            call_errors.append(err_record)
            print(
                f"[real_smoke_workload] shard={run_id} call={index + 1}/{num_calls} "
                f"non-fatal error: {err_record['error_type']} "
                f"status={err_record['status_code']} code={err_record['code']}"
            )
            continue

        elapsed = time.time() - call_started
        per_call_seconds.append(elapsed)

        try:
            sample_outputs.append(response.choices[0].message.content or "")
        except Exception:  # noqa: BLE001 - response shape is provider-defined
            sample_outputs.append("")

        usage = getattr(response, "usage", None)
        if usage is not None:
            total_prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            total_completion_tokens += getattr(usage, "completion_tokens", 0) or 0

    elapsed_total = time.time() - started

    summary = {
        "run_id": run_id,
        "environment_profile": profile,
        "real_completion_calls": num_calls,
        "failed_calls": failed_calls,
        "elapsed_seconds": elapsed_total,
        "per_call_seconds": per_call_seconds,
        "deployment": deployment,
        "prompt_tokens_total": total_prompt_tokens,
        "completion_tokens_total": total_completion_tokens,
        "averaged_metrics": {
            "elapsed_seconds": elapsed_total,
            "avg_per_call_seconds": (sum(per_call_seconds) / num_calls) if num_calls else 0.0,
        },
        "time_cost": [elapsed_total],
        "real_work": True,
        "dry_run": False,
        "call_errors": call_errors,
    }
    metadata = {
        "run_id": run_id,
        "environment_profile": profile,
        "matrix_shard": run_id,
        "agent_name": "real_smoke_workload",
        "dataset": job.get("dataset", "real_smoke"),
        "sub_dataset": job.get("sub_dataset", "real_smoke"),
        "artifact_root": str(bundle_root),
        "state_root": str(bundle_root / "_state"),
        "real_work": True,
        "dry_run": False,
        "hostname": os.environ.get("HOSTNAME", ""),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    results = {
        "data": [
            {"call_index": i, "elapsed_seconds": s, "sample_output": o}
            for i, (s, o) in enumerate(zip(per_call_seconds, sample_outputs))
        ],
        "averaged_metrics": summary["averaged_metrics"],
        "real_work": True,
    }

    (bundle_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (bundle_root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (bundle_root / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    return summary


def main() -> int:
    args = parse_args()
    if not args.shards_dir and not args.matrix_spec:
        raise SystemExit("Either --shards_dir or --matrix_spec is required.")

    if args.shards_dir:
        jobs = _load_jobs_from_dir(Path(args.shards_dir))
        source = f"shards_dir={args.shards_dir}"
    else:
        jobs = _load_jobs_from_matrix(Path(args.matrix_spec))
        source = f"matrix_spec={args.matrix_spec}"

    if args.shard_index is not None:
        if not (0 <= args.shard_index < len(jobs)):
            raise SystemExit(
                f"--shard_index {args.shard_index} out of range (matrix has {len(jobs)} shard(s))"
            )
        jobs = [jobs[args.shard_index]]
        source += f" shard_index={args.shard_index}"

    print(f"[real_smoke_workload] processing {len(jobs)} shard(s) from {source}")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    client = _build_client()
    deployment = _resolve_deployment()
    print(f"[real_smoke_workload] deployment={deployment}")

    started = time.time()
    summaries: list[dict] = []
    for index, job in enumerate(jobs, start=1):
        run_id = job.get("run_id") or job.get("name") or f"shard-{index}"
        bundle_root = out_root / run_id
        print(f"[{index}/{len(jobs)}] real {run_id}  -> {bundle_root}")
        summary = run_shard(client, deployment, job, bundle_root, args)
        summaries.append(summary)
        print(
            f"[{index}/{len(jobs)}] done  elapsed={summary['elapsed_seconds']:.2f}s "
            f"calls={summary['real_completion_calls']} "
            f"prompt_tokens={summary['prompt_tokens_total']} "
            f"completion_tokens={summary['completion_tokens_total']}"
        )

    elapsed = time.time() - started
    overall = {
        "total_shards": len(jobs),
        "elapsed_seconds": elapsed,
        "out_root": str(out_root),
        "summaries": summaries,
    }
    (out_root / "_real_smoke_summary.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
    print(f"[real_smoke_workload] done in {elapsed:.2f}s ({overall['total_shards']} shards)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
