"""Offline smoke check for the run-scoped bundle output layout.

This harness exercises the production code paths for runtime-config setup and
result serialization without invoking any LLM, dataset, or backend service.
It is the deterministic fallback for environments that don't have API keys.

Run:
    python aml/scripts/smoke_check_bundle.py --run_id smoke-001 \
        --workdir ./runs/smoke-001
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from initialization import setup_configs_and_directories  # noqa: E402
from main import save_results_to_file  # noqa: E402


SAMPLE_AGENT_CONFIG = REPO_ROOT / "configs/agent_conf/RAG_Agents/gpt-4o-mini/Simple_rag_gpt-4o-mini-bm25.yaml"
SAMPLE_DATASET_CONFIG = REPO_ROOT / "configs/data_conf/Accurate_Retrieval/Ruler/QA/Ruler_qa1_197k.yaml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline smoke check for bundle layout")
    parser.add_argument("--run_id", default="smoke-001")
    parser.add_argument("--workdir", default=None,
                        help="Optional explicit artifact_root; default uses ./runs/<run_id>")
    parser.add_argument("--state_root", default=None,
                        help="Optional explicit state_root; default is <artifact_root>/_state")
    parser.add_argument("--agent_config", default=str(SAMPLE_AGENT_CONFIG))
    parser.add_argument("--dataset_config", default=str(SAMPLE_DATASET_CONFIG))
    return parser.parse_args()


def _build_cli_namespace(args: argparse.Namespace) -> SimpleNamespace:
    artifact_root = args.workdir or str(REPO_ROOT / "runs" / args.run_id)
    state_root = args.state_root or os.path.join(artifact_root, "_state")
    return SimpleNamespace(
        agent_config=args.agent_config,
        dataset_config=args.dataset_config,
        chunk_size_ablation=0,
        max_test_queries_ablation=0,
        force=False,
        run_id=args.run_id,
        state_root=state_root,
        artifact_root=artifact_root,
    )


def _make_synthetic_results():
    results = [
        {
            "query": "Who is the manager of the team?",
            "answer": "Alice",
            "output": "Alice",
            "input_len": 1024,
            "output_len": 4,
            "memory_construction_time": 0.0,
            "query_time_len": 0.12,
            "query_id": 0,
        }
    ]
    metrics = defaultdict(list)
    metrics["exact_match"].append(1.0)
    metrics["f1"].append(1.0)
    metrics["input_len"].append(1024)
    metrics["output_len"].append(4)
    metrics["query_time_len"].append(0.12)
    return results, metrics


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    args = _parse_args()
    cli_namespace = _build_cli_namespace(args)

    agent_config, dataset_config, runtime_config, output_path = setup_configs_and_directories(cli_namespace)

    _expect(os.path.isdir(runtime_config["artifact_root"]), "artifact_root not created")
    _expect(os.path.isdir(runtime_config["state_root"]), "state_root not created")
    _expect(os.path.isdir(runtime_config["agent_state_root"]), "agent_state_root not created")
    _expect(os.path.isdir(runtime_config["retrieval_artifacts_root"]), "retrieval_artifacts_root not created")
    _expect(os.path.isdir(runtime_config["letta_dir"]), "letta_dir not created")
    _expect(os.path.isdir(runtime_config["cognee_data_root"]), "cognee_data_root not created")

    _expect(runtime_config["run_id"] == args.run_id, "run_id not propagated")
    _expect(runtime_config["letta_dir"].startswith(runtime_config["state_root"]),
            "letta_dir not under state_root")

    results, metrics = _make_synthetic_results()
    save_results_to_file(
        output_path=output_path,
        agent_config=agent_config,
        dataset_config=dataset_config,
        runtime_config=runtime_config,
        results=results,
        metrics=metrics,
        time_cost_list=[0.5],
        start_time=0.0,
    )

    for path_key in ("results_path", "summary_path", "metadata_path"):
        path_value = runtime_config[path_key]
        _expect(os.path.isfile(path_value), f"{path_key} bundle file missing: {path_value}")

    with open(runtime_config["summary_path"], "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    with open(runtime_config["metadata_path"], "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    _expect(summary["run_id"] == args.run_id, "summary run_id mismatch")
    _expect(summary["total_queries"] == len(results), "summary total_queries mismatch")
    _expect("exact_match" in summary["averaged_metrics"], "summary missing averaged_metrics.exact_match")

    _expect(metadata["agent_name"] == agent_config["agent_name"], "metadata agent_name mismatch")
    _expect(metadata["dataset"] == dataset_config["dataset"], "metadata dataset mismatch")
    _expect(metadata["sub_dataset"] == dataset_config["sub_dataset"], "metadata sub_dataset mismatch")
    _expect(metadata["state_root"] == runtime_config["state_root"], "metadata state_root mismatch")
    _expect(metadata["artifact_root"] == runtime_config["artifact_root"], "metadata artifact_root mismatch")

    print("PASS: bundle layout valid")
    print(f"  artifact_root      = {runtime_config['artifact_root']}")
    print(f"  state_root         = {runtime_config['state_root']}")
    print(f"  results.json       = {runtime_config['results_path']}")
    print(f"  summary.json       = {runtime_config['summary_path']}")
    print(f"  metadata.json      = {runtime_config['metadata_path']}")
    print(f"  retrieval_root     = {runtime_config['retrieval_artifacts_root']}")
    print(f"  letta_dir          = {runtime_config['letta_dir']}")
    print(f"  cognee_data_root   = {runtime_config['cognee_data_root']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
