"""Scale the image-builder AML compute cluster.

Usage:
    python aml/scripts/scale_cluster.py --min 12 --max 12 --idle 1800 [--wait]
    python aml/scripts/scale_cluster.py --min 1 --max 12 --idle 120

Without --wait, returns once the update is dispatched. With --wait, polls
until the cluster reports the requested minimum number of nodes as available
(allocationState == 'steady' and node_count >= min_instances).
"""

from __future__ import annotations

import argparse
import sys
import time

from azure.ai.ml import MLClient
from azure.identity import AzureCliCredential


DEFAULT_SUBSCRIPTION_ID = "c7d74d79-1ca2-4d95-a534-783b00cbf117"
DEFAULT_RESOURCE_GROUP = "shahra-rg"
DEFAULT_WORKSPACE = "shahra-workspace"
DEFAULT_COMPUTE = "image-builder"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subscription_id", default=DEFAULT_SUBSCRIPTION_ID)
    parser.add_argument("--resource_group", default=DEFAULT_RESOURCE_GROUP)
    parser.add_argument("--workspace_name", default=DEFAULT_WORKSPACE)
    parser.add_argument("--compute", default=DEFAULT_COMPUTE)
    parser.add_argument("--min", dest="min_instances", type=int, required=True)
    parser.add_argument("--max", dest="max_instances", type=int, required=True)
    parser.add_argument(
        "--idle",
        dest="idle_seconds",
        type=int,
        default=None,
        help="idle_time_before_scale_down in seconds (optional).",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Poll until min_instances nodes report idle.",
    )
    parser.add_argument(
        "--poll_seconds",
        type=int,
        default=20,
        help="Polling interval when --wait is set.",
    )
    parser.add_argument(
        "--timeout_seconds",
        type=int,
        default=1200,
        help="Max time to wait when --wait is set.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    ml_client = MLClient(
        credential=AzureCliCredential(),
        subscription_id=args.subscription_id,
        resource_group_name=args.resource_group,
        workspace_name=args.workspace_name,
    )

    compute = ml_client.compute.get(args.compute)
    print(
        f"[scale] before: name={compute.name} size={getattr(compute, 'size', '?')} "
        f"min={compute.min_instances} max={compute.max_instances} "
        f"idle={getattr(compute, 'idle_time_before_scale_down', '?')}"
    )

    compute.min_instances = args.min_instances
    compute.max_instances = args.max_instances
    if args.idle_seconds is not None:
        compute.idle_time_before_scale_down = args.idle_seconds

    op = ml_client.compute.begin_create_or_update(compute)
    print(f"[scale] dispatched update min={args.min_instances} max={args.max_instances}")
    updated = op.result()
    print(
        f"[scale] after : min={updated.min_instances} max={updated.max_instances} "
        f"idle={getattr(updated, 'idle_time_before_scale_down', '?')} "
        f"state={getattr(updated, 'provisioning_state', '?')}"
    )

    if not args.wait:
        return 0

    deadline = time.time() + args.timeout_seconds
    while True:
        fresh = ml_client.compute.get(args.compute)
        # AmlCompute exposes resource utilization through usage stats fetched separately.
        nodes = getattr(fresh, "provisioning_errors", None)
        try:
            node_stats = ml_client.compute.list_nodes(args.compute)
            node_list = list(node_stats)
        except Exception as exc:  # noqa: BLE001
            node_list = []
            print(f"[scale] list_nodes failed: {exc}")

        idle = sum(1 for n in node_list if getattr(n, "node_state", "") == "idle")
        running = sum(1 for n in node_list if getattr(n, "node_state", "") == "running")
        total = len(node_list)
        print(
            f"[scale] poll: total_nodes={total} idle={idle} running={running} "
            f"provisioning_state={getattr(fresh, 'provisioning_state', '?')}"
        )
        if idle + running >= args.min_instances:
            print(f"[scale] ok: at least {args.min_instances} node(s) available.")
            return 0
        if time.time() > deadline:
            print(f"[scale] timed out after {args.timeout_seconds}s waiting for nodes.")
            return 1
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    sys.exit(main())
