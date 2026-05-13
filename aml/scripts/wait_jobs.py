"""Wait until one or more AML pipeline jobs reach a terminal state.

Polls each job until status in {Completed, Failed, Canceled} and prints a
one-line status row each poll cycle.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone

from azure.ai.ml import MLClient
from azure.identity import AzureCliCredential


DEFAULT_SUBSCRIPTION_ID = "c7d74d79-1ca2-4d95-a534-783b00cbf117"
DEFAULT_RESOURCE_GROUP = "shahra-rg"
DEFAULT_WORKSPACE = "shahra-workspace"

TERMINAL_STATES = {"Completed", "Failed", "Canceled", "Cancelled", "NotResponding"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", nargs="+", required=True, help="One or more job names to wait on.")
    parser.add_argument("--subscription_id", default=DEFAULT_SUBSCRIPTION_ID)
    parser.add_argument("--resource_group", default=DEFAULT_RESOURCE_GROUP)
    parser.add_argument("--workspace_name", default=DEFAULT_WORKSPACE)
    parser.add_argument("--poll_seconds", type=int, default=30)
    parser.add_argument("--timeout_seconds", type=int, default=3600)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    ml_client = MLClient(
        credential=AzureCliCredential(),
        subscription_id=args.subscription_id,
        resource_group_name=args.resource_group,
        workspace_name=args.workspace_name,
    )

    deadline = time.time() + args.timeout_seconds
    statuses: dict[str, str] = {name: "Unknown" for name in args.jobs}

    while True:
        any_pending = False
        for name in args.jobs:
            if statuses[name] in TERMINAL_STATES:
                continue
            job = ml_client.jobs.get(name)
            statuses[name] = getattr(job, "status", "Unknown") or "Unknown"
            if statuses[name] not in TERMINAL_STATES:
                any_pending = True
        now = datetime.now(timezone.utc).strftime("%H:%M:%S")
        row = "  ".join(f"{name}={status}" for name, status in statuses.items())
        print(f"[wait {now}] {row}")
        if not any_pending:
            print("[wait] all jobs terminal.")
            failed = [n for n, s in statuses.items() if s != "Completed"]
            return 0 if not failed else 2
        if time.time() > deadline:
            print(f"[wait] timed out after {args.timeout_seconds}s.")
            return 1
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    sys.exit(main())
