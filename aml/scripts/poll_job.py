"""Poll an existing AML job until it reaches a terminal state."""

import argparse
import sys
import time

from azure.ai.ml import MLClient
from azure.identity import AzureCliCredential


SUBSCRIPTION_ID = "c7d74d79-1ca2-4d95-a534-783b00cbf117"
RESOURCE_GROUP = "shahra-rg"
WORKSPACE_NAME = "shahra-workspace"

TERMINAL_STATES = {"Completed", "Failed", "Canceled", "NotResponding"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_name", required=True)
    parser.add_argument("--poll_interval", type=int, default=20)
    parser.add_argument("--max_poll_seconds", type=int, default=1800)
    parser.add_argument("--show_children", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ml_client = MLClient(
        credential=AzureCliCredential(),
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
    )

    deadline = time.time() + args.max_poll_seconds
    last_status = None
    while time.time() < deadline:
        job = ml_client.jobs.get(args.job_name)
        status = job.status
        if status != last_status:
            print(f"[{time.strftime('%H:%M:%S')}] status={status}")
            last_status = status
        if status in TERMINAL_STATES:
            break
        time.sleep(args.poll_interval)

    final = ml_client.jobs.get(args.job_name)
    print(f"final status      : {final.status}")
    print(f"studio url        : {final.studio_url}")

    if args.show_children:
        try:
            children = list(ml_client.jobs.list(parent_job_name=args.job_name))
            for child in children:
                print(f"  child {child.name:40s} status={child.status:10s} display={getattr(child, 'display_name', '')}")
        except Exception as exc:  # noqa: BLE001
            print(f"failed to list children: {exc}")

    return 0 if final.status == "Completed" else 1


if __name__ == "__main__":
    sys.exit(main())
