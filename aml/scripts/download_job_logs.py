"""Download logs/outputs for an AML job to a local directory."""

import argparse
import sys
from pathlib import Path

from azure.ai.ml import MLClient
from azure.identity import AzureCliCredential


SUBSCRIPTION_ID = "c7d74d79-1ca2-4d95-a534-783b00cbf117"
RESOURCE_GROUP = "shahra-rg"
WORKSPACE_NAME = "shahra-workspace"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_name", required=True)
    parser.add_argument("--download_path", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ml_client = MLClient(
        credential=AzureCliCredential(),
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
    )

    target = Path(args.download_path)
    target.mkdir(parents=True, exist_ok=True)
    print(f"downloading job {args.job_name} -> {target}")
    ml_client.jobs.download(name=args.job_name, download_path=str(target), all=True)

    for path in sorted(target.rglob("*")):
        if path.is_file():
            print(path.relative_to(target))

    return 0


if __name__ == "__main__":
    sys.exit(main())
