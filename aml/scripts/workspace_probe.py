"""One-off probe to verify access to the AML workspace and list compute targets."""

from azure.ai.ml import MLClient
from azure.identity import AzureCliCredential


SUBSCRIPTION_ID = "c7d74d79-1ca2-4d95-a534-783b00cbf117"
RESOURCE_GROUP = "shahra-rg"
WORKSPACE_NAME = "shahra-workspace"


def main() -> None:
    ml_client = MLClient(
        credential=AzureCliCredential(),
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
    )

    workspace = ml_client.workspaces.get(WORKSPACE_NAME)
    print(f"workspace : {workspace.name}")
    print(f"location  : {workspace.location}")
    print(f"id        : {workspace.id}")

    print("--- compute targets ---")
    for compute in ml_client.compute.list():
        provisioning = getattr(compute, "provisioning_state", "?")
        size = getattr(compute, "size", None) or getattr(compute, "vm_size", None) or ""
        nodes = getattr(compute, "max_instances", None)
        print(f"  - {compute.name:30s} type={compute.type:25s} size={size:18s} max_nodes={nodes} state={provisioning}")


if __name__ == "__main__":
    main()
