# Teams Message — AML parallel benchmarking pipeline result

> Paste this into Teams as-is; the **bold** and links render correctly.

---

🚀 **AML parallel benchmark pipeline is now 2.52× faster than the sequential baseline** (12-shard dry-run matrix, `image-builder` cluster, 4× `Standard_E4ds_v4` warm).

| Run | Wall clock | Job |
| --- | --- | --- |
| Sequential | **1533.6 s** (~25.6 min) | [neat_owl_s6llyg0t2d](https://ml.azure.com/runs/neat_owl_s6llyg0t2d?wsid=/subscriptions/c7d74d79-1ca2-4d95-a534-783b00cbf117/resourcegroups/shahra-rg/workspaces/shahra-workspace&tid=72f988bf-86f1-41af-91ab-2d7cd011db47) |
| Parallel | **607.9 s** (~10.1 min) | [khaki_milk_nvxv2r3spq](https://ml.azure.com/runs/khaki_milk_nvxv2r3spq?wsid=/subscriptions/c7d74d79-1ca2-4d95-a534-783b00cbf117/resourcegroups/shahra-rg/workspaces/shahra-workspace&tid=72f988bf-86f1-41af-91ab-2d7cd011db47) |

**Saved ~15.4 min** on a 12-shard matrix; gap widens linearly as matrix grows.

Two gotchas we hit and fixed along the way:
1. **AML step-output cache** silently skipped sequential's children → set `is_deterministic: false` on every dryrun component.
2. **Cluster cold-start** of 3 extra nodes wasted ~5 min on the parallel side → bumped `min_instances` to 4 and waited for all 4 nodes idle before submitting.

Full write-up: `docs/parallel-runtime-comparison.md`. Branch: `feature/aml-safe-parallel-benchmarks`.
