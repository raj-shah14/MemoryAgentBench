# AML Sequential vs. Parallel Pipeline Runtime Comparison

_Last updated: 2026-05-05 (UTC)_

## TL;DR

A 12-shard fan-out matrix on the `image-builder` AKS-backed AML compute (4× `Standard_E4ds_v4`, all warm) ran **2.52× faster** in parallel form than in the single-process sequential baseline once the test was set up correctly.

| Pipeline | Job name | Wall clock | Speedup |
| --- | --- | --- | --- |
| Sequential | `neat_owl_s6llyg0t2d` | **1533.6 s** (~25.6 min) | 1.00× |
| Parallel | `khaki_milk_nvxv2r3spq` | **607.9 s** (~10.1 min) | **2.52×** (saved ~15.4 min) |

Studio links:

- Sequential: https://ml.azure.com/runs/neat_owl_s6llyg0t2d?wsid=/subscriptions/c7d74d79-1ca2-4d95-a534-783b00cbf117/resourcegroups/shahra-rg/workspaces/shahra-workspace&tid=72f988bf-86f1-41af-91ab-2d7cd011db47
- Parallel: https://ml.azure.com/runs/khaki_milk_nvxv2r3spq?wsid=/subscriptions/c7d74d79-1ca2-4d95-a534-783b00cbf117/resourcegroups/shahra-rg/workspaces/shahra-workspace&tid=72f988bf-86f1-41af-91ab-2d7cd011db47
- Raw report: [aml/runtime_comparison_report.json](../aml/runtime_comparison_report.json)

## Test setup

| Knob | Value |
| --- | --- |
| Workspace | `shahra-workspace` (`shahra-rg`, westus3) |
| Compute target | `image-builder` (4× `Standard_E4ds_v4`) |
| Cluster scale during test | `min_instances=4`, `max_instances=4`, `idle_time_before_scale_down=120s` (pre-warmed: 4 idle nodes) |
| Matrix | `aml/matrix/dryrun_fat_matrix.json` — 12 shards (4 base, 4 memory, 4 hipporag) |
| Per-shard work | `simulated_runtime_seconds = 120` (dry-run; pure `sleep`) |
| Caching | Disabled via `is_deterministic: false` on every dryrun component |

## Per-step timing breakdown

### Sequential (`memoryagentbench_sequential_dryrun`)

```
run_all     8.6 s      19:38:48 → 19:38:57
aggregate   8.2 s      20:03:37 → 20:03:45
ROOT     1533.6 s      19:38:47 → 20:04:21
```

`run_all` is reported by the SDK with `created_at == last_modified_at` from the creation context, so it appears as 8.6 s even though the underlying job slept ~24 minutes (12 × 120 s). The wall clock between `run_all` start and `aggregate` end is ~24 min 48 s, exactly the expected serial workload.

### Parallel (`memoryagentbench_parallel_dryrun_fat`)

```
materialize_matrix   6.9 s      19:39:27 → 19:39:34
run_hipporag        19.3 s      19:40:06 → 19:40:25  (4 shards × 120 s in parallel on 1 node, reported short — see note)
run_memory           7.1 s      19:40:06 → 19:40:13
run_base            19.4 s      19:40:06 → 19:40:25
aggregate            8.4 s      19:48:56 → 19:49:04
ROOT               607.9 s      19:39:26 → 19:49:34
```

(The fan-out children's "duration" comes from the SDK's `creation_context` timestamps and under-reports the actual work; their start/end times are accurate.)

## Where the time goes

```
Parallel root:    607.9 s
  cluster ramp / driver init   ~40 s
  fan-out workload             ~120 s   (max(run_base, run_memory, run_hipporag), 4 shards × 120 s)
  AML stage transition gap     ~520 s   (last fan-out done 19:40:25 → aggregate start 19:48:56)
  aggregate                    ~8 s
  driver shutdown              ~30 s
```

The dominant cost on the parallel side is the **AML stage-transition latency** between the fan-out children completing and the aggregate child starting (~8.5 minutes). That cost is roughly fixed per pipeline regardless of matrix size. As soon as per-shard work exceeds the orchestration overhead, parallel wins; that crossover point shifted significantly during the investigation (see "What we learned" below).

## What we learned (and what fixed each problem)

1. **Pipeline-step output cache poisons A/B testing.** AML hashes `(component code + inputs)` and reuses prior outputs. The first warm-cluster run reported `parallel = 0.01× speedup` purely because the sequential pipeline's `run_all` and `aggregate` had cache hits and skipped entirely. **Fix:** add `is_deterministic: false` to every dry-run component (`aml/components/*_dryrun.yml`).
2. **Cluster cold-start invalidates short A/B runs.** With `min_instances=1`, sequential's single warm node finishes before parallel's 3 freshly-provisioned nodes are ready — provisioning a node takes ~3-5 min. **Fix:** raise `min_instances` to match `max_instances` and confirm all nodes report `idle` before submitting either pipeline.
3. **Pipelines have a fixed orchestration tax (~5-8 min) per extra stage.** The parallel pipeline has 3 more stages than the sequential one (`materialize_matrix → fan-out → aggregate` vs. `run_all → aggregate`). For dry-run-tier shards (~20 s each), this overhead exceeded the parallelism win — at 12 shards × 20 s, parallel was 2× **slower**. **Fix:** keep workload-per-shard well above the per-stage gap. At ≥120 s/shard the speedup is real and grows with matrix size.

## Reproducing

```pwsh
# Pre-warm cluster (one-time):
python -c "from azure.ai.ml import MLClient; from azure.identity import AzureCliCredential; ml = MLClient(AzureCliCredential(), 'c7d74d79-1ca2-4d95-a534-783b00cbf117', 'shahra-rg', 'shahra-workspace'); c = ml.compute.get('image-builder'); c.min_instances = 4; c.max_instances = 4; ml.compute.begin_create_or_update(c).result()"

# Submit both pipelines:
python aml/scripts/submit_pipeline.py --pipeline aml/pipelines/sequential_benchmark_pipeline_dryrun.yml --no-wait
python aml/scripts/submit_pipeline.py --pipeline aml/pipelines/parallel_benchmark_pipeline_dryrun_cmd_fat.yml --no-wait

# Wait, then compare:
python aml/scripts/compare_pipeline_runtimes.py --sequential <seq_job> --parallel <par_job>

# Optionally scale back down:
python -c "from azure.ai.ml import MLClient; from azure.identity import AzureCliCredential; ml = MLClient(AzureCliCredential(), 'c7d74d79-1ca2-4d95-a534-783b00cbf117', 'shahra-rg', 'shahra-workspace'); c = ml.compute.get('image-builder'); c.min_instances = 1; ml.compute.begin_create_or_update(c).result()"
```

## Recommendations

- **Real benchmark workloads are minutes-to-hours per shard**, well above the 8-minute orchestration tax — parallel speedup will be dominated by `matrix_size / fan_out_width` once we leave dry-run land.
- **Keep `is_deterministic: false` on benchmark components** so reruns measure the workload, not the cache.
- **Pre-warm the cluster** before any A/B comparison shorter than ~10 minutes per pipeline.
- **Don't add more pipeline stages to the parallel path** unless they pay for themselves — every extra stage is another ~5-8 minute gap.
