# Safe Parallel Benchmarking Design

## Goal

Run benchmark workloads in parallel without changing per-workload semantics or allowing backends to collide on local state.

## Current Constraint

The benchmark harness is sequential inside a workload. Context memorization, query execution, and per-config shell wrappers all run in order. That behavior should stay intact because it preserves the benchmark contract and keeps result interpretation simple.

The unsafe part is cross-workload parallelism. Several backends write to shared local paths today:

- Letta reads and writes a shared `.letta` directory and SQLite file.
- Cognee defaults to shared `.data_storage` and `.cognee_system` roots.
- Retrieved-context debug artifacts are written under shared `./outputs/rag_retrieved/...` paths.
- Agent state is stored under shared `./agents/...` paths.

## Parallelism Boundary

Parallelize only across independent benchmark tuples:

- backend
- dataset
- model
- ablation settings
- seed or run id

Each child run remains internally sequential.

## Isolation Model

Every child run gets a unique `run_id` and a unique `state_root`.

Under that root, the runner creates job-scoped writable locations for:

- agent state snapshots
- Letta state
- Cognee data root
- Cognee system root
- retrieved-context artifacts
- results bundle files

This makes child jobs idempotent and replayable. Re-running the same `run_id` can safely resume or overwrite inside the same bundle without affecting other runs.

## Result Contract

Each child publishes one result bundle instead of writing a shared flat JSON file.

Recommended bundle layout:

- `results.json`: full per-query output
- `summary.json`: averaged metrics and counts
- `metadata.json`: run id, config paths, backend, environment profile, timestamps, state roots
- `retrieval/`: optional retrieved-context artifacts

The reducer step scans bundles and produces a combined leaderboard plus a machine-readable manifest.

## AMLv2 Orchestration

Use an AMLv2 parent pipeline with three phases:

1. Materialize matrix
   - Input: one JSON matrix spec
   - Output: one shard file per child workload
   - Split shards by `environment_profile`

2. Fan out child runs
   - Use AML parallel jobs with `mini_batch_size: 1`
   - One shard file maps to one child run
   - Bound concurrency with `resources.instance_count` and `max_concurrency_per_instance`
   - Keep `max_concurrency_per_instance: 1` on GPU compute unless a backend is proven safe at higher density

3. Reduce artifacts
   - Read all child bundles
   - Emit combined JSON and CSV leaderboard outputs

## Environment Strategy

Do not force all backends into one AML environment. Use separate environments at least for:

- `base`: long-context and non-conflicting RAG workloads
- `memory`: Letta, Cognee, Mem0, Zep workloads
- `hipporag`: workloads that need the older OpenAI pin noted in the repo README

This keeps dependency conflicts out of the scheduler and avoids brittle runtime installs.

## Operational Defaults

- Use managed identity for AML job data access.
- Keep code and dataset inputs read-only.
- Make artifact outputs upload-only AML outputs.
- Treat matrix shards as immutable input records.
- Retry failed child runs at the shard level, not inside the benchmark loop.
- Tag every child artifact with backend, dataset, model, and `run_id`.

## Immediate Implementation Scope

The repo changes in this branch should do four things:

1. Parameterize writable paths with `run_id` and `state_root`.
2. Emit one result bundle per child run.
3. Add AML templates for matrix materialization, bounded parallel fan-out, and reduction.
4. Separate AML environments for conflicting backend groups.