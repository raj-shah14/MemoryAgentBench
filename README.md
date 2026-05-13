# ⚙️ MemoryAgentBench: Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions

[Yuanzhe Hu](https://hust-ai-hyz.github.io), [Yu Wang](https://yuwang.us), [Julian McAuley](https://cseweb.ucsd.edu/~jmcauley/).

This project benchmarks agents with memory capabilities. Follow the steps below to set up your environment and install dependencies. 

[Full paper](https://arxiv.org/abs/2507.05257)


## 🧠 LongMemEval Overview

Four Core Competencies for Evaluation:
* Accurate Retrieval (AR)

* Test-Time Learning (TTL)

* Long-Range Understanding (LRU)

* Conflict Resolution (CR)


![Example Questions in MemoryAgentBench](assets/intro.png)

We collected and reformulated data from previous benchmarks and datasets. All data is split into chunks to simulate real multi-turn interaction scenarios—just like your daily conversations with an AI assistant. We also newly constructed two datasets **EventQA** and **FactConsolidation**.

Notably, the team adopted a "inject once, query multiple times" design philosophy—one long text corresponds to multiple questions, significantly improving evaluation efficiency.

## 🚧 Update
- [x] (Jan. 26th, 2026)
    Our paper is accepted by Fourteenth International Conference on Learning Representations (ICLR 2026). We will make some improvement for our current benchmark. 
      
- [x] (Sep. 28th, 2025)  
    We publish a new version of our paper. 

- [x] (Aug. 5th, 2025)  
    We optimized the ```template.py``` for better usage.

- [x] (July 22th, 2025)  
    We updated the ```Readme.md``` and release the code for ```longmemeval``` and ```infbench_sum```. They are needed to evaluate by using ```gpt-4o``` as a judge. 

    We change the ```uuid``` into ```qa_pair_id``` in our code.

    We updated the huggingface dataset slightly. 

- [x] (July 7th, 2025) 
    We released the code for reproducing the main experiment. 


- TODO List ✍️ .
    
    <del> [x] New Dataset in Long Range Understanding (LRU). </del>

    [] Leaderboard website for our benchmark.

    [] The code framework with separated front-end and back-end is easier to integrate with custom memory agents.

**🌟 More details (such as datasets collection) coming soon! 🌟**


## 🚀 Quick Setup

### 1. Create a Conda Environment

It’s recommended to use a dedicated conda environment for reproducibility:
```
conda create --name MABench python=3.10.16
```

### 2. Install Python Dependencies

```
pip install torch
pip install -r requirements.txt
pip install "numpy<2"
```
We did not include the `hipporag` in `requirements.txt` since the current version of `hipporag` will cause some conflicts on pacakge version. You can create another environment with hipporag instead.  

Sometimes you can try to supplement the lacked packages for `cognee` and `letta`. If you met some package related errors after installing `requirements.txt`. 
```
pip install letta
pip uninstall letta   
pip install cognee
pip uninstall cognee
```

## 📥 Data Download & API Settings

To use this project, you need to download the processed data files and place them in the correct directory.

### 1. Download the Data from HuggingFace 🤗 

- HuggingFace dataset [link](https://huggingface.co/datasets/ai-hyz/MemoryAgentBench). It can be automatically downloaded if you run the code directly. 

- Do not forget the `entity2id.json` for Movie Recommendation task.


### 2. Environment Variable Settings

To run this project, you need to configure your API keys and model settings in a `.env` file at the project root.

Create a `.env` file and add the following content, replacing the placeholder values with your actual API keys:

#### OpenAI API Keys

```
OPENAI_API_KEY= ###your_openai_api_key
```

#### Settings for Cognee
```
LLM_MODEL=gpt-4o-mini
LLM_API_KEY=  ###your_api_key
```

#### Other API Keys
```
Anthropic_API_KEY= ###your_anthropic_api
Google_API_KEY=    ###your_google_api
```

## 🏃‍♂️ Run Evaluation

Follow these steps to evaluate the benchmarking agent:


### Run Example Evaluation Command

You can run an evaluation using the following example command:

#### Long Context Agents
```
bash bash_files/eniac/run_memagent_longcontext.sh
```
- `--agent_config`: Path to the agent/model configuration file.
- `--dataset_config`: Path to the dataset configuration file.

#### Rag Agents and Agentic Memory Methods

```
bash bash_files/eniac/run_memagent_rag_agents.sh
```
#### Ablation Study for Chunk Size
```
bash bash_files/eniac/run_memagent_rag_agents_chunksize.sh
```

Remember that `hipporag (2.0.0a3)` reuqires `openai==1.58.1`, which may cause some latest OpenAI models could not be used in same environment. 


### Run LLM-based Metric Evaluation 

You can run an evaluation using the following example python files, you also need to set the configs

#### LongmemEval

```
python llm_based_eval/longmem_qa_evaluate.py
```

#### InfBench Summarization 
```
python llm_based_eval/summarization_evaluate.py
```

## 👍 Acknowledgement 

We thank the open-source code and datasets from RULER, InfBench, HELMET and LongmemEval.

## 📝 Citation 

We would appreciate it if you could cite the following paper if you found the repository useful for your work:
```
@article{hu2025evaluating,
  title={Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions},
  author={Hu, Yuanzhe and Wang, Yu and McAuley, Julian},
  journal={arXiv preprint arXiv:2507.05257},
  year={2025}
}
```
## ☁️ Running Benchmarks in Parallel on Azure Machine Learning

The default `main.py` driver is single-process and runs every (agent, dataset) pair sequentially. For larger sweeps we ship an Azure Machine Learning v2 pipeline layout under [aml/](aml/) that fans out per-shard benchmark runs across an AML compute cluster while keeping each run's artifacts isolated.

### Layout

- [aml/scripts/run_parallel_benchmark.py](aml/scripts/run_parallel_benchmark.py) — per-shard worker (works under both `type: parallel` and `type: command` jobs via `--shards_dir`).
- [aml/scripts/run_sequential_benchmark.py](aml/scripts/run_sequential_benchmark.py) — sequential baseline; iterates the entire matrix in a single command job.
- [aml/scripts/materialize_matrix.py](aml/scripts/materialize_matrix.py) — splits a single matrix JSON into per-profile shard folders consumed by the fan-out steps.
- [aml/scripts/aggregate_results.py](aml/scripts/aggregate_results.py) / [aml/scripts/aggregate_results_flat.py](aml/scripts/aggregate_results_flat.py) — reducers that combine bundles into `combined_runs.json` and `leaderboard.csv`.
- [aml/pipelines/parallel_benchmark_pipeline_dryrun_cmd.yml](aml/pipelines/parallel_benchmark_pipeline_dryrun_cmd.yml) — `command`-job fan-out (no Storage Tables RBAC needed).
- [aml/pipelines/parallel_benchmark_pipeline_dryrun.yml](aml/pipelines/parallel_benchmark_pipeline_dryrun.yml) — `parallel`-job fan-out (requires Storage Table data perms; see below).
- [aml/pipelines/sequential_benchmark_pipeline_dryrun.yml](aml/pipelines/sequential_benchmark_pipeline_dryrun.yml) — sequential baseline pipeline used for runtime comparisons.
- [aml/scripts/submit_pipeline.py](aml/scripts/submit_pipeline.py), [aml/scripts/poll_job.py](aml/scripts/poll_job.py), [aml/scripts/download_job_logs.py](aml/scripts/download_job_logs.py), [aml/scripts/compare_pipeline_runtimes.py](aml/scripts/compare_pipeline_runtimes.py) — submission, polling, log download, and wall-clock comparison helpers.

### Quick start

```powershell
# 1. Probe the workspace and verify the compute cluster is reachable
python aml/scripts/workspace_probe.py

# 2. Register the lightweight dry-run environment (one-time)
python aml/scripts/submit_pipeline.py --register-environments --environments dryrun

# 3. Submit the command-fan-out smoke pipeline (3 shards, ~5 min)
python aml/scripts/submit_pipeline.py `
  --pipeline aml/pipelines/parallel_benchmark_pipeline_dryrun_cmd.yml `
  --no-wait

# 4. Poll and download artifacts
python aml/scripts/poll_job.py --job_name <returned_name> --show_children
python aml/scripts/download_job_logs.py --job_name <returned_name> --download_path .\logs
```

### Choosing a fan-out shape

| Matrix size | Cluster `max_instances` | Recommended pipeline |
|---|---|---|
| 1–12 shards | 1 | `sequential_benchmark_pipeline_dryrun.yml` (per-step overhead dominates) |
| 12–50 shards | ≥3 | `parallel_benchmark_pipeline_dryrun_cmd.yml` (true profile-level fan-out, no extra RBAC) |
| 50+ shards | ≥3 | `parallel_benchmark_pipeline_dryrun.yml` with `mini_batch_size > 1` (multiplies fan-out beyond profile count) |

### Common pitfalls

- **`type: parallel` requires Storage Table RBAC.** The AML parallel runner's `master_poller` queries an Azure Table on the workspace storage account for cross-instance coordination. If the compute cluster's managed identity lacks `Storage Table Data Contributor` on that storage account, every parallel job fails with `AuthorizationPermissionMismatch` (HTTP 403, `SystemExit: 42`) before any user code runs. Either grant the role (see below) or use the command-based fan-out pipeline.
- **Reserved output name `artifacts`.** AML implicitly reserves `outputs/artifacts/` for auto-collected system artifacts. A user-defined `outputs.artifacts` on a `command` job silently fails to bind, and the token `${{outputs.artifacts}}` renders as the literal string `DatasetOutputConfig:artifacts`. Use any other name (the components here use `bundles`).
- **`max_retries: 0` is invalid for parallel jobs.** It must be ≥ 1.
- **Dry-run env needs `azureml-core` + `azureml-dataset-runtime` + `azureml-defaults`.** Without them, parallel jobs fail to import `azureml_sys`.
- **`max_instances=1` silently serializes fan-out.** Concurrent fan-out children declared in the pipeline DAG are queued onto the single available node, eliminating any parallelism benefit and adding per-step orchestration overhead.

### Granting Storage Table RBAC for production-scale `type: parallel` runs

The `command`-fan-out pipeline is sufficient for smoke validation and small fan-out scales (≤ a few dozen shards). To unlock the `type: parallel` job's mini-batch concurrency at production scale, grant the cluster's managed identity the `Storage Table Data Contributor` role on the AML workspace storage account. With Azure CLI:

```powershell
$WORKSPACE_STORAGE = az ml workspace show -n shahra-workspace -g shahra-rg --query storage_account -o tsv
$CLUSTER_PRINCIPAL = az ml compute show -n image-builder -w shahra-workspace -g shahra-rg --query "identity.principal_id" -o tsv
az role assignment create `
  --assignee $CLUSTER_PRINCIPAL `
  --role "Storage Table Data Contributor" `
  --scope $WORKSPACE_STORAGE
```

If your compute cluster has no system-assigned identity, use the AML workspace's managed identity instead (`az ml workspace show ... --query "identity.principal_id"`).

### Wall-clock comparison

Use `compare_pipeline_runtimes.py` to extract execution windows from any pair of pipeline runs:

```powershell
python aml/scripts/compare_pipeline_runtimes.py `
  --sequential <seq_job_name> `
  --parallel <par_job_name>
```

Speedup is dominated by `max_instances`: with `max_instances=1` the parallel pipeline is 2× **slower** than the sequential baseline because step overhead is paid 5× instead of 2×. Bumping `max_instances` to at least the number of fan-out children (3 for our base/memory/hipporag profile split) is required to see any parallelism benefit.
