# RepoOpt: Distributed Scientific Optimization Trace Factory

Autonomous trace factory for generating verified Python & R scientific optimization trajectories via **Harbor (`pi`) + OpenRouter (`z-ai/glm-5.3`)**, archiving canonical ATIF-v1.8 & raw provider traces in **S3** (or local disk), logging interactive runs & traces to **MLflow on Databricks**, converting trajectories into **Thinking Machines Tinker (`Qwen/Qwen3.5-9B`)** SFT datasets, and training on-policy RL with the same `repoopt-bench` benchmark engine.

## Quickstart (Local or Remote Execution Host)

### 1. Environment Setup
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Configure Credentials Safely (`.env` from `.env.example`)
Never commit `.env` or paste secrets into chat. Copy the template and populate values on the execution host:
```bash
cp .env.example .env
# Edit .env with your OpenRouter, Tinker, AWS S3, Databricks MLflow, and Anyscale credentials
```

For quick terminal entry of API tokens without shell history exposure:
```bash
printf "Enter OPENROUTER_API_KEY (typing hidden): " && read -s val && echo && echo "OPENROUTER_API_KEY=$val" >> .env
printf "Enter TINKER_API_KEY (typing hidden): " && read -s val && echo && echo "TINKER_API_KEY=$val" >> .env
printf "Enter DATABRICKS_TOKEN (typing hidden): " && read -s val && echo && echo "DATABRICKS_TOKEN=$val" >> .env
```

### 3. Run Offline Unit, Tinker, and MLflow Tracking Tests
```bash
pytest tests/test_stats_and_rewards.py tests/test_store_and_gateway.py tests/test_tinker_conversion.py tests/test_mlflow_tracking.py -v
```

### 4. Verify Pilot Docker Tasks (Python + R)
```bash
repoopt preflight-verify tasks/py-lj-forces
repoopt preflight-verify tasks/r-spatial-kernel
```

### 5. Run a Live Rollout (S3 Archive + Databricks MLflow Trace)
When `REPOOPT_STORAGE_BACKEND=s3` and `REPOOPT_MLFLOW_ENABLED=true` are set in `.env`:
```bash
repoopt rollout --task-dir tasks/py-lj-forces --exp-id pilot-001
```
Each rollout archives `ATIF-v1.8` + raw provider turns + `candidate.patch` + `evaluation.json` to S3 and logs an interactive **MLflow Run + GenAI Trace** (`AGENT` -> `CHAT_MODEL` -> `TOOL` -> `independent_verifier`) to your Databricks MLflow Experiment (`MLFLOW_EXPERIMENT_NAME`).

You can also backfill or re-sync any S3 or local experiment into Databricks MLflow at any time:
```bash
repoopt sync-mlflow --exp-id pilot-001
```

### 6. Convert to Qwen3.5 Datums & Smoke-Test Tinker
```bash
repoopt convert-tinker --exp-id pilot-001 --model Qwen/Qwen3.5-9B --require-verified-speedup
repoopt smoke-tinker --exp-id pilot-001 --model Qwen/Qwen3.5-9B
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full v0.4 system design, dual S3 + Databricks MLflow observability, RL reward contract, and storage layout.
