# RepoOpt v0.4 Architecture

RepoOpt is an autonomous scientific software optimization trace factory built on **Harbor** (`pi` agent), **OpenRouter** (`z-ai/glm-5.3`), **Docker / Ray (Anyscale)**, **ATIF-v1.8**, and **Thinking Machines Tinker** (`Qwen/Qwen3.5-9B`).

## Pressure-Tested Design Decisions (v0.4)

1. **Two-Agent Split (`BenchArchitect` vs. `Optimizer`) + `dev` / `holdout` Split**
   - `BenchArchitect` (`src/repoopt/architect/`) prepares a pinned, history-squashed repository, writes `dev_cases.*` (exposed via `/usr/local/bin/repoopt-bench` during agent episodes) and private `tests/holdout_cases.*` (mounted only during verification), and passes an automated **Preflight Gate** (`src/repoopt/bench/preflight_gate.py`):
     1. Baseline no-op passes `dev` + `holdout` and scores `reward = 0.0` (`CORRECT_NO_SPEEDUP`).
     2. Synthetic `mutation_bad.patch` fails numerical checks (`INCORRECT`, `reward = 0.0`).
     3. Reference optimization passes with `VERIFIED_SPEEDUP`.
   - `Optimizer` runs stock Harbor `pi` (`read`, `write`, `edit`, `bash`) in Python and R containers equipped with native compilers (`gcc`, `g++`, `cython3`, `Rcpp`, `OpenBLAS`), calling `repoopt-bench` interactively to profile and check accuracy/speedup live against `/opt/baseline`.

2. **Live Paired A/B/A/B Benchmarking & Clean-Room Rebuild**
   - `repoopt-bench` compares `/workspace` live against read-only `/opt/baseline_venv` (Python) or `/opt/baseline_rlib` (R) in isolated subprocesses (`python3 -I` / `Rscript --vanilla`) with single-thread BLAS/OpenMP and regenerated inputs per iteration.
   - `tests/run_verifier.py` extracts a text-only `candidate.patch` (rejecting binary smuggling), applies it to pristine `/opt/baseline_src` in a temporary directory, rebuilds from source, and grades both `dev` and unseen `holdout` workloads.

3. **Anti-Collapse RL Reward (`/logs/verifier/reward.txt` & `reward.json`)**
   - `INCORRECT`, `TIMEOUT`, `ENVIRONMENT_FAILURE`, or empty no-op patch -> `0.0`
   - `CORRECT_NO_SPEEDUP` / `CORRECT_BUT_SLOWER` -> `0.05`
   - `VERIFIED_SPEEDUP` (`correct == True`, `speedup > 1.05`, `ci95_low > 1.02`) -> `min(4.0, log2(speedup))`

4. **Zero-Secret Host Capture & Concurrency Gateway (`src/repoopt/proxy/gateway.py`)**
   - Containers never hold `OPENROUTER_API_KEY` or S3 credentials. Instead, `pi` talks to `http://host.docker.internal:<port>/v1` using an attempt-scoped token (`attempt:<exp_id>:<task_id>:<attempt_id>`).
   - The host gateway enforces central API concurrency (`asyncio.Semaphore`), reassembles streaming SSE chunks (including `reasoning`/`reasoning_content` and incremental `tool_calls`), archives raw provider JSON (`raw-provider-events/turn-NNNN.json`), and flushes immutable JSONL event segments (`events/seg-NNNNNN-<sha12>.jsonl`) on completed turns and at `<= 15s` intervals.

5. **Compaction- & `<think>`-Aware Qwen3.5 / Tinker Conversion (`src/repoopt/tinker_adapter/`)**
   - Validates ATIF (`TrajectoryValidator` + raw tool-call JSON validation; upgrades Harbor `pi` `ATIF-v1.7` outputs to `ATIF-v1.8`).
   - Splits at context compaction boundaries and uses `Qwen3_5Renderer` (`has_extension_property=False`) to emit prefix-consistent `tinker.Datum` instances where system/user/compacted-summary/tool-observation tokens receive `weight = 0.0` and assistant reasoning + tool calls + responses receive `weight = 1.0`.
   - `rl_bridge.py` exposes Pi's exact 4-tool surface (`read`, `write`, `edit`, `bash`) over `SandboxInterface` for on-policy Tinker RL (`harbor_rl`).

6. **Dual Storage & Databricks MLflow Trace Observability (`src/repoopt/tracking/`)**
   - **S3 (`S3ArtifactStore`)** remains the immutable canonical dataset archive (`experiments/<exp>/tasks/<task>/attempts/<attempt>/{raw-provider-events,events,trajectory.json,candidate.patch,evaluation.json,completion.json}` and Tinker `Datum` shards).
   - **MLflow on Databricks (`mlflow_logger.py`)** provides searchable interactive trace visualization and experiment comparison alongside S3:
     - Each rollout logs an **MLflow Run** (outcome tags, speedup/CI/accuracy/reward/token metrics, and `candidate.patch` / `evaluation.json` / `trajectory.json` artifacts) linked via `s3://...` URI.
     - Inside that Run, it logs a structured **MLflow GenAI Trace** with a root `AGENT` span, child `CHAT_MODEL` spans (delta prompt messages, `<think>` reasoning, tool calls, latency, and `mlflow.chat.tokenUsage`), `TOOL` spans (`read`/`write`/`edit`/`bash` with error propagation), an `independent_verifier` span, and trace assessments (`outcome_label`, `reward`, `verified_speedup`).
     - Logging runs automatically when `REPOOPT_MLFLOW_ENABLED=true` (non-fatal if offline) or on-demand/idempotently via `repoopt sync-mlflow`.
