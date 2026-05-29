# MCQ Bias-Mitigation Scale-Generalisation

This repository contains the code and experiment pipeline for a study on whether MCQ positional-bias mitigation methods generalise reliably across open-source model families and scales.

**Research question:** Do MCQ bias-mitigation methods remain reliable when tested across different open-source model sizes and families?

This work extends the findings of the two-stage prompting study (see `main` branch), which showed that naive prompting interventions fail to reduce MCQ positional bias and reduce end-to-end accuracy. The current direction asks whether any existing method is stable across a model hierarchy, or whether apparent gains are artefacts of testing on a narrow set of models.

---

## Motivation

Most MCQ bias-mitigation papers evaluate on one or two models and claim general results. This study tests the same set of methods across a controlled scale ladder within two model families (Qwen and Llama), using the same benchmarks and metrics as the prior work. A method that improves results on a 7B model but fails on a 32B or 70B model is not strong evidence of general mitigation.

---

## Methods Under Evaluation

| Key | Description | Logprobs required |
|---|---|---|
| `baseline` | Direct MCQ, single prompt | No |
| `cyclic` | Four cyclic option rotations, majority vote | No |
| `two_prompt` | Free-text answer then option matching | No |
| `pride` | Logprob-based positional prior debiasing (Zheng et al., ICLR 2024) | Yes |

Additional methods (answer-level calibration, additional-option prompting, text-answer extraction) are candidates for a second pass after the core comparison works.

---

## Models

Local inference via HuggingFace `transformers` on Kelvin2 HPC (NVIDIA A100 / V100 nodes).

| Family | Sizes |
|---|---|
| Qwen 2.5 Instruct | 7B, 32B, 72B |
| Llama 3.1 Instruct | 8B, 70B |

Exact model list is subject to Kelvin2 memory and queue constraints. 405B is not feasible on available hardware.

---

## Benchmarks

MMLU and ARC-Challenge, reusing the same 1,000-question robustness splits from the two-stage study to allow direct comparison.

---

## Metrics

- **End-to-end accuracy** (`correct / total`) — headline metric, includes unscorable outputs in denominator
- **Conditional accuracy** (`correct / scored`) — supplementary
- **MAD** — mean absolute deviation from ground-truth answer-position distribution, primary bias metric
- **Parse / fallback rate** — proportion of unscorable outputs per method
- **Compute cost** — number of model calls per question per method

---

## Repository Structure

```
config/
  default.yaml              job matrix, model configs, rate limits

data/                       benchmark data, normalised CSVs, stratified splits

scripts/
  run_experiment.py         overnight runner
  evaluate_run.py           accuracy, bias, overlap, per-subject stats
  aggregate_results.py      paper-ready tables
  prepare_data.py           one-time data preprocessing

archive/scripts/            scripts retained for reference only
  generate_figures.py       figure generation from prior study
  smoke_clients.py          one-off API connectivity check

src/twoprompt/
  backends/                 local HuggingFace inference backends (Qwen, Llama, Dummy)
  clients/                  async cloud API clients (OpenAI, Gemini, Groq, Together)
  runners/                  method runners (baseline, cyclic, two-stage, PriDe)
  infra/                    disk cache, checkpointing
  benchmarks/               benchmark loaders (MMLU, ARC-Challenge)
  parsing/                  answer parser
  scoring/                  scorer
  pipeline/                 prompt builder

prompts/v1/                 prompt templates
tests/                      test suite
```

`runs/`, `reports/`, `checkpoints/`, and `.cache/` are gitignored and generated locally.

---

## Setup

```bash
# Install all dependencies (dev + local model inference)
pip install -e ".[dev,local]"

cp .env.example .env       # fill in API keys if using cloud models
```

For local model inference, `torch`, `transformers`, and `accelerate` are installed via the `local` extra. Cloud API keys (`OPENAI_API_KEY`, `GEMINI_API_KEY`, `GROQ_API_KEY`, `TOGETHER_API_KEY`) are only required for cloud-backed runs.

---

## Running Experiments

```bash
# Dry run — cost and time estimate, no model calls
python scripts/run_experiment.py --dry-run

# Full run (prompts for confirmation)
python scripts/run_experiment.py

# Resume a previous run
python scripts/run_experiment.py --run-id <RUN_ID> --yes
```

## Evaluation

```bash
python scripts/evaluate_run.py <RUN_ID> --benchmark mmlu
python scripts/evaluate_run.py <RUN_ID> --benchmark arc
python scripts/aggregate_results.py
```

Reports written to `reports/<RUN_ID>/<benchmark>/`.

---

## HPC Setup and Kelvin2 Run Ladder

Experiments run on Kelvin2 (Queen's University Belfast) via SLURM batch jobs. SLURM scripts live in `slurm/` and invoke `scripts/run_experiment.py`. Python code has no SLURM dependency.

**First-time setup on Kelvin2:**

```bash
# 1. Clone repo into scratch (home quota is too small for outputs)
cd /mnt/scratch2/users/$USER
git clone <repo-url> repos/model-generalization
cd repos/model-generalization

# 2. Create venv on scratch (not home — home quota is 50GB)
python3 -m venv /mnt/scratch2/users/$USER/venvs/mcq-generalization
source /mnt/scratch2/users/$USER/venvs/mcq-generalization/bin/activate

# 3. Install dependencies
pip install -e ".[dev,local]"

# 4. Copy benchmark data from local machine
#    (data/ is gitignored — run scp from your laptop)
#    scp -r data/ <user>@kelvin2.alces.network:/mnt/scratch2/users/$USER/repos/model-generalization/

# 5. Authenticate with HuggingFace (needed for Llama models)
#    Do NOT put the HF token in .env — use hf auth login interactively instead.
#    HF_HOME and HF_HUB_CACHE are set inside each SLURM script automatically.
hf auth login
```

**Run ladder (after first-time setup):**

```bash
# SSH into Kelvin2, then:
cd /mnt/scratch2/users/$USER/repos/model-generalization

# If the local repo has changed since last session:
git pull
# If Kelvin2 has local emergency edits that conflict and GitHub now has the fixes:
# git reset --hard && git pull

# Load environment (sets HF_HOME, PYTHONPATH, venv activation, etc.)
source scripts/env_kelvin2.sh

# Syntax-check all SLURM scripts before submitting anything
bash -n slurm/*.sh

# Confirm benchmark data is present and parseable
PYTHONPATH=src python scripts/prepare_data.py

# Confirm model weights are available (download if needed)
hf download Qwen/Qwen2.5-0.5B-Instruct
hf download Qwen/Qwen2.5-7B-Instruct

# Confirm dummy backend runs end-to-end without GPU
PYTHONPATH=src python scripts/run_experiment.py --config config/dummy.yaml --yes

# Tiny real run — 5 questions, Qwen 7B, no two_prompt, confirms weights load on GPU
sbatch --export=CONFIG=config/tiny_qwen7b_mmlu.yaml slurm/05_run_config.sh

# Monitor the job
squeue -u $USER
sacct -j JOBID --format=JobID,JobName,Partition,State,ExitCode,Elapsed,MaxRSS
cat logs/*JOBID*.out
cat logs/*JOBID*.err

# Inspect output CSVs once the job finishes
ls runs/

# After tiny run passes, submit core configs (cheap — no two_prompt)
sbatch --export=CONFIG=config/qwen7b_mmlu_core.yaml    slurm/05_run_config.sh
sbatch --export=CONFIG=config/qwen7b_arc_core.yaml     slurm/05_run_config.sh

# After core configs pass, submit expensive configs (two_prompt + cyclic + pride)
sbatch --export=CONFIG=config/qwen7b_mmlu_expensive.yaml slurm/05_run_config.sh
sbatch --export=CONFIG=config/qwen7b_arc_expensive.yaml  slurm/05_run_config.sh
```

**Notes:**

- Do not run heavy jobs on the login node — use `sbatch` to submit to the compute queue.
- Do not put HF tokens in `.env` — use `hf auth login` interactively on the login node.
- If Kelvin2 has local emergency edits and GitHub now has the fixes: `git reset --hard` then `git pull`.

**Future larger models:**

| Model size | Partition | GRES | Notes |
|---|---|---|---|
| 7B | `k2-gpu-a100mig` | `gpu:a100mig_3g.40gb:1` | Current target |
| 32B | `k2-gpu-a100` | `gpu:a100:1` | Full 80GB A100 |
| 72B / H100 | `k2-gpu-h100` | `gpu:h100:1` | H100 for larger models |
| 32B+ quantized | `k2-gpu-a100` | `gpu:a100:1` | May need multi-GPU or 4-bit quant |

Verify partition and GRES names with `sinfo` on the login node before submitting — names above reflect current Kelvin2 config but may change.

---

## Prior Work

The `main` branch contains the two-stage prompting study:
*Two-Stage Prompting Does Not Mitigate MCQ Positional Bias in LLMs*
Karl Hanna, 2026

The current branch (`model-generalization`) extends that infrastructure to local open-source models and a scale-generalisation design.

---

## AI Usage Disclosure

The research question, experiment design, and interpretation of results are my own. Infrastructure code (backends, checkpointing, caching, retry logic, orchestration) and test scaffolding were written with substantial AI assistance under my direction and reviewed manually. Logic that directly affects paper claims was reviewed and verified manually.

---

**Author:** Karl Hanna
