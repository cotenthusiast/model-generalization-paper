# MCQ Bias-Mitigation Scale-Generalisation

This repository contains the experiment pipeline for a study on whether MCQ positional-bias mitigation methods generalise reliably across open-source model sizes and families. It extends a prior two-stage prompting study (see the `main` branch) which showed that naive prompting interventions fail to reduce MCQ positional bias and reduce end-to-end accuracy. This direction asks whether any existing method is stable across a controlled model scale ladder, or whether apparent gains are artefacts of testing on a narrow set of models. Models are run locally via HuggingFace `transformers` on Kelvin2 HPC. Benchmarks are MMLU and ARC-Challenge using the same 1,000-question robustness splits from the prior work, allowing direct comparison.

---

## Project structure

```
config/
  default.yaml                 full job matrix for Qwen 7B + Llama 8B, all methods
  dummy.yaml                   all methods, dummy backend — smoke-tests the full pipeline
  tiny_real.yaml               5 questions, Qwen 0.5B, core methods — first real end-to-end test
  tiny_qwen7b_mmlu.yaml        5 questions, Qwen 7B, core methods — confirms 7B weights load
  qwen7b_mmlu_core.yaml        1000 questions, Qwen 7B, core methods (baseline/calibration/additional_option)
  qwen7b_arc_core.yaml         same as above for ARC-Challenge
  qwen7b_mmlu_expensive.yaml   1000 questions, Qwen 7B, expensive methods (two_prompt/cyclic/pride)
  qwen7b_arc_expensive.yaml    same as above for ARC-Challenge

scripts/
  run_experiment.py            main runner: loads config, runs jobs, writes CSVs
  evaluate_run.py              computes accuracy, bias, overlap, and per-subject stats
  aggregate_results.py         builds paper tables (plain-text + LaTeX) from evaluate outputs
  prepare_data.py              one-time download and normalisation of MMLU and ARC-Challenge
  env_kelvin2.sh               source on Kelvin2 to set paths, load module, activate venv

slurm/
  00_smoke_test.sh             smoke test job (no GPU)
  01_download_models.sh        model weight download job
  02_small_batch.sh            20-question sanity check, Qwen 7B + Llama 8B
  03_tiny_real.sh              5-question real weights test, Qwen 0.5B
  04_full_run.sh               full-scale run job
  05_run_config.sh             reusable A100 MIG job; pass CONFIG= at submit time

src/modelgen/
  backends/
    base.py                    abstract backend interface
    hf_causal_lm.py            shared HuggingFace CausalLM backend (generate + score_options)
    qwen.py                    Qwen-family backend (subclass of hf_causal_lm)
    llama.py                   Llama-family backend (subclass of hf_causal_lm)
    dummy.py                   deterministic dummy backend for pipeline testing
    types.py                   LocalGenerationConfig, ModelGenerationResult, ScoreResult
  runners/
    local_base.py              shared runner base: prompt loading, parse+score, result row builder
    direct_mcq.py              baseline runner
    two_stage.py               two-prompt runner
    permutation.py             cyclic permutation runner
    pride.py                   PriDe runner (calibration + Eq.(8) inference)
    calibration.py             answer-level calibration runner
    additional_option.py       additional-option ("I don't know") runner
  clients/                     async cloud API clients (OpenAI, Gemini, Groq, Together)
                               — retained from main branch; not used in local inference
  infra/
    cache.py                   disk-backed JSON cache for cloud client responses
    checkpoint.py              per-job checkpoint manager for resumable runs
  benchmarks/                  MMLU and ARC-Challenge loaders and split builders
  config/
    experiment.py              split definitions, subject lists, method name constants
    paths.py                   default filesystem path constants
  parsing/                     answer letter extractor
  scoring/                     correct/incorrect scorer
  pipeline/                    prompt template renderer
  io/                          CSV readers and writers

prompts/v1/
  direct_mcq.txt               template for single-turn MCQ prompts
  free_text.txt                template for two-stage Stage 1 (free-text answer)
  option_matching.txt          template for two-stage Stage 2 (option matching)

tests/                         pytest suite mirroring src/ structure
data/processed/                normalised benchmark CSVs (gitignored, generate with prepare_data.py)
data/splits/                   split ID files (gitignored, generate with prepare_data.py)
runs/                          run output CSVs (gitignored)
checkpoints/                   in-progress job state (gitignored)
reports/                       evaluate_run.py and aggregate_results.py outputs (gitignored)
.cache/responses/              cloud API response cache (gitignored)
```

---

## Method stack

| Config key | Runner class | Backend calls / question | Description |
|---|---|---|---|
| `baseline` | `DirectMCQRunner` | 1 | Presents the standard MCQ prompt and parses the first answer letter from the response. |
| `two_prompt` | `TwoStageRunner` | 2 | Stage 1 elicits a free-text answer without showing options; Stage 2 asks the model to match that answer to one of the four options. |
| `cyclic` | `PermutationRunner` | 4 | Runs four cyclic rotations of the option order, un-permutes each parsed answer back to canonical ordering, and selects the final answer by majority vote. |
| `pride` | `PriDeRunner` | 1 + 4×calibration_n | Estimates a positional prior P_eprior from held-out calibration questions (Eq. 7), then applies Eq. 8 debiasing to `score_options()` logits at inference. Calibration is saved as a JSON sidecar and reused on reruns. |
| `calibration` | `AnswerCalibrationRunner` | 1 | Scores options against a content-free neutral prompt to estimate per-label bias, then subtracts that prior from real-question scores before picking the answer. |
| `additional_option` | `AdditionalOptionRunner` | 1 | Same as baseline but adds a fifth option `E: I don't know` to the prompt. |

`pride` and `calibration` both require a backend that implements `score_options()` (first-token log-probability scoring). The local HF backend supports this; cloud backends do not (with the exception of Together AI in the `main` branch study).

---

## Configuration

All run configuration lives in `config/*.yaml`. The Python modules under `src/modelgen/config/` define constants and should not be edited for ordinary experiment configuration.

**Schema** (`config/tiny_real.yaml` is the canonical reference):

```yaml
models:
  <model-key>:           # exact string that appears in run CSVs; must match MODEL_ORDER
    model_path: ...      # HuggingFace hub ID or absolute local path
    family: qwen|llama|dummy
    size_label: 7B       # informational only
    device: auto         # auto | cuda | cpu

run:
  temperature: 0.0
  max_tokens: 64         # see note below
  seed: 42
  prompt_version: "v1"
  max_questions: 5       # cap per job; omit or set null for no cap
  checkpoint_every_n: 5
  pride_calibration_n: 50
  pride_calibration_seed: 42
  fallback_on_parse_failure: false
  jobs:
    - model: <model-key>
      methods: [baseline, calibration, additional_option]
      benchmark: mmlu    # mmlu | arc_challenge
      split: robustness

paths:
  runs_dir: runs
  checkpoints_dir: checkpoints
  cache_dir: .cache/responses
  reports_dir: reports
  data_processed_dir: data/processed
  data_splits_dir: data/splits
  prompts_dir: prompts
```

**`max_tokens` and method groups:**

All jobs in a config share one `max_tokens` value — per-method token limits are not supported.

- **Core methods** (`baseline`, `calibration`, `additional_option`): `max_tokens: 64` is safe and preferred. These methods either parse a single letter or use `score_options()` and never generate long responses.
- **Expensive methods** (`two_prompt`, `cyclic`, `pride`): any config that includes `two_prompt` must use `max_tokens: 500`. Stage 1 of `two_prompt` generates free-text reasoning; truncating at 64 tokens silently cuts off the answer. `cyclic` and `pride` would be safe at 64, but because all jobs share one value, `500` is required whenever `two_prompt` is present.

The configs are split accordingly: `*_core.yaml` files use 64; `*_expensive.yaml` files use 500.

---

## Local development

```bash
# Create and activate venv
python3 -m venv .venv && source .venv/bin/activate

# Install all dependencies including local model inference
pip install -e ".[dev,local]"
# dev adds: pytest, pytest-asyncio, httpx
# local adds: torch, transformers, accelerate

# Copy and fill in API keys (only needed for cloud clients from main branch)
cp .env.example .env

# Confirm the pipeline works end-to-end without any model weights
PYTHONPATH=src python scripts/run_experiment.py --config config/dummy.yaml --yes

# Run the test suite
pytest
```

`PYTHONPATH=src` is required because the package is installed in editable mode from `src/` but scripts live at the repo root.

---

## Kelvin2 HPC

Experiments run on Kelvin2 (Queen's University Belfast) via SLURM. SLURM scripts in `slurm/` invoke `scripts/run_experiment.py`; the Python code has no SLURM dependency.

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

# 5. Authenticate with HuggingFace (needed for gated models such as Llama)
#    Do NOT put the HF token in .env — use hf auth login interactively.
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

# Confirm model weights are available (download if not already cached)
hf download Qwen/Qwen2.5-0.5B-Instruct
hf download Qwen/Qwen2.5-7B-Instruct

# Confirm dummy backend runs end-to-end without a GPU
PYTHONPATH=src python scripts/run_experiment.py --config config/dummy.yaml --yes

# Tiny real run — 5 questions, Qwen 7B, core methods only, confirms weights load on GPU
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

- Do not run heavy jobs on the login node — submit via `sbatch`.
- Do not put HF tokens in `.env` — use `hf auth login` interactively on the login node.
- If Kelvin2 has local emergency edits that conflict with upstream fixes: `git reset --hard` then `git pull`.

**Future larger models:**

| Model size | Partition | GRES | Notes |
|---|---|---|---|
| 7B | `k2-gpu-a100mig` | `gpu:a100mig_3g.40gb:1` | Current target |
| 32B | `k2-gpu-a100` | `gpu:a100:1` | Full 80GB A100 |
| 72B / H100 | `k2-gpu-h100` | `gpu:h100:1` | H100 for larger models |
| 32B+ quantized | `k2-gpu-a100` | `gpu:a100:1` | May need multi-GPU or 4-bit quant |

Verify partition and GRES names with `sinfo` on the login node before submitting — names above reflect current Kelvin2 config but may change.

---

## Experiment output

**Generating data:**

```bash
# One-time: download benchmarks and build splits
PYTHONPATH=src python scripts/prepare_data.py

# Run experiment (--dry-run for preflight estimate only; --yes skips confirmation)
PYTHONPATH=src python scripts/run_experiment.py --config config/qwen7b_mmlu_core.yaml --yes

# Evaluate a completed run
PYTHONPATH=src python scripts/evaluate_run.py <run_id> --benchmark mmlu
PYTHONPATH=src python scripts/evaluate_run.py <run_id> --benchmark arc   # arc aliases arc_challenge

# Build paper tables
PYTHONPATH=src python scripts/aggregate_results.py <run_id> --benchmark mmlu
PYTHONPATH=src python scripts/aggregate_results.py <run_id> --cross-benchmark
```

**Output locations:**

`runs/<run_id>/` — one CSV per `(method, model, benchmark)` job, named `<method>__<model>__<benchmark>.csv`. Also contains a snapshotted `config.yaml` and `prompts/<version>/` directory copied at run start.

`checkpoints/<run_id>/` — one JSON file per in-progress job, named `<method>__<model>__<benchmark>.json`. Holds completed question IDs and accumulated result rows. Deleted automatically when a job finishes successfully. Resume a partial run with `--run-id <existing_id> --yes`.

`reports/<run_id>/<benchmark>/` — evaluation outputs from `evaluate_run.py`:

| File | Contents |
|---|---|
| `accuracy.csv` | End-to-end and conditional accuracy with 95% Clopper-Pearson CIs, plus API failure and parse failure counts |
| `positional_bias.csv` | Per-option prediction distribution, mean absolute deviation (MAD) from ground-truth distribution, and 95% bootstrap CI |
| `overlap.csv` | Question-level overlap between baseline and each method |
| `choice_shifts.csv` | Broken and fixed answer counts vs baseline, by (model, method) |
| `subject_accuracy.csv` | Per-subject accuracy breakdown (MMLU only) |
| `two_stage_metrics.csv` | Free-text availability rate and latency for `two_prompt` rows |

`reports/<run_id>/<benchmark>/paper/` — aggregated tables from `aggregate_results.py`, in plain-text (`tables.txt`) and LaTeX (`tables.tex`). `--cross-benchmark` writes `tables_cross_benchmark.{txt,tex}` to `reports/<run_id>/paper/`.

**Metric definitions:**

- **End-to-end accuracy**: `correct / total` — unscorable outputs count as incorrect in the denominator.
- **Conditional accuracy**: `correct / scored` — denominator excludes rows where no answer was parsed. Can mask parse failures; always read alongside the unscorable count.
- **MAD**: mean absolute deviation of the model's answer-position distribution from the ground-truth distribution, in percentage points. Lower means less positional bias.

---

## Development notes

**`--dry-run` and `--yes`:**
`--dry-run` prints a preflight call count estimate and exits without loading any model weights or making any backend calls. `--yes` skips the interactive confirmation prompt and is required for unattended SLURM runs.

**Checkpoint and cache behaviour:**
Checkpoints are written to `checkpoints/` every `checkpoint_every_n` questions. On a resumed run (`--run-id <id> --yes`), completed question IDs are skipped automatically. Checkpoints are deleted after each job completes successfully; if a job is interrupted mid-write, the `.tmp` file is left behind and the last clean checkpoint is used on next start.

The disk cache in `.cache/responses/` is only used by the cloud API clients (`CachingClientWrapper` in `infra/cache.py`). Local HF backends do not use it.

**PriDe calibration sidecar:**
After calibration, `PriDeRunner` writes `runs/<run_id>/pride_calibration__<model>__<benchmark>.json`. On rerun, if the sidecar exists and its calibration question IDs and seed match, calibration is skipped. If the run CSV is deleted and you need to rerun PriDe from scratch, also delete the matching sidecar.

**Model key consistency:**
The model key in `config.yaml` (the dict key under `models:`) must exactly match the `model_name` column written to run CSVs, which must match entries in `MODEL_ORDER` in `evaluate_run.py` and `aggregate_results.py`. A mismatch silently drops that model from evaluation output. `Qwen/Qwen2.5-7B-Instruct` and `Qwen/Qwen2.5-7B-Instruct-Turbo` are distinct keys.

**`evaluate_run.py` `MODEL_ORDER`:**
Currently includes cloud model names from the main branch study alongside local model names. Rows for models not in `MODEL_ORDER` are silently excluded from evaluation tables. Update `MODEL_ORDER` in both `evaluate_run.py` and `aggregate_results.py` when adding new models.

---

## Prior work

The `main` branch contains the two-stage prompting study:
*Two-Stage Prompting Does Not Mitigate MCQ Positional Bias in LLMs*, Karl Hanna, 2026.

The current branch (`model-generalization`) extends that infrastructure to local open-source models and a scale-generalisation design.

---

## AI usage disclosure

The research question, experiment design, and interpretation of results are my own. Infrastructure code (backends, checkpointing, caching, retry logic, orchestration) and test scaffolding were written with substantial AI assistance under my direction and reviewed manually. Logic that directly affects paper claims was reviewed and verified manually.

---

**Author:** Karl Hanna
