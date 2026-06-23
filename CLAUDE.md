# model-generalization

Research codebase testing whether MCQ positional-bias-mitigation methods that
were validated on closed/cloud models (see the companion `two-stage-prompting`
repo) generalize across open-weight model families and parameter scales.
Models run locally via HuggingFace `transformers`, on Kelvin2 (Queen's
University Belfast HPC) A100s via SLURM. This is a separate git repository
(`model-generalization-paper` on GitHub) with a single `main` branch — not a
branch of `two-stage-prompting`, despite some wording in this repo's own
README implying otherwise (leftover from when both studies lived in one
monorepo; harmless but worth knowing if you go looking for a
`model-generalization` branch that doesn't exist here).

**Models (5):** `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-32B-Instruct`,
`Qwen/Qwen2.5-72B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct`,
`meta-llama/Llama-3.1-70B-Instruct` — two families, roughly a 10x scale ladder
within each.

**Benchmarks:** MMLU, ARC-Challenge (same 1,000-question robustness splits as
`two-stage-prompting`, for direct comparison).

**Methods (9):** see the table below. The full target matrix is 5 models × 9
methods × 2 benchmarks; each `run_id` is scoped to one model, so
`scripts/build_master_table.py` / `scripts/build_merged_run.py` exist
specifically to stitch the matrix back together across many run folders.

---

## How Claude should work in this repository

Claude should behave conservatively here for the same reason as
`two-stage-prompting`: this is an active research pipeline feeding a paper,
not a generic framework to refactor.

**Prefer:**

- Small, reviewable diffs
- Preserving existing file structure (backends/runners/parsing/scoring split)
- Explaining the intended change before editing
- Adding tests when modifying runner/parsing/scoring logic
- Avoiding unnecessary new dependencies and premature abstraction

**Do not:**

- Submit or run SLURM jobs, or run expensive local-model inference, unless explicitly instructed
- Put HF tokens in `.env` or any committed file — authenticate interactively with `hf auth login` (see slurm script headers)
- Delete `runs/`, `checkpoints/`, or `reports/` content unless explicitly instructed
- Change run-CSV schemas or silently rename method/model keys (`evaluate_run.py` and `aggregate_results.py` key off exact strings — see Gotchas)
- Edit `src/modelgen/config/` for ordinary experiment configuration (that lives in `config/*.yaml`); `experiment.py` there does define an `ALL_METHODS` list that is currently unused anywhere in the codebase and was not updated for `independent_hypothesis` — leave it as-is unless something starts depending on it
- Rewrite working modules for style, or generalize this into a framework, before the paper is finished

**Rule:** if AI vanished tomorrow, Karl should still be able to explain,
modify, and continue the project slowly.

---

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,local]"   # dev: pytest, pytest-asyncio, httpx | local: torch, transformers, accelerate
cp .env.example .env            # HPC paths only — no cloud API keys are used in this repo
```

Full local-dev and first-time-Kelvin2 walkthroughs (venv-on-scratch, `hf auth
login`, syncing `data/`, run ladder from tiny → core → expensive) are in
`README.md` — that file is the maintained step-by-step reference; treat this
file as the behavioral contract and quick orientation, not a duplicate.

---

## Running experiments

```bash
# Preflight estimate only, no model weights loaded, no backend calls
python scripts/run_experiment.py --config config/qwen7b_mmlu_core.yaml --dry-run

# Full run, asks for confirmation
python scripts/run_experiment.py --config config/qwen7b_mmlu_core.yaml

# Resume a partial run by its existing run_id (skips completed_ids via checkpoint)
python scripts/run_experiment.py --config config/qwen7b_mmlu_core.yaml --run-id 20260529_144439 --yes

# Unattended / SLURM runs always need --yes
```

`PYTHONPATH=src` is only needed when *not* using an editable install. All run
configuration lives in `config/*.yaml` (model list, job matrix, `max_tokens`,
seed, checkpoint cadence) — see README's "Configuration" section for the full
schema and the core-vs-expensive `max_tokens` rule (`two_prompt` requires 500;
everything else is safe at 64, but all jobs in one config share a single value).

---

## Project structure

See `README.md` for the full annotated tree. Short version:

```text
src/modelgen/
  backends/   base.py, hf_causal_lm.py (generate + score_options), qwen.py, llama.py, dummy.py, types.py
  runners/    local_base.py (shared base) + one file per method (see table below)
  parsing/    answer-letter extraction (parser.py) — shared by most methods
  scoring/    correct/incorrect scorer
  pipeline/   prompt_builder.py — versioned template loader + per-method prompt formatters
  infra/      checkpoint.py (per-job resumable JSON checkpoints)
  benchmarks/ MMLU and ARC-Challenge loaders/split builders
  io/         CSV readers/writers
  config/     experiment.py (split/subject/method-name constants), paths.py — not for run config

config/    one YAML per (model, benchmark, method-group); tiny_*/​*_core/*_expensive/*_text_extraction/*_abcd/*_independent_hypothesis
prompts/v1/  templates: direct_mcq, free_text, option_matching, text_extraction, abcd, independent_hypothesis
slurm/     numbered scripts (00-31+); see README for the early ones, git log/ls for current
scripts/   run_experiment.py, evaluate_run.py, aggregate_results.py, build_master_table.py,
           build_merged_run.py, recompute_pride_calibration.py, prepare_patch_checkpoints.py, prepare_data.py
tests/     pytest suite mirroring src/ structure

runs/  checkpoints/  reports/  data/processed/  data/splits/   (all gitignored)
```

No `clients/` or `infra/cache.py` exist in this repo (those are
`two-stage-prompting`-only; the README's project-structure listing still
mentions them as "retained from main branch" — that's stale, this repo never
had them).

---

## Experiment methods

| Key | Class | Backend calls / question | Description |
| --- | ----- | ------------------------- | ----------- |
| `baseline` | `DirectMCQRunner` | 1 | Standard MCQ prompt, parses the first answer letter. |
| `two_prompt` | `TwoStageRunner` | 2 | Stage 1 free-text answer (no options shown); Stage 2 matches it to a lettered option. |
| `cyclic` | `PermutationRunner` | N (= real option count, usually 4) | N cyclic option-order rotations scored via `score_options()`, un-permuted and averaged per Eq.(1) (Zheng et al. 2024), argmax of the averaged distribution. |
| `pride` | `PriDeRunner` | 1 + N×calibration_n | Estimates positional prior P_eprior from held-out calibration questions (Eq. 7), applies Eq. 8 debiasing to inference-time `score_options()` logits. Calibration cached as a `runs/<run_id>/pride_calibration__<model>__<benchmark>.json` sidecar. |
| `calibration` | `AnswerCalibrationRunner` | 1 (+ setup calls once) | Scores options against a content-free neutral prompt to estimate a per-label bias prior, subtracts it from real-question scores before picking the answer. |
| `additional_option` | `AdditionalOptionRunner` | 1 | Baseline plus a 5th "E: I don't know" option; one `score_options()` pass, argmax restricted to the 4 real options (Choi et al. 2025, Eq. 6) — IDK can never be selected. |
| `text_extraction` | `TextExtractionRunner` | 1 | Shows all options with A/B/C/D labels but instructs free-text output; resolved via leading-letter/cue-letter shortcuts or sentence-embedding cosine match (no second LLM call). |
| `abcd` | `ABCDRunner` | 1 | Reproduces the M&D protocol of Nowak, Cadet & Chin, "ABCD: All Biases Come Disguised" (arXiv:2602.17445): options shown under neutral dash labels (no letters), four-tier regex span extraction (Appendix F.2) + embedding-similarity resolution (Section 4). |
| `independent_hypothesis` | `IndependentHypothesisRunner` | N (= real option count, usually 4) | Each option evaluated in total isolation: one `generate()` call per option framed as "Hypothesis: the correct answer is X," model outputs brief reasoning plus a `<score>0-100</score>` confidence tag. Final prediction is the argmax of the N regex-parsed scores; ties broken by an RNG seeded from `(seed, question_id)` for reproducibility. All N raw responses/scores are kept in the result row (`option_a_score` etc.) so aggregation can be redone post-hoc without rerunning inference. |

`pride`, `calibration`, and `cyclic` all require a backend that implements
`score_options()` (first-token log-probability access) — the local HF backend
does; the API clients in the companion repo mostly don't (Together AI is the
exception there).

---

## Evaluation outputs

```bash
python scripts/evaluate_run.py <run_id> --benchmark mmlu
python scripts/evaluate_run.py <run_id> --benchmark arc   # arc aliases arc_challenge
python scripts/aggregate_results.py <run_id> --benchmark mmlu
python scripts/aggregate_results.py <run_id> --cross-benchmark
```

Per benchmark, written to `reports/<run_id>/<benchmark>/`: `accuracy.csv`,
`positional_bias.csv` (MAD from ground-truth distribution), `overlap.csv`,
`choice_shifts.csv`, `subject_accuracy.csv`, `two_stage_metrics.csv`.
Aggregated paper tables go to `reports/<run_id>/<benchmark>/paper/` (plain
text + LaTeX). See README's "Experiment output" section for full column/file
definitions and the end-to-end-vs-conditional-accuracy distinction.

---

## Gotchas

- **Model key consistency.** The `models:` dict key in a config must exactly match the `model_name` written to run CSVs, which must match `MODEL_ORDER` in `evaluate_run.py` and `aggregate_results.py` — a mismatch silently drops that model from evaluation output. Update `MODEL_ORDER` in both files when adding a model. `MODEL_ORDER` currently still lists cloud model names from the companion repo's study alongside local ones.
- **ARC's missing-D questions.** A handful of ARC-Challenge robustness-split questions have only 3 real options (`choice_d` arrives as NaN). `LocalExperimentRunner._build_options` drops it; every runner must read real option count from that, never hardcode 4 (see `cyclic`, `pride`, and `independent_hypothesis`, which all loop over the real letter set).
- **`--benchmark arc`** aliases `arc_challenge` in `evaluate_run.py`.
- **Checkpoint/PriDe sidecar coupling.** If you delete a run CSV and need to rerun PriDe from scratch, also delete the matching `pride_calibration__*.json` sidecar — they can't be rerun independently.
- **`config/dummy.yaml`** lists every registered method and is the fastest way to smoke-test a new runner end-to-end (no GPU, instant `DummyBackend`) before writing a real config or SLURM script.

---

## Coding style

Same as the companion repo: simple, explicit Python. **Prefer:**
dataclasses/typed dicts, clear function boundaries, explicit error handling,
deterministic seeds, stable CSV schemas, small helpers, tests for
parsing/scoring/runner changes. **Avoid:** hidden global state, broad
exception swallowing, changing CSV columns or method/model keys without a
migration plan, heavy new dependencies.

---

## Testing

```bash
pytest
python scripts/run_experiment.py --config config/dummy.yaml --yes   # full-pipeline smoke test, no GPU
python scripts/run_experiment.py --config config/<tiny config>.yaml --dry-run
```

When adding a new runner: write tests mirroring `tests/runners/test_permutation.py`'s
pattern for multi-call methods — a `DummyBackend` subclass that keys its
response on prompt content (which option text appears), not call order, so
tests don't depend on execution sequence. When modifying parsing/scoring:
test A/B/C/D extraction, the 3-option ARC case, unscorable outputs, and
deterministic scoring.

---

## Kelvin2 / HPC caution

- Do not run heavy jobs on the login node — use `sbatch`.
- Start with a `tiny_*` config before a `*_core`/`*_expensive`/full config.
- Do not assume local laptop paths work on HPC — `scripts/env_kelvin2.sh` sets `HF_HOME`, `MODEL_ROOT`, `PYTHONPATH`, and activates the scratch venv.
- Verify partition/GRES names with `sinfo -o "%P %D %G %m %l %N"` before submitting — they can change.
- Mental model: login node = prepare/submit; compute node = run workload; SLURM = scheduler.
