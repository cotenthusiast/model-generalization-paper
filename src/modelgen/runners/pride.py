# src/modelgen/runners/pride.py

"""PriDe runner (Zheng et al., ICLR 2024) — local backend, score_options() for logits.

Two-phase method:
  Phase 1 (calibration): estimate the model's positional bias prior P_eprior from a
  held-out calibration set. For each calibration question, run 4 cyclic permutations
  of the option text, collect a 4x4 probability matrix, apply Eq.(7) to extract a
  per-question prior, then average across all calibration questions into one global prior.

  Phase 2 (inference): for each eval question, get one score_options() call to obtain
  P_obs, then apply Eq.(8) — divide P_obs by P_eprior and renormalize — to get the
  debiased distribution. Pick the argmax.

Calibration runs once before the first eval question. The resulting prior is saved to a
JSON sidecar so it can be reused on reruns without repeating the calibration backend calls.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from modelgen.backends.types import LocalGenerationConfig
from modelgen.parsing.types import PARSE_OK, ParseResult
from modelgen.pipeline.prompt_builder import build_direct_mcq_prompt
from modelgen.runners.local_base import LocalExperimentRunner
from modelgen.runners.permutation import PermutationRunner
from modelgen.runners.pride_debias import (
    OPTION_LETTERS,
    CalibrationState,
    apply_debiased_choice_from_defaults,
    average_prior_probability_vectors,
    calibration_state_from_sidecar,
    calibration_state_uniform,
    equation7_prior_from_rollouts,
    logprob_map_to_label_distribution,
)

logger = logging.getLogger(__name__)

# Bump this when the sidecar JSON format changes in a backwards-incompatible way.
# A mismatch causes the runner to refit rather than load from disk.
_SIDE_SCHEMA_VERSION = 3


def _pick_calibration_rows(
        full: list[dict],
        k: int,
        seed: int,
) -> tuple[list[str], list[dict]]:
    """Return a deterministic random subset of k rows from full.

    Using a seeded shuffle ensures that the same calibration questions are
    picked across reruns, which is required for sidecar cache reuse.
    """
    import random
    if not full:
        return [], []
    kk = max(0, min(int(k), len(full)))
    if kk == 0:
        return [], []
    if kk == len(full):
        chosen_idx = list(range(len(full)))
    else:
        rng = random.Random(int(seed))
        idx = list(range(len(full)))
        rng.shuffle(idx)
        chosen_idx = sorted(idx[:kk])
    rows = [full[i] for i in chosen_idx]
    qids = [r["question_id"] for r in rows]
    return qids, rows


class PriDeRunner(LocalExperimentRunner):
    """Cyclic permutation prior estimation (Paper §3) then Eq.(8) transfer inference.

    Calibration questions must be disjoint from evaluation questions so that
    the estimated prior is not contaminated by in-distribution label leakage.
    All evaluation rows use eq8_transfer mode.

    Logprobs come from backend.score_options() rather than API logprob responses.
    """

    def __init__(
            self,
            backend: Any,
            method_name: str,
            split_name: str,
            prompt_version: str,
            prompts_dir: Path,
            run_id: str,
            generation_config: LocalGenerationConfig | None = None,
            perturbation_name: str | None = None,
            *,
            calibration_n: int = 50,
            calibration_seed: int = 42,
            calibration_benchmark: str = "",
            calibration_runs_dir: Path | None = None,
            calibration_questions: list[dict] | None = None,
    ) -> None:
        super().__init__(
            backend=backend,
            method_name=method_name,
            split_name=split_name,
            prompt_version=prompt_version,
            prompts_dir=prompts_dir,
            run_id=run_id,
            generation_config=generation_config,
            perturbation_name=perturbation_name,
        )

        self._calibration_n = max(0, int(calibration_n))
        self._calibration_seed = int(calibration_seed)
        self._calibration_benchmark = calibration_benchmark or split_name
        self._calibration_runs_dir = Path(calibration_runs_dir or Path("."))
        # Full pool of candidate calibration questions (must not overlap eval split).
        self._calibration_questions: list[dict] = list(calibration_questions or [])

        # Lazily initialised — set to True once the prior has been estimated or loaded.
        self._calibration_ready: bool = False
        # Starts as uniform; replaced with the fitted prior before any eval call.
        self._calibration_state: CalibrationState = calibration_state_uniform()

    def _sidecar_path(self) -> Path:
        """Return the path where the calibration prior JSON is saved for this run."""
        slug = (
            self.backend.metadata.model_path
            .replace("/", "_")
            .replace(" ", "_")
            .replace(":", "_")
        )
        return (
            self._calibration_runs_dir
            / self.run_id
            / f"pride_calibration__{slug}__{self._calibration_benchmark}.json"
        )

    def run_many(self, question_rows: Sequence[Any]) -> list[dict]:
        # Calibrate once before processing any eval questions.
        self._ensure_calibration()
        return [self.run_one(row, i) for i, row in enumerate(question_rows)]

    # ------------------------------------------------------------------
    # Calibration — Phase 1
    # ------------------------------------------------------------------

    def _ensure_calibration(self) -> None:
        """Run calibration if it hasn't happened yet; no-op on subsequent calls."""
        if self._calibration_ready:
            return

        # Pick which questions to use for calibration (deterministic, seeded).
        cal_qids, cal_rows = _pick_calibration_rows(
            self._calibration_questions,
            self._calibration_n,
            self._calibration_seed,
        )
        sorted_ids = tuple(sorted(cal_qids))
        path = self._sidecar_path()

        # Try to load a previously saved prior that matches this exact calibration set.
        # If no valid sidecar exists, compute the prior from scratch and save it.
        state = self._try_load_calibration_sidecar(path, sorted_ids)
        if state is None:
            state = self._fit_calibration_prior(cal_rows)
            self._save_calibration_sidecar(path, state, sorted_ids)

        self._calibration_state = state
        self._calibration_ready = True

    def _try_load_calibration_sidecar(
            self,
            path: Path,
            sorted_ids: tuple[str, ...],
    ) -> CalibrationState | None:
        """Load and validate a previously saved calibration prior from disk.

        Returns None if the file doesn't exist, is malformed, or doesn't match
        the current calibration question set and seed.
        """
        if not sorted_ids or not path.exists():
            return None
        try:
            blob = json.loads(path.read_text())
            # Validate that the sidecar was produced with the same questions and seed.
            if (
                blob.get("schema_version") == _SIDE_SCHEMA_VERSION
                and tuple(sorted(blob.get("calibration_question_ids") or [])) == sorted_ids
                and int(blob.get("calibration_seed", -1)) == self._calibration_seed
            ):
                state = calibration_state_from_sidecar(blob)
                logger.info("PriDe loaded sidecar (K=%d) → %s", len(sorted_ids), path)
                return state
        except (json.JSONDecodeError, KeyError, OSError, TypeError, ValueError) as exc:
            logger.warning("PriDe sidecar unreadable (%s); refitting.", exc)
        return None

    def _fit_calibration_prior(self, cal_rows: list[dict]) -> CalibrationState:
        """Estimate P_eprior from calibration questions using Eq.(7).

        For each calibration question:
          1. Run 4 cyclic permutations of the options through score_options() to get
             a 4x4 probability matrix (rows=permutations, cols=letters A-D).
          2. Apply Eq.(7) — geometric mean across permutations in log space — to
             extract a per-question positional prior vector.

        Then average all per-question priors into one global prior P_eprior.
        """
        if not cal_rows:
            logger.warning("PriDe: no calibration questions available — using uniform prior.")
            return calibration_state_uniform()

        prior_vectors: list[np.ndarray] = []
        for row in cal_rows:
            # Shape (4, 4): row k = probability distribution over A-D under permutation k.
            roll_mat = self._cyclic_rollout_prob_matrix(row)
            # Eq.(7): geometric mean across permutations → per-question prior vector.
            prior_vectors.append(equation7_prior_from_rollouts(roll_mat))

        # Arithmetic mean of all per-question priors → global P_eprior.
        pep_global = average_prior_probability_vectors(prior_vectors)
        return CalibrationState(
            peprior_probs={
                OPTION_LETTERS[i]: float(pep_global[i])
                for i in range(len(OPTION_LETTERS))
            },
            epsilon=1e-12,
            estimation_question_ids=(),
        )

    def _save_calibration_sidecar(
            self,
            path: Path,
            state: CalibrationState,
            sorted_ids: tuple[str, ...],
    ) -> None:
        """Persist the fitted prior to disk so reruns can skip calibration."""
        payload = {
            "schema_version": _SIDE_SCHEMA_VERSION,
            "version": state.version,
            "calibration_seed": self._calibration_seed,
            "n_options": len(OPTION_LETTERS),
            "calibration_question_ids": list(sorted_ids),
            "peprior_probs": {
                L: float(state.peprior_probs.get(L, 0.0))
                for L in OPTION_LETTERS
            },
            "epsilon": state.epsilon,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))

    def _cyclic_rollout_prob_matrix(self, question_row: Any) -> np.ndarray:
        """Build the 4x4 probability matrix needed by Eq.(7) for one question.

        Runs score_options() once per cyclic permutation of the option texts.
        Each call shifts which content appears under which label (A/B/C/D) while
        keeping the labels fixed, so position and content effects are separable
        when the rows are averaged in log space by Eq.(7).

        Returns shape (4, 4): mat[k, j] = P(model picks letter j | permutation k).
        Falls back to a uniform row if score_options() raises.
        """
        canon = self._build_options(question_row)
        permutations = PermutationRunner._generate_permutations(canon)
        prompts = [
            PermutationRunner._build_permuted_prompt(
                question_row,
                perm,
                self._prompts["direct_mcq"],
            )
            for perm in permutations
        ]

        # Uniform fallback used when score_options() fails for a permutation.
        uni = np.ones(len(OPTION_LETTERS), dtype=np.float64) / len(OPTION_LETTERS)
        rows: list[np.ndarray] = []
        for prompt in prompts:
            try:
                score_result = self.backend.score_options(prompt, list(OPTION_LETTERS))
                lp = score_result.scores
                # Convert raw log-probs dict → normalized probability vector (index = letter).
                rows.append(
                    logprob_map_to_label_distribution(lp) if lp else uni.copy()
                )
            except Exception:
                rows.append(uni.copy())

        return np.stack(rows, axis=0).astype(np.float64)

    # ------------------------------------------------------------------
    # Inference — Phase 2
    # ------------------------------------------------------------------

    def run_one(self, question_row: Any, sample_index: int) -> dict:
        """Score one eval question, apply Eq.(8) debiasing, return result row."""
        self._ensure_calibration()
        prompt = self._build_prompt(question_row)

        # Get the debiased answer letter and the raw logprobs used to produce it.
        adjusted_letter, lp_scores, latency, error = self._score_and_debias(
            prompt, question_row
        )

        # Wrap the debiased choice in a ParseResult so _build_result_row and the
        # evaluator treat it the same way as any other runner's parsed output.
        if adjusted_letter is not None:
            parse_result = ParseResult(
                final_choice=adjusted_letter,
                status=PARSE_OK,
                raw_text=None,
                normalized_text=adjusted_letter,
                reason="pride_eq8",
            )
            score_result = self._score(parse_result, question_row["correct_option"])
        else:
            parse_result = None
            score_result = None

        row = self._build_result_row(
            question_row=question_row,
            prompt=prompt,
            sample_index=sample_index,
            generation_result=None,   # PriDe uses score_options, not generate()
            latency_seconds=latency,
            parsed_result=parse_result,
            score_result=score_result,
            error=error,
        )

        # _build_result_row sets model_status="error" when generation_result is None.
        # Override it to reflect the actual score_options outcome instead.
        if error is None and adjusted_letter is not None:
            row["model_status"] = "success"
            row["error_type"] = None
            row["error_message"] = None
            row["error_stage"] = None
            row["error_retryable"] = None

        # PriDe-specific columns stored alongside the standard schema.
        row["pride_inference_mode"] = "eq8_transfer"
        row["pride_adjusted_choice"] = adjusted_letter
        # Snapshot the prior used so results are reproducible and auditable.
        row["peprior_json"] = json.dumps(self._calibration_state.peprior_probs)
        row["option_logprob_json"] = (
            json.dumps(lp_scores) if lp_scores is not None else None
        )
        return row

    def _score_and_debias(
            self,
            prompt: str,
            question_row: Any,
    ) -> tuple[str | None, dict[str, float] | None, float, str | None]:
        """Run one score_options() call and apply Eq.(8) debiasing.

        Returns (adjusted_letter, raw_logprobs, latency_seconds, error_message).
        adjusted_letter is None if score_options() failed or returned empty scores.
        """
        start = time.perf_counter()
        try:
            score_result_obj = self.backend.score_options(prompt, list(OPTION_LETTERS))
            lp_scores = score_result_obj.scores
        except Exception as exc:
            latency = time.perf_counter() - start
            logger.warning(
                "PriDe: score_options failed for question %s — %s",
                question_row["question_id"],
                exc,
            )
            return None, None, latency, str(exc)

        latency = time.perf_counter() - start

        if not lp_scores:
            logger.warning(
                "PriDe: empty logprobs for question %s — skipping debiasing.",
                question_row["question_id"],
            )
            return None, lp_scores, latency, None

        # Eq.(8): divide observed probs by the calibrated prior, renormalize, argmax.
        adjusted_letter = apply_debiased_choice_from_defaults(
            self._calibration_state,
            lp_scores,
            eps_prob=1e-12,
        )
        return adjusted_letter, lp_scores, latency, None

    def _build_prompt(self, question_row: Any) -> str:
        return build_direct_mcq_prompt(
            template=self._prompts["direct_mcq"],
            question=question_row["question_text"],
            options=self._build_options(question_row),
        )
