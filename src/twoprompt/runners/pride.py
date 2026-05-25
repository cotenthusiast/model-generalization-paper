# src/twoprompt/runners/pride.py

"""PriDe runner (Zheng et al., ICLR 2024) — local backend, score_options() for logits.

The position prior is estimated on a calibration set that is disjoint from
the evaluation split. All evaluation questions are scored with Eq.(8)
transfer debiasing; the Eq.(1) estimation-only path has been removed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from twoprompt.backends.types import LocalGenerationConfig
from twoprompt.parsing.types import PARSE_OK, ParseResult
from twoprompt.pipeline.prompt_builder import build_direct_mcq_prompt
from twoprompt.runners.local_base import LocalExperimentRunner
from twoprompt.runners.permutation import PermutationRunner
from twoprompt.runners.pride_debias import (
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

_SIDE_SCHEMA_VERSION = 3


def _pick_calibration_rows(
        full: list[dict],
        k: int,
        seed: int,
) -> tuple[list[str], list[dict]]:
    """Random subset of size *k* from *full* using a seeded shuffle."""
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
    All evaluation rows use ``eq8_transfer`` mode.

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
        self._calibration_questions: list[dict] = list(calibration_questions or [])

        self._calibration_ready: bool = False
        self._calibration_state: CalibrationState = calibration_state_uniform()

    def _sidecar_path(self) -> Path:
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
        self._ensure_calibration()
        return [self.run_one(row, i) for i, row in enumerate(question_rows)]

    def _ensure_calibration(self) -> None:
        if self._calibration_ready:
            return

        cal_qids, cal_rows = _pick_calibration_rows(
            self._calibration_questions,
            self._calibration_n,
            self._calibration_seed,
        )
        sorted_ids = tuple(sorted(cal_qids))

        # Try to reuse a matching sidecar from a previous run.
        path = self._sidecar_path()
        if sorted_ids and path.exists():
            try:
                blob = json.loads(path.read_text())
                if (
                    blob.get("schema_version") == _SIDE_SCHEMA_VERSION
                    and tuple(sorted(blob.get("calibration_question_ids") or [])) == sorted_ids
                    and int(blob.get("calibration_seed", -1)) == self._calibration_seed
                ):
                    self._calibration_state = calibration_state_from_sidecar(blob)
                    self._calibration_ready = True
                    logger.info(
                        "PriDe loaded sidecar (K=%d) → %s",
                        len(sorted_ids),
                        path,
                    )
                    return
            except (json.JSONDecodeError, KeyError, OSError, TypeError, ValueError) as exc:
                logger.warning("PriDe sidecar unreadable (%s); refitting.", exc)

        if not cal_rows:
            logger.warning(
                "PriDe: no calibration questions available — using uniform prior."
            )
            self._calibration_state = calibration_state_uniform()
        else:
            prior_vectors: list[np.ndarray] = []
            sample_index_hint = 0
            for row in cal_rows:
                roll_mat = self._cyclic_rollout_prob_matrix(row, sample_index_hint)
                sample_index_hint += len(OPTION_LETTERS)
                prior_vectors.append(equation7_prior_from_rollouts(roll_mat))

            pep_global = average_prior_probability_vectors(prior_vectors)
            self._calibration_state = CalibrationState(
                peprior_probs={
                    OPTION_LETTERS[i]: float(pep_global[i])
                    for i in range(len(OPTION_LETTERS))
                },
                epsilon=1e-12,
                estimation_question_ids=tuple(sorted_ids),
            )

        self._calibration_ready = True

        sidecar_payload = {
            "schema_version": _SIDE_SCHEMA_VERSION,
            "version": self._calibration_state.version,
            "calibration_seed": self._calibration_seed,
            "n_options": len(OPTION_LETTERS),
            "calibration_question_ids": list(sorted_ids),
            "peprior_probs": {
                L: float(self._calibration_state.peprior_probs.get(L, 0.0))
                for L in OPTION_LETTERS
            },
            "epsilon": self._calibration_state.epsilon,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sidecar_payload, indent=2))

    def _cyclic_rollout_prob_matrix(
            self,
            question_row: Any,
            sample_index_base: int,
    ) -> np.ndarray:
        """Run score_options for 4 cyclic permutations → shape-(4,4) probability matrix."""
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

        rows: list[np.ndarray] = []
        uni = np.ones(len(OPTION_LETTERS), dtype=np.float64) / len(OPTION_LETTERS)
        for prompt in prompts:
            try:
                score_result = self.backend.score_options(prompt, list(OPTION_LETTERS))
                lp = score_result.scores
                rows.append(
                    logprob_map_to_label_distribution(lp) if lp else uni.copy()
                )
            except Exception:
                rows.append(uni.copy())

        return np.stack(rows, axis=0).astype(np.float64)

    def run_one(self, question_row: Any, sample_index: int) -> dict:
        self._ensure_calibration()

        options = self._build_options(question_row)
        prompt = self._build_prompt(question_row)
        generation_result, latency, error = self._call_backend(prompt)

        parsed_result_raw = None
        score_raw = None
        score_adjusted = None
        adjusted_letter: str | None = None
        lp_scores: dict[str, float] | None = None

        if generation_result is not None:
            parsed_result_raw, score_raw = self._parse_and_score(
                raw_text=generation_result.raw_text,
                correct_option=question_row["correct_option"],
                options=options,
            )
            try:
                score_result_obj = self.backend.score_options(prompt, list(OPTION_LETTERS))
                lp_scores = score_result_obj.scores
            except Exception as exc:
                logger.warning(
                    "PriDe: score_options failed for question %s — %s",
                    question_row["question_id"],
                    exc,
                )

            if not lp_scores:
                logger.warning(
                    "PriDe: empty logprobs for question %s — skipping debiasing.",
                    question_row["question_id"],
                )
            else:
                adjusted_letter = apply_debiased_choice_from_defaults(
                    self._calibration_state,
                    lp_scores,
                    eps_prob=1e-12,
                )
                adj_parse = ParseResult(
                    final_choice=adjusted_letter,
                    status=PARSE_OK,
                    raw_text=generation_result.raw_text,
                    normalized_text=(
                        adjusted_letter
                        if parsed_result_raw is None
                        else parsed_result_raw.normalized_text
                    ),
                    reason="pride_eq8",
                )
                score_adjusted = self._score(adj_parse, question_row["correct_option"])

        row = self._build_result_row(
            question_row=question_row,
            prompt=prompt,
            sample_index=sample_index,
            generation_result=generation_result,
            latency_seconds=latency,
            parsed_result=parsed_result_raw,
            score_result=score_raw,
            error=error,
        )
        row["pride_inference_mode"] = "eq8_transfer"
        row["pride_adjusted_choice"] = adjusted_letter
        row["is_correct_raw"] = score_raw.is_correct if score_raw else None
        row["score_status_raw"] = score_raw.status if score_raw else None
        row["peprior_json"] = json.dumps(self._calibration_state.peprior_probs)
        row["option_logprob_json"] = (
            json.dumps(lp_scores) if lp_scores is not None else None
        )
        if score_adjusted is not None:
            row["score_status"] = score_adjusted.status
            row["is_correct"] = score_adjusted.is_correct

        return row

    def _build_prompt(self, question_row: Any) -> str:
        return build_direct_mcq_prompt(
            template=self._prompts["direct_mcq"],
            question=question_row["question_text"],
            option_a=question_row["choice_a"],
            option_b=question_row["choice_b"],
            option_c=question_row["choice_c"],
            option_d=question_row["choice_d"],
        )
