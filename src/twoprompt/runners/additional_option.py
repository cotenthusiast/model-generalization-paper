# src/twoprompt/runners/additional_option.py

from typing import Any

from twoprompt.pipeline.prompt_builder import build_direct_mcq_prompt
from twoprompt.runners.local_base import LocalExperimentRunner
from twoprompt.runners.direct_mcq import DirectMCQRunner

class AdditionalOptionRunner(DirectMCQRunner):
    """Runner for the additional-option condition.

    Presents the model with a standard multiple-choice question plus an
    additional "I don't know" option. Expects a single letter response.
    One backend call per question.
    """

    def _build_prompt(self, question_row: Any) -> str:
        return build_direct_mcq_prompt(
            template=self._prompts["direct_mcq"],
            question=question_row["question_text"],
            options=list(self._build_options(question_row).values()) + ["I don't know"]
        )
    
    def _build_options(self, question_row: Any) -> dict[str, str]:
        options = super()._build_options(question_row)
        options["E"] = "I don't know"
        return options