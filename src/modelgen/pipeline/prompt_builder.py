# src/modelgen/pipeline/prompt_builder.py

from pathlib import Path


_TEMPLATE_NAMES = ("direct_mcq", "free_text", "option_matching", "text_extraction", "abcd")


def load_prompt_templates(version: str, prompts_dir: Path) -> dict[str, str]:
    """Load all prompt templates for a given version from disk.

    Templates are plain text files with Python str.format-style placeholders.
    Each version lives in its own subdirectory under prompts_dir:

        prompts_dir / v1 / direct_mcq.txt
        prompts_dir / v1 / free_text.txt
        prompts_dir / v1 / option_matching.txt

    Args:
        version: Version string matching a subdirectory name (e.g. "v1").
        prompts_dir: Root directory containing versioned prompt folders.

    Returns:
        Dict mapping template name to raw template string.

    Raises:
        FileNotFoundError: If the version directory or any template file is missing.
    """
    version_dir = prompts_dir / version
    if not version_dir.is_dir():
        raise FileNotFoundError(
            f"Prompt version directory not found: {version_dir}. "
            f"Create {version_dir}/ with direct_mcq.txt, free_text.txt, "
            f"and option_matching.txt to use prompt version {version!r}."
        )

    templates = {}
    for name in _TEMPLATE_NAMES:
        path = version_dir / f"{name}.txt"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing prompt template: {path}. "
                f"Expected one of: {[f'{n}.txt' for n in _TEMPLATE_NAMES]}"
            )
        templates[name] = path.read_text(encoding="utf-8")

    return templates


def _build_options_block(options: dict[str, str], label_style: str = "letter") -> str:
    """Render an options dict as a labeled block, one line per option.

    Args:
        options: Mapping from label (e.g. "A", "E") to option text. Rendered
            in dict order using the dict's own keys — never regenerated from
            position — so a label is never reassigned to the wrong option
            text when a question has fewer than 4 real options (e.g. an
            additional_option "E" must stay "E" even when "D" is missing,
            not silently shift into the gap left by D).
        label_style: "letter" for "A. text" lines (used by direct_mcq,
            text_extraction, option_matching), or "dash" for "- text" lines
            (used by abcd, which deliberately avoids letter cues).

    Returns:
        Newline-joined block of labeled option lines.
    """
    if label_style == "dash":
        return "\n".join(f"- {text}" for text in options.values())
    return "\n".join(f"{letter}. {text}" for letter, text in options.items())


def build_direct_mcq_prompt(
    template: str,
    question: str,
    options: dict[str, str],
) -> str:
    """Format the direct MCQ template with question and option text.

    Args:
        template: Raw template string from load_prompt_templates.
        question: Question stem to present to the model.
        options: Mapping from label to option text, e.g. {"A": ..., "B": ...}.

    Returns:
        Fully formatted prompt string.
    """
    options_block = _build_options_block(options, "letter")
    return template.format(question=question, options=options_block)


def build_free_text_prompt(template: str, question: str) -> str:
    """Format the free-text template with a question stem.

    Args:
        template: Raw template string from load_prompt_templates.
        question: Question stem to present without answer options.

    Returns:
        Fully formatted prompt string.
    """
    return template.format(question=question)


def build_text_extraction_prompt(
    template: str,
    question: str,
    options: dict[str, str],
) -> str:
    """Format the text-extraction template for stage one of the text-extraction condition.

    Options are shown with A/B/C/D labels; the model is instructed to respond
    in free text rather than stating a letter.

    Args:
        template: Raw template string from load_prompt_templates.
        question: Question stem.
        options: Mapping from label to option text, e.g. {"A": ..., "B": ...}.

    Returns:
        Fully formatted prompt string.
    """
    options_block = _build_options_block(options, "letter")
    return template.format(question=question, options=options_block)


def build_abcd_prompt(
    template: str,
    question: str,
    options: dict[str, str],
) -> str:
    """Format the ABCD template for stage one of the uniform-label condition.

    Options are shown under neutral dash labels (no A/B/C/D letter cues) so
    the model must respond in free text.

    Args:
        template: Raw template string from load_prompt_templates.
        question: Question stem.
        options: Mapping from label to option text; only the text is rendered.

    Returns:
        Fully formatted prompt string.
    """
    options_block = _build_options_block(options, "dash")
    return template.format(question=question, options=options_block)


def build_option_matching_prompt(
    template: str,
    question: str,
    free_text: str,
    options: dict[str, str],
) -> str:
    """Format the option-matching template for stage two of two-stage methods.

    Args:
        template: Raw template string from load_prompt_templates.
        question: Original question stem.
        free_text: Free-text answer produced in stage one.
        options: Mapping from label to option text, e.g. {"A": ..., "B": ...}.

    Returns:
        Fully formatted prompt string.
    """
    options_block = _build_options_block(options, "letter")
    return template.format(question=question, free_text=free_text, options=options_block)
