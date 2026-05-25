# src/twoprompt/backends/hf_causal_lm.py

from __future__ import annotations

import logging

from twoprompt.backends.base import LocalBackend
from twoprompt.backends.types import (
    LocalGenerationConfig,
    LocalModelMetadata,
    ModelGenerationResult,
    ModelOptionScoreResult,
)

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = LocalGenerationConfig()


class HFCausalLMBackend(LocalBackend):
    """Local inference backend using HuggingFace transformers AutoModelForCausalLM.

    Works with any causal LM loadable via from_pretrained, including Qwen and Llama.

    torch and transformers are imported lazily inside load() so that importing
    this module does not require them to be installed. They are required at
    runtime: install with pip install -e '.[local]'.
    """

    def __init__(
        self,
        model_path: str,
        family: str,
        size_label: str | None = None,
        device: str | None = None,
    ) -> None:
        """
        Args:
            model_path:  HuggingFace hub ID or absolute path to a local model directory.
            family:      Model family string, e.g. "qwen" or "llama".
            size_label:  Informational size label, e.g. "7B". Not used for loading.
            device:      "cuda", "cpu", "mps", or "auto". Auto-detected if None.
                         Use "auto" for multi-GPU via device_map=auto on large models.
        """
        self._model_path = model_path
        self._family = family
        self._size_label = size_label
        self._device = device or self._detect_device()
        self._model = None
        self._tokenizer = None
        self._torch = None
        self._F = None
        self._loaded = False

    @staticmethod
    def _detect_device() -> str:
        """Return the best available device without requiring torch at import time."""
        try:
            import torch
        except ImportError:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @property
    def metadata(self) -> LocalModelMetadata:
        return LocalModelMetadata(
            model_name=self._model_path.split("/")[-1],
            model_path=self._model_path,
            family=self._family,
            backend_type="hf_transformers",
            device=self._device,
            size_label=self._size_label,
        )

    def load(self) -> None:
        """Load tokenizer and model weights into memory.

        Uses float16 and trust_remote_code=True, which is standard for Qwen and Llama.
        Sets model to eval mode after loading.
        """
        if self._loaded:
            return

        try:
            import torch
            import torch.nn.functional as F
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError as exc:
            raise ImportError(
                "torch and transformers are required for local model inference. "
                "Install with: pip install -e '.[local]'"
            ) from exc

        self._torch = torch
        self._F = F

        logger.info("Loading tokenizer: %s", self._model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_path,
            trust_remote_code=True,
        )

        logger.info("Loading model: %s  device=%s", self._model_path, self._device)
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_path,
            torch_dtype=torch.float16,
            device_map=self._device,
            trust_remote_code=True,
        )
        self._model.eval()
        self._loaded = True
        logger.info("Model ready: %s", self._model_path)

    def _get_input_device(self):
        """Return the device input tensors should be moved to before a forward pass.

        With device_map="auto" the model may span multiple GPUs; inputs go to
        whichever device holds the embedding layer (the first parameter).
        """
        if self._device == "auto":
            return next(self._model.parameters()).device
        return self._device

    def generate(
        self,
        prompt: str,
        config: LocalGenerationConfig | None = None,
    ) -> ModelGenerationResult:
        if not self._loaded:
            raise RuntimeError("Call load() before generate().")

        cfg = config or _DEFAULT_CONFIG

        if cfg.seed is not None:
            self._torch.manual_seed(cfg.seed)

        inputs = self._tokenizer(prompt, return_tensors="pt")
        prompt_len = inputs["input_ids"].shape[-1]
        input_device = self._get_input_device()
        inputs = {k: v.to(input_device) for k, v in inputs.items()}

        generate_kwargs: dict = {
            "max_new_tokens": cfg.max_new_tokens,
            "do_sample": cfg.do_sample,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        # temperature is only meaningful when sampling; passing it with greedy
        # decoding triggers a HuggingFace warning.
        if cfg.do_sample and cfg.temperature > 0.0:
            generate_kwargs["temperature"] = cfg.temperature

        with self._torch.no_grad():
            outputs = self._model.generate(**inputs, **generate_kwargs)

        # Decode only the newly generated tokens, not the prompt tokens.
        generated_ids = outputs[0][prompt_len:]
        raw_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

        last_token_id = outputs[0][-1].item()
        finish_reason = "eos" if last_token_id == self._tokenizer.eos_token_id else "length"

        return ModelGenerationResult(
            raw_text=raw_text,
            prompt_tokens=prompt_len,
            completion_tokens=len(generated_ids),
            finish_reason=finish_reason,
            metadata={"model_path": self._model_path, "device": str(self._device)},
        )

    def score_options(
        self,
        prompt: str,
        options: list[str],
    ) -> ModelOptionScoreResult:
        if not self._loaded:
            raise RuntimeError("Call load() before score_options().")

        if not options:
            raise ValueError("options must not be empty.")

        # Resolve each option to a single token ID.
        # add_special_tokens=False gives the bare token without BOS or padding.
        # If a letter encodes to more than one token the method cannot proceed:
        # there is no single logit position to read for a multi-token span.
        option_token_ids: dict[str, int] = {}
        for opt in options:
            ids = self._tokenizer.encode(opt, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(
                    f"Option '{opt}' encodes to {len(ids)} token(s) in this tokenizer. "
                    "score_options requires each option to be exactly one token. "
                    "Try passing the space-prefixed form (e.g. ' A') or check the tokenizer."
                )
            option_token_ids[opt] = ids[0]

        inputs = self._tokenizer(prompt, return_tensors="pt")
        input_device = self._get_input_device()
        inputs = {k: v.to(input_device) for k, v in inputs.items()}

        with self._torch.no_grad():
            # Single forward pass; logits shape is (1, seq_len, vocab_size).
            outputs = self._model(**inputs)

        # Position -1 predicts the token that follows the entire prompt.
        last_logits = outputs.logits[0, -1, :]
        log_probs = self._F.log_softmax(last_logits, dim=-1)

        scores = {opt: log_probs[tid].item() for opt, tid in option_token_ids.items()}

        return ModelOptionScoreResult(
            scores=scores,
            raw_logprobs=dict(scores),
            metadata={"model_path": self._model_path, "device": str(self._device)},
        )
