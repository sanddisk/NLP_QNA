# Complete Fixed summarizers/abstractive.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartForConditionalGeneration,
    pipeline,
)


@dataclass(frozen=True)
class AbstractiveModelConfig:
    key: str
    model_name: str
    label: str
    family: str
    languages: List[str]
    default_max_length: int = 160
    default_min_length: int = 30
    chunk_chars: int = 2000
    language_prompts: Optional[Dict[str, str]] = None
    description: str = ""


MODEL_REGISTRY: Dict[str, AbstractiveModelConfig] = {
    "bart-cnn": AbstractiveModelConfig(
        key="bart-cnn",
        model_name="facebook/bart-large-cnn",
        label="BART Large CNN",
        family="bart",
        languages=["en"],
        default_max_length=140,
        default_min_length=25,
        chunk_chars=1600,
        description="English-only abstractive summarization model (fast, reliable English output).",
    ),
    "mbart-large-50": AbstractiveModelConfig(
        key="mbart-large-50",
        model_name="facebook/mbart-large-50-many-to-many-mmt",
        label="mBART Large 50",
        family="mbart",
        languages=["en", "hi"],
        default_max_length=150,
        default_min_length=25,
        chunk_chars=1800,
        description="Multilingual seq2seq model from Facebook supporting 50 languages including Hindi and English.",
    ),
    "mt5-small": AbstractiveModelConfig(
        key="mt5-small",
        model_name="google/mt5-small",
        label="mT5 Small",
        family="mt5",
        languages=["en", "hi"],
        default_max_length=140,
        default_min_length=25,
        chunk_chars=1500,
        language_prompts={
            "en": "summarize in English:",
            "hi": "हिंदी में सारांश लिखें:",
        },
        description="Google mT5-small multilingual T5 model fine-tuned via prompt instructions.",
    ),
    "mt5-base": AbstractiveModelConfig(
        key="mt5-base",
        model_name="google/mt5-base",
        label="mT5 Base",
        family="mt5",
        languages=["en", "hi"],
        default_max_length=180,
        default_min_length=30,
        chunk_chars=1800,
        language_prompts={
            "en": "summarize in English:",
            "hi": "हिंदी में सारांश लिखें:",
        },
        description="Larger mT5 base model for higher-quality multilingual summaries (slower on CPU).",
    ),
    "indicbart": AbstractiveModelConfig(
        key="indicbart",
        model_name="ai4bharat/IndicBART",
        label="IndicBART",
        family="indicbart",
        languages=["hi"],
        default_max_length=150,
        default_min_length=25,
        chunk_chars=1400,
        language_prompts={
            "hi": "हिंदी में सारांश लिखें:",
        },
        description="IndicBART model from AI4Bharat geared towards Indic-language summarization.",
    ),
}


class AbstractiveSummarizer:
    """Loads and runs multilingual abstractive summarization models."""

    def __init__(self, model_key: str = "mbart-large-50", prefer_offline: bool = False):
        if model_key not in MODEL_REGISTRY:
            raise ValueError(f"Unknown abstractive model '{model_key}'.")

        self.config = MODEL_REGISTRY[model_key]
        self.model_key = model_key
        self.model_name = self.config.model_name
        self.pipe = None
        self._tokenizer = None
        self.prefer_offline = prefer_offline
        
        self._load_pipeline()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    @staticmethod
    def list_available_models() -> Dict[str, str]:
        return {key: cfg.label for key, cfg in MODEL_REGISTRY.items()}

    @staticmethod
    def get_model_config(model_key: str) -> AbstractiveModelConfig:
        return MODEL_REGISTRY[model_key]

    def supports_language(self, language: str) -> bool:
        return language in self.config.languages

    # ------------------------------------------------------------------
    # Loading and configuration
    # ------------------------------------------------------------------
    def _load_pipeline(self) -> None:
        try:
            if self.config.family == "mbart":
                tokenizer = None
                errors = []

                common_kwargs = {"local_files_only": self.prefer_offline}
                for loader, kwargs in (
                    (MBart50TokenizerFast.from_pretrained, {}),
                    (AutoTokenizer.from_pretrained, {"use_fast": False}),
                ):
                    try:
                        merged = {**common_kwargs, **kwargs}
                        tokenizer = loader(self.model_name, **merged)
                        break
                    except Exception as err:
                        errors.append(str(err))

                if tokenizer is None:
                    raise RuntimeError(
                        "; ".join(errors)
                    )

                self._tokenizer = tokenizer
                model = MBartForConditionalGeneration.from_pretrained(self.model_name, local_files_only=self.prefer_offline)
            else:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=self.prefer_offline)
                model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, local_files_only=self.prefer_offline)

            # Set a sane tokenizer max length to avoid warnings
            try:
                if getattr(self._tokenizer, "model_max_length", None) is None or self._tokenizer.model_max_length > 100000:
                    self._tokenizer.model_max_length = 1024
            except Exception:
                pass

            self.pipe = pipeline(
                "summarization",
                model=model,
                tokenizer=self._tokenizer,
                framework="pt",
                device=-1,
                model_kwargs={"torch_dtype": torch.float32},
                truncation=True,
            )
        except Exception as exc:
            hint = (
                "Failed to load abstractive summarizer '{}'. "
                "If you are using mBART models, ensure the 'sentencepiece' and 'sacremoses' "
                "packages are installed. Original error: {}"
            ).format(self.model_name, exc)
            raise RuntimeError(hint) from exc

    # ------------------------------------------------------------------
    # Summarization
    # ------------------------------------------------------------------
    def summarize(
        self,
        text: str,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        language: str = "en",
    ) -> str:
        if not text or not text.strip():
            return "[Error: No text to summarize]"

        text = text.strip()
        language = language or "en"
        if not self.supports_language(language):
            # Fallback to first supported language to avoid runtime errors
            language = self.config.languages[0]

        max_length = max_length or self.config.default_max_length
        min_length = min_length or self.config.default_min_length
        chunk_chars = self.config.chunk_chars

        if len(text.split()) < min_length:
            return text

        prepared_text = self._prepare_text(text, language)

        if len(prepared_text) <= chunk_chars:
            return self._summarize_once(prepared_text, max_length, min_length)

        chunks = self._chunk_text(prepared_text, chunk_chars)
        chunk_summaries: List[str] = []
        for idx, chunk in enumerate(chunks):
            summary = self._summarize_once(chunk, max_length // 2, max(10, min_length // 2))
            chunk_summaries.append(summary)

        combined = " ".join(chunk_summaries)
        if len(combined) <= chunk_chars:
            return combined

        return self._summarize_once(combined, max_length, min_length)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _prepare_text(self, text: str, language: str) -> str:
        if self.config.family == "mbart":
            lang_codes = {"en": "en_XX", "hi": "hi_IN"}
            lang_code = lang_codes.get(language, lang_codes.get(self.config.languages[0], "en_XX"))
            try:
                self._tokenizer.src_lang = lang_code
                self._tokenizer.tgt_lang = lang_code
            except Exception:
                pass
            return text

        prompts = self.config.language_prompts or {}
        prompt = prompts.get(language) or prompts.get("default")
        if prompt:
            prompt = prompt.strip()
            if not text.lower().startswith(prompt.lower()):
                return f"{prompt} {text}".strip()

        if self.config.family in {"mt5", "indicbart"}:
            if not text.lower().startswith("summarize"):
                return f"summarize: {text}".strip()

        return text

    def _chunk_text(self, text: str, chunk_chars: int) -> List[str]:
        chunks: List[str] = []
        start = 0
        length = len(text)

        while start < length:
            end = min(start + chunk_chars, length)
            chunk = text[start:end]
            
            if end < length:
                boundary = max(chunk.rfind("।"), chunk.rfind("."), chunk.rfind("!"), chunk.rfind("?"))
                if boundary != -1 and boundary > len(chunk) * 0.5:
                    end = start + boundary + 1
                    chunk = text[start:end]
            
            chunks.append(chunk.strip())
            start = end

        return [c for c in chunks if c]

    def _summarize_once(self, text: str, max_length: int, min_length: int) -> str:
        if not self.pipe:
            return "[Error: Model not loaded]"
            
        try:
            result = self.pipe(
                text,
                max_new_tokens=max(64, min(200, max_length if max_length is not None else self.config.default_max_length)),
                do_sample=False,
                num_beams=4,
                no_repeat_ngram_size=3,
                repetition_penalty=1.15,
                length_penalty=1.0,
                early_stopping=True,
                truncation=True,
                clean_up_tokenization_spaces=True,
            )
            
            if isinstance(result, list) and result:
                item = result[0]
                if isinstance(item, dict) and "summary_text" in item:
                    return item["summary_text"].strip()
                if isinstance(item, dict) and "generated_text" in item:
                    return item["generated_text"].strip()
            
            return str(result)
        except Exception as exc:
            return f"[Summarization Error: {str(exc)[:120]}...]"


def available_abstractive_models() -> Dict[str, str]:
    """Convenience helper for UI components."""
    return AbstractiveSummarizer.list_available_models()