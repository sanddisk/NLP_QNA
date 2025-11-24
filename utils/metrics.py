from typing import Dict, List, Optional
import re

# ROUGE
from rouge_score import rouge_scorer

# BLEU
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# BERTScore
from bert_score import score as bert_score


def _normalize_text_for_lang(text: str, lang: Optional[str]) -> str:
    if not text:
        return ""
    s = text.strip()
    # Normalize Indic sentence terminators to periods for Lsum
    s = s.replace("।", ".").replace("॥", ".")
    # Collapse excessive whitespace
    s = re.sub(r"\s+", " ", s)
    return s


def compute_rouge(reference: str, candidate: str, lang: Optional[str] = None) -> Dict[str, float]:
    # For non-English, disable stemming to avoid English-only stemmer harming scores
    use_stemmer = (lang in (None, "en", "english"))
    ref = _normalize_text_for_lang(reference, lang)
    cand = _normalize_text_for_lang(candidate, lang)
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=use_stemmer)
    scores = scorer.score(ref, cand)
    return {
        "ROUGE-1": scores["rouge1"].fmeasure,
        "ROUGE-2": scores["rouge2"].fmeasure,
        "ROUGE-Lsum": scores["rougeLsum"].fmeasure,
    }


def compute_bleu(reference: str, candidate: str) -> float:
    smoothie = SmoothingFunction().method3
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    # Use up to 4-gram BLEU
    bleu = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothie)
    return float(bleu)


def compute_bertscore(reference: str, candidate: str, lang: Optional[str] = None) -> Dict[str, float]:
    # Default to English for optimal performance
    # Use English-specific model for better accuracy on English text
    lang_code = lang or "en"
    P, R, F1 = bert_score([candidate], [reference], lang=lang_code, rescale_with_baseline=True)
    return {"BERTScore_P": float(P[0]), "BERTScore_R": float(R[0]), "BERTScore_F1": float(F1[0])}


def evaluate_text(reference: str, candidates: Dict[str, str], lang: Optional[str] = None) -> List[Dict[str, float]]:
    results = []
    for name, text in candidates.items():
        if not isinstance(text, str) or not text.strip():
            continue
        row = {"Name": name}
        try:
            row.update(compute_rouge(reference, text, lang=lang))
        except Exception:
            row.update({"ROUGE-1": 0.0, "ROUGE-2": 0.0, "ROUGE-Lsum": 0.0})
        try:
            row["BLEU"] = compute_bleu(reference, text)
        except Exception:
            row["BLEU"] = 0.0
        try:
            row.update(compute_bertscore(reference, text, lang=lang))
        except Exception:
            row.update({"BERTScore_P": 0.0, "BERTScore_R": 0.0, "BERTScore_F1": 0.0})
        results.append(row)
    return results


