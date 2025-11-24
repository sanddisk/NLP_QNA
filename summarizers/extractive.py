# Complete Fixed summarizers/extractive.py
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from typing import List
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
try:
    # Use the cached multilingual model from utils.compare
    from utils.compare import _get_embed_model
except Exception:
    _get_embed_model = None

ENGLISH_STOPWORDS = [
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being'
]

HINDI_STOPWORDS = [
    "और", "का", "की", "के", "है", "हैं", "को", "में", "पर", "से", "यह",
    "था", "थे", "था", "या", "कर", "हो", "गया", "गई", "भी", "जो", "उस",
    "उन", "उनके", "आप", "हम", "ते", "जब", "अगर", "लेकिन", "क्यों", "किस",
    "किसी", "कुछ", "कई", "सभी", "इसी", "उसी"
]


def _stopwords_for_language(language: str) -> List[str]:
    lang = (language or "").lower()
    if lang in {"en", "english"}:
        return ENGLISH_STOPWORDS
    if lang in {"hi", "hindi"}:
        return HINDI_STOPWORDS
    return []


def _summarize_with(summarizer, text: str, sentences_count: int = 2, language: str = "en") -> str:
    """Helper function to apply any sumy summarizer to text"""
    try:
        if not text or not text.strip():
            return "[Error: Empty text provided]"
        
        # Simple multilingual preprocessing: normalize Indic danda to period for sentence splitting
        normalized_text = text.replace("।", ".")

        # If not English, Sumy tokenization can fail; use multilingual fallback below
        # Normalize language code: accept both "en"/"english"
        lang_normalized = (language or "en").lower()
        if lang_normalized not in ("english", "en"):
            return _multilingual_extractive_fallback(normalized_text, sentences_count)

        # Create parser and tokenizer for English
        parser = PlaintextParser.from_string(normalized_text, Tokenizer("english"))
        
        # Check if document has enough sentences
        if len(parser.document.sentences) < sentences_count:
            # If text has fewer sentences than requested, return the original text
            return text
        
        # Generate summary
        summary_sentences = summarizer(parser.document, sentences_count)
        
        # Convert sentences to string
        summary_text = " ".join(str(sentence) for sentence in summary_sentences)
        
        if not summary_text.strip():
            return "[Error: Generated empty summary]"
            
        return summary_text
        
    except Exception as e:
        return f"[Error: {str(e)[:100]}...]"


def _split_sentences(text: str) -> List[str]:
    # Split on common sentence delimiters including Indic danda
    parts = re.split(r"(?<=[\.\!\?\u0964\u0965])\s+", text.strip())
    sentences = [s.strip() for s in parts if len(s.strip()) > 0]
    return sentences


def _multilingual_extractive_fallback(text: str, sentences_count: int) -> str:
    sentences = _split_sentences(text)
    # If too few sentences, return original text
    if len(sentences) <= sentences_count:
        return text
    # Remove very short sentences
    filtered = [s for s in sentences if len(s.split()) >= 3]
    if len(filtered) < sentences_count:
        filtered = sentences
    try:
        if _get_embed_model is None:
            # Fallback: simple length-based pick
            ranked = sorted(filtered, key=lambda s: len(s), reverse=True)[:sentences_count]
            return " ".join(ranked)
        model = _get_embed_model()
        sent_emb = model.encode(filtered, show_progress_bar=False)
        doc_emb = np.mean(sent_emb, axis=0, keepdims=True)
        sims = cosine_similarity(sent_emb, doc_emb).ravel()
        top_idx = np.argsort(-sims)[:sentences_count]
        top_idx_sorted = sorted(top_idx)  # preserve original order among selected
        selected = [filtered[i] for i in top_idx_sorted]
        return " ".join(selected)
    except Exception:
        ranked = sorted(filtered, key=lambda s: len(s), reverse=True)[:sentences_count]
        return " ".join(ranked)

def summarize_lexrank(text: str, sentences_count: int = 2, language: str = "en") -> str:
    """Generate extractive summary using LexRank algorithm (Graph-based)"""
    summarizer = LexRankSummarizer()
    summarizer.stop_words = _stopwords_for_language(language)
    return _summarize_with(summarizer, text, sentences_count, language)

def summarize_luhn(text: str, sentences_count: int = 2, language: str = "en") -> str:
    """Generate extractive summary using Luhn algorithm (Frequency-based)"""
    summarizer = LuhnSummarizer()
    summarizer.stop_words = _stopwords_for_language(language)
    return _summarize_with(summarizer, text, sentences_count, language)

def summarize_lsa(text: str, sentences_count: int = 2, language: str = "en") -> str:
    """Generate extractive summary using LSA algorithm (Latent Semantic Analysis)"""
    summarizer = LsaSummarizer()
    summarizer.stop_words = _stopwords_for_language(language)
    return _summarize_with(summarizer, text, sentences_count, language)