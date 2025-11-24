# Enhanced utils/compare.py - Adding question-to-text similarity
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as tfidf_cosine
import gensim
from gensim import corpora, models
import re
from collections import Counter
import streamlit as st


# Global variables for caching
EMBED_MODEL = None
TFIDF_VECTORIZER = None


@st.cache_resource
def _get_embed_model():
    """Get cached sentence transformer model (optimized for English)."""
    # Using English-specific model for better performance on English text
    # This model provides superior semantic understanding for English
    return SentenceTransformer("all-MiniLM-L6-v2")


def _get_tfidf_vectorizer():
    """Get TF-IDF vectorizer (optimized for English)"""
    global TFIDF_VECTORIZER
    if TFIDF_VECTORIZER is None:
        TFIDF_VECTORIZER = TfidfVectorizer(
            max_features=1000,
            # Use English stop words for better English text processing
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True,
            strip_accents='unicode'
        )
    return TFIDF_VECTORIZER


def tfidf_cosine_similarity(texts: List[str]) -> np.ndarray:
    """Calculate TF-IDF cosine similarity matrix."""
    try:
        vectorizer = _get_tfidf_vectorizer()
        tfidf_matrix = vectorizer.fit_transform(texts)
        similarity_matrix = tfidf_cosine(tfidf_matrix)
        return similarity_matrix
    except Exception as e:
        st.error(f"TF-IDF similarity error: {e}")
        # Return identity matrix as fallback
        n = len(texts)
        return np.eye(n)


def soft_cosine_similarity(texts: List[str]) -> np.ndarray:
    """Calculate Soft Cosine similarity matrix using Gensim."""
    try:
        # Tokenize texts properly
        tokenized_texts = []
        for text in texts:
            # Clean and tokenize
            clean_text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
            tokens = clean_text.split()
            if len(tokens) > 0:
                tokenized_texts.append(tokens)
            else:
                tokenized_texts.append(['empty'])
        
        # Create dictionary and corpus
        dictionary = corpora.Dictionary(tokenized_texts)
        corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
        
        # Use TF-IDF model for better similarity calculation
        tfidf_model = models.TfidfModel(corpus)
        corpus_tfidf = tfidf_model[corpus]
        
        # Calculate similarity matrix
        n = len(texts)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    similarity_matrix[i][j] = 1.0
                else:
                    # Calculate cosine similarity between TF-IDF vectors
                    sim = gensim.matutils.cossim(corpus_tfidf[i], corpus_tfidf[j])
                    similarity_matrix[i][j] = max(0, sim)  # Ensure non-negative
        
        return similarity_matrix
        
    except Exception as e:
        st.warning(f"Soft Cosine similarity failed, using TF-IDF fallback: {e}")
        return tfidf_cosine_similarity(texts)


def sentence_bert_similarity(texts: List[str]) -> np.ndarray:
    """Calculate Sentence-BERT similarity matrix."""
    try:
        model = _get_embed_model()
        embeddings = model.encode(texts, show_progress_bar=False)
        similarity_matrix = cosine_similarity(embeddings)
        return similarity_matrix
    except Exception as e:
        st.error(f"Sentence-BERT similarity error: {e}")
        # Return identity matrix as fallback
        n = len(texts)
        return np.eye(n)


# NEW: Question-to-text similarity functions
def calculate_question_similarities(question: str, texts: List[str], text_labels: List[str]) -> Dict[str, List[float]]:
    """Calculate similarity between question and each text using all three methods."""
    # Combine question with texts for matrix calculation
    all_texts = [question] + texts
    
    results = {}
    
    # TF-IDF similarity
    try:
        tfidf_matrix = tfidf_cosine_similarity(all_texts)
        question_similarities = tfidf_matrix[0, 1:].tolist()  # First row, excluding self
        results['TF-IDF'] = question_similarities
    except:
        results['TF-IDF'] = [0.0] * len(texts)
    
    # Soft Cosine similarity
    try:
        soft_matrix = soft_cosine_similarity(all_texts)
        question_similarities = soft_matrix[0, 1:].tolist()
        results['Soft Cosine'] = question_similarities
    except:
        results['Soft Cosine'] = [0.0] * len(texts)
    
    # Sentence-BERT similarity
    try:
        sbert_matrix = sentence_bert_similarity(all_texts)
        question_similarities = sbert_matrix[0, 1:].tolist()
        results['Sentence-BERT'] = question_similarities
    except:
        results['Sentence-BERT'] = [0.0] * len(texts)
    
    return results


def calculate_all_similarities(texts: List[str], model_names: List[str]) -> Dict[str, Dict]:
    """Calculate all three similarity measures and return comprehensive results."""
    results = {}
    
    # Calculate TF-IDF Cosine Similarity
    with st.spinner("🔤 Calculating TF-IDF cosine similarity..."):
        tfidf_matrix = tfidf_cosine_similarity(texts)
        results['TF-IDF Cosine'] = {
            'matrix': tfidf_matrix,
            'description': 'Lexical similarity based on term frequency',
            'type': 'lexical'
        }
    
    # Calculate Soft Cosine Similarity  
    with st.spinner("🔀 Calculating Soft Cosine similarity..."):
        soft_matrix = soft_cosine_similarity(texts)
        results['Soft Cosine'] = {
            'matrix': soft_matrix,
            'description': 'Word-level semantic similarity',
            'type': 'semantic'
        }
    
    # Calculate Sentence-BERT Similarity
    with st.spinner("🧠 Calculating Sentence-BERT similarity..."):
        sbert_matrix = sentence_bert_similarity(texts)
        results['Sentence-BERT'] = {
            'matrix': sbert_matrix,
            'description': 'Context-aware semantic similarity',
            'type': 'contextual'
        }
    
    return results


def get_similarity_stats(similarity_matrix: np.ndarray, method_name: str) -> Dict:
    """Calculate statistics for a similarity matrix."""
    # Remove diagonal (self-similarity = 1.0)
    mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
    off_diagonal = similarity_matrix[mask]
    
    stats = {
        'method': method_name,
        'mean': float(np.mean(off_diagonal)),
        'std': float(np.std(off_diagonal)),
        'min': float(np.min(off_diagonal)),
        'max': float(np.max(off_diagonal)),
        'median': float(np.median(off_diagonal))
    }
    
    return stats


def extract_shared_ngrams(texts: List[str], n: int = 2, min_occurrence: int = 2):
    """Extract shared n-grams across texts."""
    def _tokenize_ngrams(text: str, n: int = 2):
        # Unicode-aware punctuation removal. Try third-party regex with \p classes; fallback to stdlib re.
        s = text
        try:
            import regex as re2  # type: ignore
            s = re2.sub(r"[\p{P}\p{S}]", " ", s)
        except Exception:
            s = re.sub(r"[\u0000-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E]", " ", s)
            s = s.replace("।", " ").replace("॥", " ")
        tokens = s.split()
        if len(tokens) < n:
            return []
        return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    
    # Extract n-grams from each text
    doc_ngrams = [set(_tokenize_ngrams(text, n=n)) for text in texts]
    
    # Count occurrences across documents
    counter = Counter()
    for ngram_set in doc_ngrams:
        for ngram in ngram_set:
            counter[ngram] += 1
    
    # Filter by minimum occurrence and return top results
    shared_ngrams = [ngram for ngram, count in counter.items() if count >= min_occurrence]
    shared_ngrams_sorted = sorted(shared_ngrams, key=lambda x: -counter[x])
    
    return shared_ngrams_sorted[:20]


# Backward compatibility function
def similarity_matrix(texts: List[str]):
    """Return sentence-BERT similarity matrix for backward compatibility."""
    matrix = sentence_bert_similarity(texts)
    return matrix.tolist()