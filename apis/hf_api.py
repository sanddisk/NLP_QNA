# apis/hf_api.py
import os
import requests
import time


def get_hf_token():
    """Get HuggingFace token from environment or Streamlit secrets."""
    # Try environment variable first
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    
    # Try Streamlit secrets
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and 'HF_TOKEN' in st.secrets:
            return st.secrets['HF_TOKEN']
    except:
        pass
    
    return None

HF_TOKEN = get_hf_token()

HF_BASE = "https://api-inference.huggingface.co/models"

HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}


def query_hf(model: str, prompt: str, max_length: int = 1024) -> str:
    """Query Hugging Face Inference API for a given model and return generated text."""
    url = f"{HF_BASE}/{model}"
    
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": max_length, "temperature": 0.7},  # Default 1024 for complete answers
        "options": {"wait_for_model": True}
    }

    last_err = None
    for attempt in range(3):
        try:
            resp = requests.post(url, headers=HEADERS, json=payload, timeout=90)
            # If auth missing or invalid, surface that explicitly
            if resp.status_code in (401, 403):
                return "[HF AUTH ERROR] Provide a valid HF_TOKEN in secrets or environment."
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and data and isinstance(data[0], dict) and "generated_text" in data[0]:
                return data[0]["generated_text"]
            if isinstance(data, dict) and "error" in data:
                return f"[HF ERROR] {data['error']}"
            return str(data)
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    return f"HF request failed after retries: {last_err}"