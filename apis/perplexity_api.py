# apis/perplexity_api.py
import os
import requests
import time
from typing import Optional


def get_perplexity_api_key() -> Optional[str]:
    """Get Perplexity API key from environment or Streamlit secrets."""
    key = os.environ.get("PERPLEXITY_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "PERPLEXITY_API_KEY" in st.secrets:
            return st.secrets["PERPLEXITY_API_KEY"]
    except Exception:
        pass
    return None


PERPLEXITY_API_KEY = get_perplexity_api_key()
PERPLEXITY_BASE_URL = "https://api.perplexity.ai/chat/completions"


def query_perplexity(model: str, prompt: str, as_text: bool = True) -> str:
    """Call Perplexity's chat completions endpoint and return text output."""
    if not PERPLEXITY_API_KEY:
        raise RuntimeError(
            "PERPLEXITY_API_KEY is not set. Add it to environment or .streamlit/secrets.toml"
        )

    url = PERPLEXITY_BASE_URL
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that provides accurate, well-researched answers."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 2048,  # Increased for complete answers
    }

    last_err = None
    for attempt in range(2):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=90)
            if resp.status_code == 400:
                # Return the server's message to help choose correct model names
                return f"Perplexity 400: {resp.text[:500]}"
            resp.raise_for_status()
            data = resp.json()
            if not as_text:
                return str(data)
            try:
                choices = data.get("choices") or []
                if choices:
                    content = choices[0]["message"]["content"]
                    return content
                return str(data)
            except Exception:
                return str(data)
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    return f"Perplexity request failed after retries: {last_err}"
