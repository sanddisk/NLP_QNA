# apis/groq_api.py
import os
import requests
import time
from typing import Optional


def get_groq_api_key() -> Optional[str]:
    """Get Groq API key from environment or Streamlit secrets."""
    key = os.environ.get("GROQ_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return None


GROQ_API_KEY = get_groq_api_key()
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def query_groq(model: str, prompt: str, as_text: bool = True) -> str:
    """Call Groq's OpenAI-compatible chat completions endpoint and return text output."""
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to environment or .streamlit/secrets.toml"
        )

    url = f"{GROQ_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
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
                return f"Groq 400: {resp.text[:500]}"
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
    return f"Groq request failed after retries: {last_err}"
