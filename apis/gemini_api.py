# apis/gemini_api.py
import os
import google.generativeai as genai
import time

# This module implements a wrapper around Google Generative AI (Gemini) using the official SDK.
# IMPORTANT: set your Gemini API key in Streamlit secrets (or env var) as GEMINI_API_KEY.

def get_gemini_api_key():
    """Get Gemini API key from environment or Streamlit secrets."""
    # Try environment variable first
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        return api_key
    
    # Try Streamlit secrets
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and 'GEMINI_API_KEY' in st.secrets:
            return st.secrets['GEMINI_API_KEY']
    except:
        pass
    
    # Try reading from secrets.toml directly for command line usage
    try:
        import toml
        secrets_path = ".streamlit/secrets.toml"
        if os.path.exists(secrets_path):
            with open(secrets_path, 'r') as f:
                secrets = toml.load(f)
                return secrets.get('GEMINI_API_KEY')
    except:
        pass
    
    return None

GEMINI_API_KEY = get_gemini_api_key()

def query_gemini(model: str, prompt: str, as_text: bool = True, retries: int = 3, timeout: int = 90) -> str:
    """Call Gemini using the official Google SDK and return text output."""
    if not GEMINI_API_KEY:
        return "GEMINI_API_KEY is not set. Add it to environment or .streamlit/secrets.toml"
    
    # Configure the API
    genai.configure(api_key=GEMINI_API_KEY)
    
    # Use the correct working model name - use latest available model
    if model.startswith("gemini-1.5-flash") or model == "gemini-1.5-flash":
        model_name = "gemini-2.0-flash"  # Use newer model
    elif model.startswith("gemini-"):
        model_name = model
    else:
        model_name = "gemini-2.0-flash"  # Default to latest
    
    # Create the model instance
    try:
        model_instance = genai.GenerativeModel(model_name)
    except Exception as e:
        return f"[ERROR: Model {model_name} not found] {e}"
    
    # Simple retry with backoff for transient issues
    last_err = None
    for attempt in range(retries):
        try:
            # Generate content
            response = model_instance.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=2048,  # Increased for complete answers
                )
            )
            
            if response and response.text:
                return response.text
            else:
                return f"Unexpected response format: {str(response)[:500]}"
                
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            return f"[ERROR calling Gemini ({model_name})] {e}"
    
    return f"[ERROR calling Gemini ({model_name})] Failed after {retries} attempts."