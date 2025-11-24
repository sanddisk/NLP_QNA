# Complete Fixed summarizers/llm_summary.py
from apis.gemini_api import query_gemini


def summarize_with_gemini(text: str, lang_hint: str | None = None) -> str:
    """
    Generate intelligent summary using Gemini LLM.
    Handles multiple model answers and identifies conflicts or consensus.
    """
    try:
        if not text or not text.strip():
            return "[Error: No text to summarize]"
        
        language_names = {"en": "English", "hi": "Hindi"}
        lang_code = (lang_hint or "").strip()
        lang_name = language_names.get(lang_code, "the user's language")
        language_line = (
            f"Always write the summary in {lang_name} (language code: {lang_code})."
            if lang_code
            else "Match the dominant language of the input."
        )

        # Enhanced prompt for better summarization
        prompt = f"""You are an expert research assistant tasked with summarizing multiple AI model responses to a question.

{language_line}

Your task:
1. Analyze the following set of model answers
2. Identify key points and main themes
3. Note any conflicts or disagreements between models
4. Provide a concise, factual summary (3-4 sentences)
5. If there are conflicting answers, mention this briefly
 6. Preserve the user's language if possible.

Text to summarize:
{text}

Please provide a clear, objective summary that captures the essence of all responses."""

        # Query Gemini with the enhanced prompt (using latest model)
        summary = query_gemini("gemini-2.0-flash", prompt)
        
        if not summary or not summary.strip():
            return "[Error: Generated empty summary from Gemini]"
            
        return summary.strip()
        
    except Exception as e:
        return f"[Error: Gemini summarization failed - {str(e)[:100]}...]"