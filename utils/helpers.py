# Complete utils/helpers.py - Additional utility functions

def clean_text(text: str) -> str:
    """Clean and preprocess text for analysis"""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = " ".join(text.split())
    
    # Basic cleaning
    text = text.strip()
    
    return text

def format_error(error: Exception, context: str = "") -> str:
    """Format error messages consistently"""
    error_msg = str(error)
    if len(error_msg) > 100:
        error_msg = error_msg[:100] + "..."
    
    if context:
        return f"[{context} Error: {error_msg}]"
    else:
        return f"[Error: {error_msg}]"

def validate_inputs(question: str, selected_models: list) -> tuple[bool, str]:
    """Validate user inputs"""
    if not question or not question.strip():
        return False, "Please enter a question first!"
    
    if not selected_models:
        return False, "Please select at least one model!"
    
    if len(question.strip()) < 10:
        return False, "Please enter a more detailed question!"
    
    return True, ""

def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text for display purposes"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."
    