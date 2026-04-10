from config import Config
from fastapi import HTTPException, Header
import re
import json

Config = Config()

def validate_api_key(api_key: str = Header(None, alias="X-API-Key")):
    keys = Config.USER_API_KEYS
    if api_key not in keys.values():
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key



def remove_repeated_text(text: str, min_phrase_length: int = 20, max_repetitions: int = 2) -> str:
    """
    Remove consecutively repeated phrases/sentences from OCR output.
    
    Args:
        text: The OCR output text
        min_phrase_length: Minimum length of phrase to consider for deduplication
        max_repetitions: Maximum allowed repetitions before removal
    
    Returns:
        Cleaned text with repetitions removed
    """
    if not text or len(text) < min_phrase_length * 2:
        return text
    
    # Strategy 1: Remove exact consecutive duplicate sentences/phrases
    # Split by common delimiters (., ،, newlines)
    sentences = re.split(r'([.،\n]+)', text)
    
    cleaned_parts = []
    prev_sentence = None
    repeat_count = 0
    
    for part in sentences:
        stripped = part.strip()
        
        # Keep delimiters
        if re.match(r'^[.،\n]+$', part):
            cleaned_parts.append(part)
            continue
        
        if stripped == prev_sentence and len(stripped) >= min_phrase_length:
            repeat_count += 1
            if repeat_count < max_repetitions:
                cleaned_parts.append(part)
        else:
            cleaned_parts.append(part)
            prev_sentence = stripped
            repeat_count = 0
    
    result = ''.join(cleaned_parts)
    
    # Strategy 2: Detect and remove repeated substring patterns
    # Find phrases that repeat more than 3 times consecutively
    for phrase_len in range(min_phrase_length, min(200, len(result) // 3)):
        pattern = re.compile(r'(.{' + str(phrase_len) + r',}?)\1{3,}', re.DOTALL)
        result = pattern.sub(r'\1', result)
    
    return result.strip()



def _remove_repeated_text(text: str) -> str:
    """Ultra-fast OCR hallucination fix"""
    if not text or len(text) < 100:
        return text
    
    sample = text[:50]
    pos = text.find(sample, 50)
    
    if 0 < pos < 600 and text.startswith(text[:pos] * 2):
        return text[:pos]
    
    return text


def clean_page_text(text: str) -> str:
    """Extract actual text from OCR response, handling JSON metadata wrapper"""
    if not text:
        return text
    
    stripped = text.strip()
    
    # Case 1: Entire text is the JSON metadata object
    if stripped.startswith('{"primary_language"'):
        # Try JSON parsing first
        try:
            data = json.loads(stripped)
            if isinstance(data, dict) and 'natural_text' in data:
                natural = data.get('natural_text')
                return natural if natural else ''
        except json.JSONDecodeError:
            pass
        
        # Fallback: Extract natural_text using regex (handles malformed JSON)
        match = re.search(r'"natural_text"\s*:\s*(?:"((?:[^"\\]|\\.)*)"|null)', stripped, re.DOTALL)
        if match:
            if match.group(1) is not None:
                # Unescape JSON string
                extracted = match.group(1)
                extracted = extracted.replace('\\n', '\n')
                extracted = extracted.replace('\\t', '\t')
                extracted = extracted.replace('\\"', '"')
                extracted = extracted.replace('\\\\', '\\')
                return extracted
            else:
                return ''
        
        # If it looks like JSON but we couldn't extract, return empty
        if '"natural_text"' in stripped:
            return ''
    
    # Case 2: JSON is embedded somewhere in the text
    pattern = r'\{"primary_language":"[^"]*","is_rotation_valid":(true|false),"rotation_correction":\d+,"is_table":(true|false),"is_diagram":(true|false),"natural_text":(null|"(?:[^"\\]|\\.)*")\}'
    
    match = re.search(pattern, text, re.DOTALL)
    if match:
        try:
            json_str = match.group(0)
            data = json.loads(json_str)
            natural_text = data.get('natural_text')
            
            if natural_text:
                return text[:match.start()] + natural_text + text[match.end():]
            else:
                return (text[:match.start()] + text[match.end():]).strip()
        except:
            # Fallback: just remove the JSON block
            return (text[:match.start()] + text[match.end():]).strip()
    
    # text = remove_repeated_text(text)
    text = _remove_repeated_text(text)

    return text



import re
from collections import Counter


def clean_ocr_output(text: str) -> str:
    """
    Main function to clean OCR output.
    Call this as the FINAL step before returning text to user.
    
    Args:
        text: Raw OCR output (possibly with \n, code blocks, duplicates)
    
    Returns:
        Clean, properly formatted markdown text
    """
    if not text:
        return text
    
    # Step 1: Fix literal newlines and escapes
    text = fix_literal_escapes(text)
    
    # Step 2: Remove code block wrappers
    text = remove_code_blocks(text)
    
    # Step 3: Remove duplicate lines (OCR hallucination)
    text = remove_duplicate_lines(text)
    
    # Step 4: Remove duplicate phrases
    text = remove_duplicate_phrases(text)
    
    # Step 5: Final cleanup
    text = final_cleanup(text)
    
    return text.strip()


def fix_literal_escapes(text: str) -> str:
    """
    Convert literal escape sequences and newlines to markdown-compatible format.
    All newlines become <br> for proper markdown rendering.
    """
    if not text:
        return text
    
    # Step 1: Convert literal \n (backslash + n) to real newlines first
    max_iterations = 20
    for _ in range(max_iterations):
        old_text = text
        text = text.replace('\\n', '\n')
        if text == old_text:
            break
    
    # Step 2: Normalize all newline types
    text = text.replace('\r\n', '\n')
    text = text.replace('\r', '\n')
    
    # Step 3: Replace ALL newlines with <br>
    text = text.replace('\n', '<br>')
    
    # Step 4: Handle other escapes
    escape_map = [
        ('\\t', '    '),
        ('\\\\', '\\'),
        ('\\"', '"'),
        ("\\'", "'"),
        ('\\#', '#'),
        ('\\*', '*'),
        ('\\-', '-'),
        ('\\|', '|'),
        ('\\[', '['),
        ('\\]', ']'),
        ('\\(', '('),
        ('\\)', ')'),
        ('\\`', '`'),
        ('\\>', '>'),
    ]
    
    for escaped, unescaped in escape_map:
        text = text.replace(escaped, unescaped)
    
    return text


def remove_code_blocks(text: str) -> str:
    """
    Remove code block wrappers (```) that incorrectly wrap normal text.
    """
    if not text:
        return text
    
    # Pattern 1: ```language\ncontent\n```
    text = re.sub(
        r'```(?:markdown|text|plaintext|json|)?\s*\n?(.*?)\n?\s*```',
        r'\1',
        text,
        flags=re.DOTALL | re.IGNORECASE
    )
    
    # Pattern 2: Any remaining triple backticks
    text = re.sub(r'```\w*', '', text)
    text = re.sub(r'```', '', text)
    
    return text


def remove_duplicate_lines(text: str, threshold_ratio: float = 0.1, min_count: int = 5) -> str:
    """
    Remove lines that repeat excessively (OCR hallucination).
    
    Args:
        text: Input text
        threshold_ratio: Line is considered duplicate if it appears more than this ratio of total lines
        min_count: Minimum repetition count to consider as duplicate
    
    Returns:
        Text with duplicate lines removed (keeps first occurrence)
    """
    if not text:
        return text
    
    lines = text.split('\n')
    
    if len(lines) < 5:
        # For short text, just remove consecutive duplicates
        return remove_consecutive_duplicates(text)
    
    # Count line occurrences
    line_counts = Counter(line.strip() for line in lines if line.strip())
    total_lines = len([l for l in lines if l.strip()])
    
    if total_lines == 0:
        return text
    
    # Calculate threshold
    threshold = max(min_count, int(total_lines * threshold_ratio))
    
    # Find excessively repeated lines (hallucinations)
    # Only consider substantial lines (15+ chars)
    repeated_lines = {
        line for line, count in line_counts.items()
        if count >= threshold and len(line) >= 15
    }
    
    # Build cleaned output - keep first occurrence of repeated lines
    cleaned = []
    seen_repeated = set()
    prev_stripped = None
    
    for line in lines:
        stripped = line.strip()
        
        # Skip consecutive duplicates
        if stripped and stripped == prev_stripped:
            continue
        
        # For repeated lines, keep only first occurrence
        if stripped in repeated_lines:
            if stripped not in seen_repeated:
                cleaned.append(line)
                seen_repeated.add(stripped)
        else:
            cleaned.append(line)
        
        prev_stripped = stripped
    
    return '\n'.join(cleaned)


def remove_consecutive_duplicates(text: str) -> str:
    """Remove consecutive duplicate lines."""
    lines = text.split('\n')
    cleaned = []
    prev = None
    
    for line in lines:
        stripped = line.strip()
        if stripped and stripped == prev:
            continue
        cleaned.append(line)
        prev = stripped
    
    return '\n'.join(cleaned)


def remove_duplicate_phrases(text: str, min_phrase_len: int = 30) -> str:
    """
    Remove phrases that repeat consecutively within text.
    Catches: "same text same text same text" → "same text"
    """
    if not text or len(text) < min_phrase_len * 3:
        return text
    
    # Use regex to find and remove consecutive repeated patterns
    # Pattern: capture 30+ chars, then same thing repeated 2+ more times
    max_iterations = 10
    
    for _ in range(max_iterations):
        old_text = text
        
        # Try different minimum lengths
        for min_len in [100, 50, 30]:
            if len(text) < min_len * 3:
                continue
            
            pattern = re.compile(
                r'(.{' + str(min_len) + r',}?)\1{2,}',
                re.DOTALL | re.UNICODE
            )
            text = pattern.sub(r'\1', text)
        
        if text == old_text:
            break
    
    return text


def final_cleanup(text: str) -> str:
    """
    Final cleanup: fix whitespace, ensure proper markdown formatting.
    """
    if not text:
        return text
    
    # Ensure headers have space after #
    text = re.sub(r'^(#{1,6})([^\s#])', r'\1 \2', text, flags=re.MULTILINE)
    
    # Remove excessive blank lines (more than 2 consecutive)
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    
    # Remove trailing whitespace EXCEPT two spaces (markdown line break)
    # Only remove 3+ trailing spaces, or single space/tabs
    text = re.sub(r'[ \t]{3,}\n', '  \n', text)  # 3+ spaces → 2 spaces
    text = re.sub(r'(?<! )[ \t]\n', '\n', text)  # Single space/tab (not preceded by space) → nothing
    
    # Remove leading/trailing whitespace from whole text
    text = text.strip()
    
    return text


# Convenience function for direct use
def clean_page_text(text: str) -> str:
    """
    Alias for clean_ocr_output for backward compatibility.
    """
    return clean_ocr_output(text)
