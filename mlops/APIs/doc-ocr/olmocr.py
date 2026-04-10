import base64
import requests
from pathlib import Path
from typing import Optional, Union
from PIL import Image
import os
import tempfile
import re
from collections import Counter
from dotenv import load_dotenv

load_dotenv()


class OCRClient:
    def __init__(self, base_url: str = os.getenv("MODEL_URL", 'https://ocr-ft-dgx.nimar.gov.pk')):
        """
        Initialize OCR client for vision-language model inference.
        
        Args:
            base_url: Base URL of the vLLM server
        """
        self.base_url = base_url.rstrip('/')
        self.chat_endpoint = f"{self.base_url}/v1/chat/completions"
    
    def resize_image(self, image_path: Union[str, Path], max_size: int = 1500) -> str:
        """
        Resize image if it's too large (cross-platform).
        """
        img = Image.open(image_path)
        
        if max(img.size) <= max_size:
            return str(image_path)
        
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        temp_dir = tempfile.gettempdir()
        resized_filename = f"resized_{Path(image_path).name}"
        resized_path = os.path.join(temp_dir, resized_filename)
        
        if img.mode in ('RGBA', 'LA', 'P'):
            img = img.convert('RGB')
        
        img.save(resized_path, quality=85, optimize=True)
        
        print(f"Image resized: {img.size} -> saved to {resized_path}")
        return resized_path
    
    def encode_image(self, image_path: Union[str, Path]) -> str:
        """Encode image to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def get_mime_type(self, image_path: Union[str, Path]) -> str:
        """Get MIME type for image."""
        image_ext = Path(image_path).suffix.lower()
        return {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }.get(image_ext, 'image/jpeg')

    def _deduplicate_text(self, text: str) -> str:
        """
        Remove duplicate/repetitive text from OCR output.
        Enhanced to handle OCR hallucinations common with non-Latin scripts (Urdu/Arabic).
        
        Handles:
        - Consecutive duplicate lines
        - Non-consecutive duplicate lines (hallucination loops)
        - Inline repeated phrases
        """
        if not text or len(text) < 50:
            return text
        
        # Step 1: Clean markdown formatting issues
        text = self._clean_markdown(text)
        
        # Step 2: Line-level deduplication
        text = self._deduplicate_lines(text)
        
        # Step 3: Phrase-level deduplication (for inline repetitions)
        text = self._deduplicate_phrases(text)
        
        # Step 4: Clean up excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        return text.strip()

    def _clean_markdown(self, text: str) -> str:
        """
        Fix common markdown formatting issues from OCR output.
        
        Handles:
        - Removes incorrect code block wrappers around normal text
        - Converts literal \\n to actual newlines
        - Fixes escaped characters
        """
        if not text:
            return text
        
        # Step 1: Remove code blocks FIRST (before converting \n)
        # Match code blocks with literal \n or actual newlines
        # Pattern handles: ```markdown\n...\n``` or ```\n...\n```
        code_block_patterns = [
            r'```(?:markdown|text|plaintext)?\\n(.*?)\\n```',  # With literal \n
            r'```(?:markdown|text|plaintext)?\s*\n(.*?)\n\s*```',  # With actual newlines
            r'```(?:markdown|text|plaintext)?\s*(.*?)\s*```',  # Generic fallback
        ]
        
        for pattern in code_block_patterns:
            text = re.sub(pattern, r'\1', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Step 2: Convert literal \n to actual newlines (multiple passes for nested escapes)
        # Handle various escape levels
        for _ in range(3):
            text = text.replace('\\n', '\n')
        
        # Step 3: Handle other escaped characters
        for _ in range(2):
            text = text.replace('\\t', '\t')
            text = text.replace('\\"', '"')
            text = text.replace("\\'", "'")
            text = text.replace('\\#', '#')
            text = text.replace('\\*', '*')
            text = text.replace('\\-', '-')
            text = text.replace('\\|', '|')
        
        # Step 4: Clean up any remaining triple backticks
        text = re.sub(r'```\w*\s*', '', text)
        text = re.sub(r'\s*```', '', text)
        
        # Step 5: Fix broken markdown formatting
        # Ensure headers have space after #
        text = re.sub(r'^(#{1,6})([^\s#])', r'\1 \2', text, flags=re.MULTILINE)
        
        # Step 6: Clean excessive whitespace
        text = re.sub(r'\n{4,}', '\n\n\n', text)
        text = re.sub(r'[ \t]+\n', '\n', text)  # Trailing whitespace
        
        return text.strip()

    def _unwrap_false_code_blocks(self, text: str) -> str:
        """
        Remove code block markers (```) that incorrectly wrap normal text.
        Preserves code blocks that contain actual code.
        """
        # Pattern to match code blocks
        code_block_pattern = re.compile(
            r'```(?:markdown|text|plaintext|)?\s*\n?(.*?)\n?```',
            re.DOTALL | re.IGNORECASE
        )
        
        def should_be_code_block(content: str) -> bool:
            """
            Heuristic to determine if content should actually be in a code block.
            Returns True if it looks like actual code, False if it's normal text.
            """
            content = content.strip()
            
            # If empty, not a code block
            if not content:
                return False
            
            # Check for code-like patterns
            code_indicators = [
                r'^\s*(def |class |import |from |function |const |let |var |public |private )',  # Function/class definitions
                r'[{}\[\]();]',  # Common code punctuation
                r'^\s*<[a-zA-Z]+[^>]*>',  # HTML/XML tags (but not markdown)
                r'^\s*[a-zA-Z_]\w*\s*=\s*[^=]',  # Variable assignments
                r'^\s*(if|for|while|switch|try|catch)\s*[\(:]',  # Control structures
                r'^\s*#include|^\s*#define',  # C/C++ preprocessor
                r'^\s*@\w+',  # Decorators
            ]
            
            for pattern in code_indicators:
                if re.search(pattern, content, re.MULTILINE):
                    return True
            
            # Check ratio of code-like characters
            code_chars = len(re.findall(r'[{}\[\]();=<>]', content))
            if len(content) > 0 and code_chars / len(content) > 0.05:
                return True
            
            return False
        
        def replace_code_block(match):
            content = match.group(1)
            
            if should_be_code_block(content):
                # Keep as code block
                return match.group(0)
            else:
                # Unwrap - return just the content
                return content.strip()
        
        return code_block_pattern.sub(replace_code_block, text)

    def _clean_json_wrapper(self, text: str) -> str:
        """
        Extract natural_text from JSON metadata wrapper if present.
        Some OCR models return JSON with metadata wrapping the actual text.
        """
        import json
        
        stripped = text.strip()
        
        if not stripped.startswith('{"'):
            return text
        
        # Try to parse as JSON
        try:
            data = json.loads(stripped)
            if isinstance(data, dict):
                # Look for common text fields
                for field in ['natural_text', 'text', 'content', 'extracted_text']:
                    if field in data and isinstance(data[field], str):
                        return data[field]
        except json.JSONDecodeError:
            pass
        
        # Handle truncated JSON - try to extract natural_text manually
        if '"natural_text"' in stripped:
            marker = '"natural_text":'
            marker_pos = stripped.find(marker)
            
            if marker_pos != -1:
                content_start = marker_pos + len(marker)
                remaining = stripped[content_start:].lstrip()
                
                if remaining.startswith('null'):
                    return ''
                
                if remaining.startswith('"'):
                    # Extract string content, handling escapes
                    content = remaining[1:]
                    result_chars = []
                    i = 0
                    while i < len(content):
                        if content[i] == '\\' and i + 1 < len(content):
                            next_char = content[i + 1]
                            escape_map = {'n': '\n', 't': '\t', '"': '"', '\\': '\\', 'r': '\r'}
                            if next_char in escape_map:
                                result_chars.append(escape_map[next_char])
                                i += 2
                            else:
                                result_chars.append(content[i])
                                i += 1
                        elif content[i] == '"':
                            break
                        else:
                            result_chars.append(content[i])
                            i += 1
                    return ''.join(result_chars)
        
        return text

    def _deduplicate_lines(self, text: str) -> str:
        """
        Remove duplicate lines - both consecutive and non-consecutive.
        If a line appears more than threshold times, keep only first occurrence.
        """
        lines = text.splitlines()
        
        if len(lines) < 5:
            # For very short text, just remove consecutive duplicates
            cleaned = []
            prev = None
            for line in lines:
                stripped = line.strip()
                if stripped and stripped == prev:
                    continue
                cleaned.append(line)
                prev = stripped
            return "\n".join(cleaned)
        
        # Count line occurrences (only non-empty lines)
        line_counts = Counter(line.strip() for line in lines if line.strip())
        
        # Calculate threshold: if a line appears more than 10% of total or 5+ times
        total_lines = len([l for l in lines if l.strip()])
        repeat_threshold = max(5, int(total_lines * 0.1))
        
        # Find excessively repeated lines (likely hallucinations)
        # Only consider lines with substantial content (15+ chars)
        repeated_lines = {
            line for line, count in line_counts.items()
            if count >= repeat_threshold and len(line) >= 15
        }
        
        # Build cleaned output
        cleaned = []
        seen_repeated = set()
        prev = None
        
        for line in lines:
            stripped = line.strip()
            
            # Always skip consecutive duplicates
            if stripped and stripped == prev:
                continue
            
            # For identified repeated lines, keep only first occurrence
            if stripped in repeated_lines:
                if stripped not in seen_repeated:
                    cleaned.append(line)
                    seen_repeated.add(stripped)
            else:
                cleaned.append(line)
            
            prev = stripped
        
        return "\n".join(cleaned)

    def _deduplicate_phrases(self, text: str, min_phrase_len: int = 20) -> str:
        """
        Remove phrases that repeat consecutively within the text.
        Catches patterns like: "same text same text same text" → "same text"
        
        This is particularly important for Urdu/Arabic OCR where the model
        sometimes hallucinates the same phrase repeatedly inline.
        """
        if len(text) < min_phrase_len * 3:
            return text
        
        # Use regex to find consecutive repeated patterns
        # This handles patterns of various lengths efficiently
        
        # Pattern explanation:
        # (.{20,}?) - capture group of 20+ chars (non-greedy)
        # \1{2,}    - same group repeated 2+ more times (so 3+ total)
        
        max_iterations = 10  # Safety limit to prevent infinite loops
        
        for _ in range(max_iterations):
            # Try to find and replace repeated patterns
            # Start with longer minimum lengths for efficiency
            new_text = text
            
            for min_len in [100, 50, 30, 20]:
                pattern = re.compile(
                    r'(.{' + str(min_len) + r',}?)\1{2,}',
                    re.DOTALL | re.UNICODE
                )
                new_text = pattern.sub(r'\1', new_text)
            
            # If no changes were made, we're done
            if new_text == text:
                break
            
            text = new_text
        
        return text

    def _normalize_for_comparison(self, text: str) -> str:
        """
        Normalize text for comparison purposes.
        Removes extra whitespace and normalizes Unicode characters.
        """
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def ocr_with_layout(
        self,
        image_path: Union[str, Path],
        model: Optional[str] = None,
        language: str = "English and Urdu",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        resize: bool = True
    ) -> dict:
        """
        Perform OCR on an image with layout preservation.
        """
        if resize:
            image_path = self.resize_image(image_path)
        
        base64_image = self.encode_image(image_path)
        mime_type = self.get_mime_type(image_path)
        
        print(f"Processing image: {Path(image_path).name}")
        print(f"Image size (base64): {len(base64_image) / 1024:.2f} KB")
        
        user_prompt = (
            f"Extract all visible text from this image. The image contains {language}. "
            "Only include text that is clearly visible in the image. "
            "Do not repeat any text - each line should appear only once. "
            "Preserve the logical structure of the document.\n\n"
            "IMPORTANT OUTPUT FORMAT RULES:\n"
            "- Do NOT wrap output in code blocks (no ```)\n"
            "- Use actual line breaks, not literal \\n characters\n"
            "- Output clean plain text directly\n"
            "- Do not escape markdown characters"
        )
        
        payload_formats = [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "temperature": temperature,
                "max_tokens": max_tokens
            },
            {
                "model": model if model else "default",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "temperature": temperature,
                "max_tokens": max_tokens
            },
        ]
        
        last_error = None
        for i, payload in enumerate(payload_formats):
            try:
                print(f"Trying payload format {i + 1}...")
                
                response = requests.post(
                    self.chat_endpoint,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=120
                )
                
                if response.status_code == 500:
                    print(f"  Format {i + 1} failed with 500")
                    last_error = response.text
                    continue
                
                response.raise_for_status()
                result = response.json()
                
                raw_text = result["choices"][0]["message"]["content"]
                extracted_text = self._deduplicate_text(raw_text)
                
                print(f"Success with format {i + 1}")
                
                return {
                    "success": True,
                    "text": extracted_text,
                    "model": result.get("model"),
                    "usage": result.get("usage", {}),
                    "raw_response": result
                }
                
            except requests.exceptions.RequestException as e:
                last_error = str(e)
                print(f"  Format {i + 1} error: {e}")
                continue
        
        return {
            "success": False,
            "error": f"All payload formats failed. Last error: {last_error}",
            "text": None
        }
    
    def ocr_with_markdown(
        self,
        image_path: Union[str, Path],
        model: Optional[str] = None,
        language: str = "English and Urdu",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        resize: bool = True
    ) -> dict:
        """
        Perform OCR and return result in Markdown format.
        """
        if resize:
            image_path = self.resize_image(image_path)
        
        base64_image = self.encode_image(image_path)
        mime_type = self.get_mime_type(image_path)
        
        user_prompt = (
            f"Extract all text from this image and format it as clean Markdown. "
            f"The image contains {language}. "
            "Use proper Markdown formatting: # for headers, **bold**, - for lists, | for tables.\n\n"
            "IMPORTANT OUTPUT FORMAT RULES:\n"
            "- Do NOT wrap output in code blocks (no ```)\n"
            "- Use actual line breaks, not literal \\n characters\n"
            "- Output clean markdown directly without escaping\n"
            "- Do not repeat any text - each line should appear only once\n"
            "- Headers should render properly (# Header, not \\# Header)"
        )
        
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if model:
            payload["model"] = model
        
        try:
            response = requests.post(
                self.chat_endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            raw_text = result["choices"][0]["message"]["content"]
            extracted_text = self._deduplicate_text(raw_text)
            
            return {
                "success": True,
                "text": extracted_text,
                "model": result.get("model"),
                "usage": result.get("usage", {}),
                "raw_response": result
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": str(e),
                "text": None
            }

    def health(self) -> dict:
        """
        Check health status of OCR server.
        """
        endpoint = f"{self.base_url}/health"

        try:
            response = requests.get(endpoint, timeout=10)

            return {
                "success": response.status_code == 200,
                "status_code": response.status_code,
                "response": response.text
            }

        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "status_code": None,
                "error": str(e)
            }