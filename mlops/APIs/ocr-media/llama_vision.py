"""
Llama 3.2 90B Vision Client - Stable JSON with Retry and Robust Error Handling
"""

import requests
import base64
import json
import os
import time
import re
from typing import Dict, Optional
from config import Config
import traceback

class Llama90BVisionClient:
    """Client for Meta Llama 3.2 90B Vision with optional authorization"""

    def __init__(
        self,
        base_url: str = Config.MODEL_URL,
        api_key: Optional[str] = None,
        auth_type: str = "none",  # "none", "bearer", "api_key", "basic"
        debug: bool = False
    ):
        self.base_url = base_url
        self.model = Config.MODEL_NAME
        self.api_key = api_key or os.environ.get("NIMAR_API_KEY")
        self.auth_type = auth_type
        self.debug = debug

        # More comprehensive headers
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",  # Added charset
            "User-Agent": "Llama-Vision-Client/1.0",
        }

        # Authentication setup
        if self.auth_type == "bearer" and self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
        elif self.auth_type == "api_key" and self.api_key:
            self.headers["X-API-Key"] = self.api_key
        elif self.auth_type == "basic" and self.api_key:
            credentials = base64.b64encode(self.api_key.encode()).decode()
            self.headers["Authorization"] = f"Basic {credentials}"

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _clean_markdown_response(self, text: str) -> str:
        """
        Clean LLM response to extract markdown content while preserving formatting.
        """
        # Remove only the wrapper code blocks if present (```markdown ... ```)
        text = re.sub(r'^```markdown\s*\n', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^```\s*\n', '', text)
        text = re.sub(r'\n```\s*$', '', text)
        
        # Remove common LLM prefixes that aren't part of the content
        prefixes_to_remove = [
            r'^Here is the extracted text in markdown:?\s*\n',
            r'^Here is the text in markdown format:?\s*\n',
            r'^Extracted text:?\s*\n',
            r'^The extracted text is:?\s*\n',
            r'^Output:?\s*\n',
            r'^Result:?\s*\n',
        ]
        for prefix in prefixes_to_remove:
            text = re.sub(prefix, '', text, flags=re.IGNORECASE | re.MULTILINE)
        
        # Fix common markdown issues
        # Ensure headers have proper spacing
        text = re.sub(r'^(#{1,6})([^#\s])', r'\1 \2', text, flags=re.MULTILINE)
        
        # Fix list formatting (ensure space after bullet/number)
        text = re.sub(r'^(\*|\-|\+)([^\s])', r'\1 \2', text, flags=re.MULTILINE)
        text = re.sub(r'^(\d+\.)([^\s])', r'\1 \2', text, flags=re.MULTILINE)
        
        # Ensure blank lines around headers
        lines = text.split('\n')
        formatted_lines = []
        for i, line in enumerate(lines):
            if line.strip().startswith('#'):
                # Add blank line before header if previous line exists and isn't blank
                if i > 0 and formatted_lines and formatted_lines[-1].strip():
                    formatted_lines.append('')
                formatted_lines.append(line)
                # Add blank line after header if next line exists and isn't blank
                if i < len(lines) - 1 and lines[i + 1].strip() and not lines[i + 1].strip().startswith('#'):
                    formatted_lines.append('')
            else:
                formatted_lines.append(line)
        
        text = '\n'.join(formatted_lines)
        
        # Remove excessive blank lines (more than 2)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Trim leading/trailing whitespace
        text = text.strip()
        
        return text

    def chat_completion(
        self,
        messages: list,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.9,
        **kwargs
    ) -> Dict:
        """Send chat completion request to the API."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": Config.TEMPERATURE,
            "max_tokens": max_tokens,
            "top_p": top_p,
            **kwargs,
        }

        # Debug: Print request details
        if self.debug:
            print(f"URL: {self.base_url}/v1/chat/completions")
            print(f"Headers: {self.headers}")
            print(f"Payload keys: {payload.keys()}")

        try:
            # Ensure JSON is properly formatted
            json_data = json.dumps(payload, ensure_ascii=False)
            
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=self.headers,
                data=json_data.encode('utf-8'),  # Send as bytes with UTF-8 encoding
                timeout=Config.REQUEST_TIMEOUT,
            )
            
            # Debug: Print response details
            if self.debug:
                print(f"Response Status: {response.status_code}")
                print(f"Response Headers: {response.headers}")
            
            if response.status_code == 401:
                raise Exception("Unauthorized - Invalid credentials.")
            elif response.status_code == 403:
                raise Exception("Forbidden - Access denied.")
            elif response.status_code == 415:
                raise Exception(f"Unsupported Media Type - Server expects different content type. Response: {response.text}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.RequestException as e:
            traceback.print_exc()
            # More detailed error information
            if hasattr(e, 'response') and e.response is not None:
                error_detail = f"Status: {e.response.status_code}, Body: {e.response.text[:500]}"
            else:
                error_detail = str(e)
            raise Exception(f"Request failed: {error_detail}")

    def chat_completion_alternative(
        self,
        messages: list,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.9,
        **kwargs
    ) -> Dict:
        """Alternative method using different request approach."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": Config.TEMPERATURE,
            "max_tokens": max_tokens,
            "top_p": top_p,
            **kwargs,
        }

        # Alternative headers approach
        headers = {
            "Accept": "*/*",  # Accept any response type
            "Content-Type": "application/json",
        }
        
        # Add auth if needed
        if self.auth_type == "bearer" and self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=headers,
                json=payload,  # Use json parameter instead of data
                timeout=Config.REQUEST_TIMEOUT,
            )
            
            response.raise_for_status()
            return response.json()
            
        except requests.RequestException as e:
            raise Exception(f"Request failed: {e}")

    def ocr_with_layout(
        self,
        image_path: str,
        language: str = "auto",
        temperature: float = 0.1,
        max_tokens: int = 4096,
        retries: int = 5,
        retry_delay: float = 2.0,
        markdown_level: str = "full",  # "none", "basic", "full"
        use_alternative: bool = False,  # Use alternative request method
    ) -> Dict:
        """
        Extract text from image with both raw text and cleaned markdown text.
        
        Returns:
            Dict with keys:
            - "raw_text": Original text from LLM
            - "markdown_text": Cleaned markdown (or same as raw if markdown_level="none")
        """
        base64_image = self.encode_image(image_path)

        if language.lower() == "urdu":
            lang_instruction = "The text in this image is in Urdu. Keep it in Urdu script."
        elif language.lower() == "english":
            lang_instruction = "The text in this image is in English. Keep it in English."
        else:
            lang_instruction = (
                "Detect automatically whether the text is in Urdu or English, "
                "and output it in the same language."
            )

        # Prompt based on markdown_level (same as before)
        if markdown_level == "full":
            prompt = f"""Extract all visible text from the image and format it as proper Markdown.

            {lang_instruction}

            ... (existing full markdown rules here) ...
            """
        elif markdown_level == "basic":
            prompt = f"""Extract all visible text from the image with basic markdown formatting.

            {lang_instruction}

            ... (existing basic markdown rules here) ...
            """
        else:  # "none"
            prompt = f"""Extract all visible text from the image exactly as it appears.

            {lang_instruction}

            Output the text directly:"""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ],
            }
        ]

        last_error = None
        for attempt in range(1, retries + 1):
            try:
                # Choose which method to use
                if use_alternative:
                    response = self.chat_completion_alternative(
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=0.95,
                    )
                else:
                    response = self.chat_completion(
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=0.95,
                    )

                # Extract raw text from response
                raw_text = response["choices"][0]["message"]["content"].strip()

                # Process markdown only if needed
                if markdown_level != "none":
                    markdown_text = self._clean_markdown_response(raw_text)
                else:
                    markdown_text = raw_text

                return {
                    "raw_text": raw_text,
                    "markdown_text": markdown_text,
                }

            except (KeyError, json.JSONDecodeError, ValueError) as e:
                last_error = str(e)
                print(e)
                if attempt < retries:
                    time.sleep(retry_delay)
            except Exception as e:
                traceback.print_exc()
                print(e)
                last_error = str(e)
                if attempt < retries:
                    time.sleep(retry_delay)

        raise Exception(f"OCR failed: {last_error}")


    def health_check(self) -> bool:
        """Check if the API is available and ready."""
        try:
            response = requests.get(
                f"{self.base_url}/v1/health/ready",
                headers={"Accept": "application/json"},
                timeout=Config.REQUEST_TIMEOUT,
            )
            return response.status_code == 200
        except Exception:
            return False