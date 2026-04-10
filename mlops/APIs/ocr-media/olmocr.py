"""
Optimized OlmOCR Client for Media OCR Service
- Async HTTP with connection pooling
- In-memory image processing (no disk I/O)
- Concurrency control with semaphores
- Thread pool for CPU-bound operations
- Text deduplication to prevent repeated OCR output
"""

import base64
import httpx
import logging
import json
import ast
import re
import asyncio
from io import BytesIO
from pathlib import Path
from typing import Optional, Union, Dict, List
from PIL import Image
from functools import lru_cache
import hashlib
import os

logger = logging.getLogger(__name__)


class OCRClient:
    def __init__(
        self,
        base_url: str = None,
        max_connections: int = 100,
        max_concurrent_requests: int = 50,
        timeout: float = 120.0,
        enable_cache: bool = True,
        cache_size: int = 256
    ):
        """
        Initialize async OCR client with connection pooling.
        
        Args:
            base_url: Base URL of the vLLM server
            max_connections: Maximum HTTP connections in pool
            max_concurrent_requests: Max concurrent OCR requests
            timeout: Request timeout in seconds
            enable_cache: Enable result caching for identical images
            cache_size: LRU cache size
        """
        self.base_url = (base_url or os.getenv("MODEL_URL", "https://ocr-ft-dgx.nimar.gov.pk")).rstrip("/")
        self.chat_endpoint = f"{self.base_url}/v1/chat/completions"
        self.timeout = timeout
        self.enable_cache = enable_cache
        
        # Concurrency control
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)
        
        # Connection pool configuration
        self._limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_connections // 2,
            keepalive_expiry=30.0
        )
        
        # Lazy-initialized async client
        self._client: Optional[httpx.AsyncClient] = None
        
        # Simple in-memory cache
        self._cache: Dict[str, Dict[str, str]] = {}
        self._cache_size = cache_size
        
        logger.info(f"Initialized async OCRClient: {self.base_url}, max_concurrent={max_concurrent_requests}")
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                limits=self._limits,
                timeout=httpx.Timeout(self.timeout, connect=10.0),
                headers={"Content-Type": "application/json"}
            )
        return self._client
    
    async def close(self):
        """Close the HTTP client and cleanup resources."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
        self._cache.clear()
    
    def _compute_cache_key(self, image_data: bytes) -> str:
        """Compute hash key for caching."""
        return hashlib.sha256(image_data).hexdigest()[:32]
    
    def _get_cached(self, key: str) -> Optional[Dict[str, str]]:
        """Get cached result if available."""
        if self.enable_cache and key in self._cache:
            logger.debug(f"Cache hit for key: {key[:8]}...")
            return self._cache[key]
        return None
    
    def _set_cached(self, key: str, value: Dict[str, str]):
        """Cache a result with LRU eviction."""
        if not self.enable_cache:
            return
        
        # Simple LRU: remove oldest if at capacity
        if len(self._cache) >= self._cache_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        self._cache[key] = value
    
    @staticmethod
    def _resize_image_sync(image_data: bytes, max_size: int = 1500) -> tuple[bytes, str]:
        """
        Resize image in memory (CPU-bound, runs in thread pool).
        
        Returns:
            Tuple of (resized_image_bytes, mime_type)
        """
        img = Image.open(BytesIO(image_data))
        
        # Determine mime type from format
        format_map = {"JPEG": "image/jpeg", "PNG": "image/png", "GIF": "image/gif", "WEBP": "image/webp"}
        img_format = img.format or "PNG"
        mime_type = format_map.get(img_format, "image/png")
        
        # Check if resize needed
        if max(img.size) <= max_size:
            return image_data, mime_type
        
        # Resize maintaining aspect ratio
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Convert to RGB if necessary (for JPEG)
        if img_format == "JPEG" and img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        
        # Save to bytes
        output = BytesIO()
        save_format = "JPEG" if img_format == "JPEG" else "PNG"
        img.save(output, format=save_format, quality=85, optimize=True)
        
        logger.debug(f"Image resized from {img.size}")
        return output.getvalue(), mime_type
    
    async def _resize_image(self, image_data: bytes, max_size: int = 1500) -> tuple[bytes, str]:
        """Async wrapper for image resizing - runs in thread pool."""
        return await asyncio.to_thread(self._resize_image_sync, image_data, max_size)
    
    @staticmethod
    def _deduplicate_text(text: str) -> str:
        """
        Remove repeated lines/paragraphs from OCR output.
        
        Handles common OCR hallucination patterns where the model
        repeats the same text multiple times.
        
        Args:
            text: Raw OCR text that may contain repetitions
            
        Returns:
            Deduplicated text with unique paragraphs/lines only
        """
        if not text or not text.strip():
            return text
        
        # Try paragraph-level deduplication first (split by double newline)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if len(paragraphs) > 1:
            # Deduplicate paragraphs using normalized text as key
            seen_normalized: set = set()
            unique_paragraphs: List[str] = []
            
            for para in paragraphs:
                # Normalize: collapse whitespace for comparison
                normalized = ' '.join(para.split())
                
                if normalized not in seen_normalized:
                    seen_normalized.add(normalized)
                    unique_paragraphs.append(para)
            
            return '\n\n'.join(unique_paragraphs)
        
        # Fall back to line-level deduplication
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        seen_normalized: set = set()
        unique_lines: List[str] = []
        
        for line in lines:
            normalized = ' '.join(line.split())
            
            if normalized not in seen_normalized:
                seen_normalized.add(normalized)
                unique_lines.append(line)
        
        return '\n'.join(unique_lines)

    @staticmethod
    def _parse_response_content(response_content: str) -> str:
        """Parse OCR response to extract natural text."""
        def extract_from_data(data):
            if isinstance(data, list) and len(data) > 0:
                json_str = data[0]
                if isinstance(json_str, str):
                    try:
                        inner_data = json.loads(json_str)
                        return inner_data.get("natural_text", "")
                    except json.JSONDecodeError:
                        return json_str
                elif isinstance(json_str, dict):
                    return json_str.get("natural_text", "")
                else:
                    return str(json_str)
            elif isinstance(data, dict):
                return data.get("natural_text", "")
            return ""

        natural_text = ""
        
        # Try JSON parse
        try:
            parsed_data = json.loads(response_content)
            natural_text = extract_from_data(parsed_data)
        except json.JSONDecodeError:
            try:
                parsed_data = ast.literal_eval(response_content)
                natural_text = extract_from_data(parsed_data)
            except (ValueError, SyntaxError):
                pass

        # Try repairing truncated JSON
        if not natural_text:
            content = response_content.strip()
            if content.startswith("{"):
                try:
                    parsed_data = json.loads(content + '"}')
                    natural_text = extract_from_data(parsed_data)
                except:
                    pass
            elif content.startswith("["):
                for suffix in ['"}]', '"}\']']:
                    try:
                        try:
                            parsed_data = json.loads(content + suffix)
                        except:
                            parsed_data = ast.literal_eval(content + suffix)
                        text = extract_from_data(parsed_data)
                        if text:
                            natural_text = text
                            break
                    except:
                        continue

        # Regex fallback
        if not natural_text:
            match = re.search(r'"natural_text"\s*:\s*"(.*?)(?<!\\)"', response_content, re.DOTALL)
            if not match:
                match = re.search(r'"natural_text"\s*:\s*"(.*)$', response_content, re.DOTALL)
            if match:
                natural_text = match.group(1)
                try:
                    natural_text = json.loads(f'"{natural_text}"')
                except:
                    pass

        # Final fallback
        if not natural_text:
            content = response_content.strip()
            if not content.startswith("{") and not content.startswith("["):
                natural_text = response_content

        return natural_text or ""

    async def ocr_with_layout(
        self,
        image_path: Optional[Union[str, Path]] = None,
        image_data: Optional[bytes] = None,
        model: Optional[str] = None,
        language: str = "English and Urdu",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        resize: bool = True
    ) -> Dict[str, str]:
        """
        Perform async OCR on an image with layout preservation.
        
        Args:
            image_path: Path to the image file (optional if image_data provided)
            image_data: Raw image bytes (optional if image_path provided)
            model: Model name (optional)
            language: Languages to recognize
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            resize: Automatically resize large images
            
        Returns:
            Dictionary with raw_text and markdown_text
        """
        # Get image data
        if image_data is None:
            if image_path is None:
                raise ValueError("Either image_path or image_data must be provided")
            with open(image_path, "rb") as f:
                image_data = f.read()
        
        # Check cache
        cache_key = self._compute_cache_key(image_data)
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        # Acquire semaphore for concurrency control
        async with self._semaphore:
            try:
                # Resize image in thread pool (CPU-bound)
                if resize:
                    image_data, mime_type = await self._resize_image(image_data)
                else:
                    # Detect mime type
                    img = Image.open(BytesIO(image_data))
                    format_map = {"JPEG": "image/jpeg", "PNG": "image/png", "GIF": "image/gif", "WEBP": "image/webp"}
                    mime_type = format_map.get(img.format or "PNG", "image/png")
                
                # Encode to base64
                base64_image = base64.b64encode(image_data).decode("utf-8")
                
                # Build prompt - explicit instructions to prevent repetition
                prompt = (
                    f"Extract all text from this image. The image contains {language}. "
                    "IMPORTANT RULES:\n"
                    "1. Output each piece of text ONLY ONCE - never repeat any text.\n"
                    "2. If you see repeated text in the image, write it only once.\n"
                    "3. Do not hallucinate or generate text that is not in the image.\n"
                    "4. Preserve the original layout and structure.\n"
                    "5. Be concise and accurate."
                )
                
                # Payload formats to try
                payloads = [
                    {
                        "messages": [{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}}
                            ]
                        }],
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    },
                    {
                        "model": model or "default",
                        "messages": [{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}}
                            ]
                        }],
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    }
                ]
                
                client = await self._get_client()
                last_error = None
                response_content = None
                
                for i, payload in enumerate(payloads):
                    try:
                        response = await client.post(self.chat_endpoint, json=payload)
                        
                        if response.status_code == 500:
                            last_error = response.text
                            continue
                        
                        response.raise_for_status()
                        result = response.json()
                        response_content = result["choices"][0]["message"]["content"]
                        logger.debug(f"OCR succeeded with format {i + 1}")
                        break
                        
                    except httpx.TimeoutException:
                        last_error = "Request timeout"
                        continue
                    except httpx.HTTPError as e:
                        last_error = str(e)
                        continue
                    except (KeyError, IndexError) as e:
                        last_error = f"Invalid response: {e}"
                        continue
                
                if response_content is None:
                    raise Exception(f"All formats failed. Last error: {last_error}")
                
                # Parse response in thread pool (potentially CPU-heavy for complex parsing)
                natural_text = await asyncio.to_thread(self._parse_response_content, response_content)
                
                # Apply deduplication to remove repeated text (common OCR hallucination)
                natural_text = await asyncio.to_thread(self._deduplicate_text, natural_text)
                
                result = {
                    "raw_text": natural_text,
                    "markdown_text": natural_text
                }
                
                # Cache result
                self._set_cached(cache_key, result)
                
                return result
                
            except Exception as e:
                logger.error(f"OCR processing failed: {e}")
                raise Exception(f"OCR processing failed: {e}")
    
    async def health(self) -> bool:
        """Async health check."""
        try:
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/health", timeout=10.0)
            is_healthy = response.status_code == 200
            
            if is_healthy:
                logger.debug("OCR server health check passed")
            else:
                logger.warning(f"OCR health check failed: {response.status_code}")
            
            return is_healthy
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False