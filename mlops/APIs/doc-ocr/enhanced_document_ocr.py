"""
Enhanced Universal OCR - Extended Format Support
Now supports 30+ document formats with parallel processing!
"""
import os
import json
import re
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path

from document_to_text_ocr import UniversalOCR as BaseUniversalOCR


logger = logging.getLogger(__name__)


class RetryStrategy(Enum):
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIXED = "fixed"


@dataclass
class OCRConfig:
    """Configuration for OCR processing"""
    max_concurrent_requests: int = 4
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    timeout_per_page: float = 120.0
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    
    @classmethod
    def from_env(cls) -> "OCRConfig":
        """Load config from environment variables"""
        strategy_map = {
            "exponential": RetryStrategy.EXPONENTIAL,
            "linear": RetryStrategy.LINEAR,
            "fixed": RetryStrategy.FIXED,
        }
        strategy_str = os.getenv("OCR_RETRY_STRATEGY", "exponential").lower()
        
        return cls(
            max_concurrent_requests=int(os.getenv("OCR_MAX_CONCURRENT", "4")),
            max_retries=int(os.getenv("OCR_MAX_RETRIES", "3")),
            base_delay=float(os.getenv("OCR_BASE_DELAY", "1.0")),
            max_delay=float(os.getenv("OCR_MAX_DELAY", "30.0")),
            timeout_per_page=float(os.getenv("OCR_TIMEOUT_PER_PAGE", "120.0")),
            retry_strategy=strategy_map.get(strategy_str, RetryStrategy.EXPONENTIAL),
        )


class RateLimitedOCRProcessor:
    """Handles parallel OCR with rate limiting and retries"""
    
    def __init__(self, ocr_client, config: Optional[OCRConfig] = None):
        self.ocr_client = ocr_client
        self.config = config or OCRConfig.from_env()
        self._semaphore = Semaphore(self.config.max_concurrent_requests)
        self._active_requests = 0
        self._completed = 0
        self._failed = 0
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay based on retry strategy"""
        if self.config.retry_strategy == RetryStrategy.EXPONENTIAL:
            delay = self.config.base_delay * (2 ** attempt)
        elif self.config.retry_strategy == RetryStrategy.LINEAR:
            delay = self.config.base_delay * (attempt + 1)
        else:
            delay = self.config.base_delay
        
        return min(delay, self.config.max_delay)
    
    def _is_retryable_error(self, error: str) -> bool:
        """Check if error is retryable"""
        retryable_patterns = [
            "rate limit",
            "timeout",
            "connection",
            "503",
            "502",
            "429",
            "temporarily unavailable",
            "overloaded",
            "server error",
            "network",
        ]
        error_lower = error.lower()
        return any(pattern in error_lower for pattern in retryable_patterns)
    
    def _process_single_page(
        self,
        page_index: int,
        image_path: str,
        language: str,
        output_format: str,
        clean_func: Callable[[str], str]
    ) -> Dict[str, Any]:
        """Process a single page with semaphore control and retries"""
        
        page_num = page_index + 1
        last_error = None
        
        for attempt in range(self.config.max_retries + 1):
            # Acquire semaphore (blocks if too many concurrent requests)
            with self._semaphore:
                try:
                    start_time = time.time()
                    
                    if output_format == "markdown":
                        result = self.ocr_client.ocr_with_markdown(
                            image_path,
                            language=language,
                            resize=True
                        )
                    else:
                        result = self.ocr_client.ocr_with_layout(
                            image_path,
                            language=language,
                            resize=True
                        )
                    
                    elapsed = time.time() - start_time
                    
                    if result["success"]:
                        cleaned_text = clean_func(result["text"])
                        self._completed += 1
                        
                        return {
                            "page": page_num,
                            "text": cleaned_text,
                            "tokens": result.get("usage", {}),
                            "success": True,
                            "elapsed_time": elapsed,
                            "attempts": attempt + 1
                        }
                    else:
                        last_error = result.get("error", "Unknown error")
                        
                        # Check if error is retryable
                        if self._is_retryable_error(last_error) and attempt < self.config.max_retries:
                            delay = self._calculate_delay(attempt)
                            logger.warning(f"Page {page_num} failed (attempt {attempt + 1}), retrying in {delay:.1f}s...")
                            time.sleep(delay)
                            continue
                        
                except Exception as e:
                    last_error = str(e)
                    
                    if self._is_retryable_error(last_error) and attempt < self.config.max_retries:
                        delay = self._calculate_delay(attempt)
                        logger.warning(f"Page {page_num} error (attempt {attempt + 1}): {last_error}, retrying in {delay:.1f}s...")
                        time.sleep(delay)
                        continue
        
        # All retries exhausted
        self._failed += 1
        logger.error(f"Page {page_num} failed after {self.config.max_retries + 1} attempts: {last_error}")
        
        return {
            "page": page_num,
            "text": "",
            "error": last_error,
            "success": False,
            "attempts": self.config.max_retries + 1
        }
    
    def process_pages(
        self,
        image_paths: List[str],
        language: str,
        output_format: str,
        clean_func: Callable[[str], str],
        combine_pages: bool = True
    ) -> Dict[str, Any]:
        """Process multiple pages in parallel with rate limiting"""
        
        total_pages = len(image_paths)
        
        print(f"\n{'='*60}")
        print(f"OCR Processing: {total_pages} page(s)")
        print(f"Concurrency: {self.config.max_concurrent_requests} | Retries: {self.config.max_retries}")
        print(f"{'='*60}\n")
        
        logger.info(f"Starting OCR: {total_pages} pages, concurrency={self.config.max_concurrent_requests}")
        
        # Reset counters
        self._completed = 0
        self._failed = 0
        
        # Use thread pool for parallel processing
        page_results = [None] * total_pages
        
        # ThreadPoolExecutor with enough workers - semaphore controls actual concurrency
        max_workers = min(total_pages, self.config.max_concurrent_requests * 2)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(
                    self._process_single_page,
                    i, path, language, output_format, clean_func
                ): i for i, path in enumerate(image_paths)
            }
            
            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    timeout = self.config.timeout_per_page * (self.config.max_retries + 1)
                    result = future.result(timeout=timeout)
                    page_results[result["page"] - 1] = result
                    
                    status = "+" if result["success"] else "x"
                    print(f"   {status} Page {result['page']}/{total_pages} (attempts: {result['attempts']})")
                    
                except TimeoutError:
                    page_idx = futures[future]
                    page_results[page_idx] = {
                        "page": page_idx + 1,
                        "text": "",
                        "error": "Processing timeout",
                        "success": False,
                        "attempts": 1
                    }
                    self._failed += 1
                    print(f"   x Page {page_idx + 1}/{total_pages} timed out")
                    logger.error(f"Page {page_idx + 1} timed out")
                    
                except Exception as e:
                    page_idx = futures[future]
                    page_results[page_idx] = {
                        "page": page_idx + 1,
                        "text": "",
                        "error": str(e),
                        "success": False,
                        "attempts": 1
                    }
                    self._failed += 1
                    print(f"   x Page {page_idx + 1}/{total_pages} error: {e}")
                    logger.error(f"Page {page_idx + 1} error: {e}")
        
        # Calculate totals
        total_tokens = sum(
            r.get("tokens", {}).get("total_tokens", 0)
            for r in page_results if r and r.get("success")
        )
        
        print(f"\n   Completed: {self._completed}/{total_pages} | Failed: {self._failed}/{total_pages}")
        logger.info(f"OCR complete: {self._completed} success, {self._failed} failed, {total_tokens} tokens")
        
        # Build response
        if combine_pages:
            combined_text = "\n\n---\n\n".join([
                f"# Page {r['page']}\n\n{r['text']}"
                for r in page_results if r and r.get('text')
            ])
            
            return {
                "success": self._failed < total_pages,
                "text": combined_text,
                "pages": total_pages,
                "pages_successful": self._completed,
                "pages_failed": self._failed,
                "total_tokens": total_tokens,
                "page_results": page_results
            }
        else:
            return {
                "success": self._failed < total_pages,
                "pages": total_pages,
                "pages_successful": self._completed,
                "pages_failed": self._failed,
                "total_tokens": total_tokens,
                "page_results": page_results
            }


class EnhancedUniversalOCR(BaseUniversalOCR):
    """Extended OCR with support for many more document formats"""
    
    # All supported formats
    SUPPORTED_FORMATS = {
        # Images
        'images': [
            '.jpg', '.jpeg', '.png', '.gif', '.webp',
            '.bmp', '.tiff', '.tif', '.heic'
        ],
        
        # PDFs
        'pdf': ['.pdf'],
        
        # Microsoft Office
        'word': ['.docx', '.rtf', '.doc'],
        'powerpoint': ['.pptx', '.ppt'],
        'excel': ['.xlsx', '.csv', '.xls'],
        
        # OpenOffice/LibreOffice
        'openoffice': ['.odt', '.ods'],
        
        # Apple iWork (manual processing only)
        'iwork': ['.pages', '.key', '.numbers'],
        
        # eBooks
        'ebooks': ['.epub'],
        
        # Other documents
        'other': ['.djvu', '.ps', '.eps', '.html', '.htm', '.txt']
    }

    def __init__(self, ocr_config: Optional[OCRConfig] = None):
        super().__init__()
        self._ocr_processor = RateLimitedOCRProcessor(
            self.ocr_client,
            ocr_config or OCRConfig.from_env()
        )
    
    def get_all_supported_formats(self) -> List[str]:
        """Get list of all supported file extensions"""
        all_formats = []
        for category in self.SUPPORTED_FORMATS.values():
            all_formats.extend(category)
        return all_formats
    
    def _clean_page_text(self, text: str) -> str:
        """Extract actual text from OCR response, handling JSON metadata wrapper.
        
        Handles:
        - Complete valid JSON with natural_text field
        - Truncated JSON (when response is cut off)
        - JSON embedded within other text
        - Unicode/escaped characters in natural_text
        """
        if not text:
            return text
        
        stripped = text.strip()
        
        # Case 1: Text starts with the JSON metadata pattern
        if stripped.startswith('{"primary_language"'):
            # Try complete JSON parsing first
            try:
                data = json.loads(stripped)
                if isinstance(data, dict) and 'natural_text' in data:
                    natural = data.get('natural_text')
                    return natural if natural else ''
            except json.JSONDecodeError:
                pass
            
            # JSON parsing failed - likely truncated. Extract natural_text manually.
            # Find the natural_text field start
            marker = '"natural_text":'
            marker_pos = stripped.find(marker)
            
            if marker_pos != -1:
                # Get content after the marker
                content_start = marker_pos + len(marker)
                remaining = stripped[content_start:].lstrip()
                
                # Check if it's null
                if remaining.startswith('null'):
                    return ''
                
                # Should start with a quote
                if remaining.startswith('"'):
                    # Find the content - we need to handle escaped quotes
                    content = remaining[1:]  # Skip opening quote
                    
                    # Build the extracted text, handling escapes
                    result_chars = []
                    i = 0
                    while i < len(content):
                        if content[i] == '\\' and i + 1 < len(content):
                            next_char = content[i + 1]
                            if next_char == 'n':
                                result_chars.append('\n')
                                i += 2
                            elif next_char == 't':
                                result_chars.append('\t')
                                i += 2
                            elif next_char == '"':
                                result_chars.append('"')
                                i += 2
                            elif next_char == '\\':
                                result_chars.append('\\')
                                i += 2
                            elif next_char == 'r':
                                result_chars.append('\r')
                                i += 2
                            else:
                                result_chars.append(content[i])
                                i += 1
                        elif content[i] == '"':
                            # End of string (unescaped quote)
                            break
                        else:
                            result_chars.append(content[i])
                            i += 1
                    
                    return ''.join(result_chars)
        
        # Case 2: JSON embedded somewhere in the text (e.g., after page header)
        # Look for the pattern start
        json_start = text.find('{"primary_language"')
        if json_start != -1:
            before_json = text[:json_start]
            json_part = text[json_start:]
            
            # Recursively clean the JSON part
            cleaned_json = self._clean_page_text(json_part)
            
            # Combine prefix with cleaned content
            return (before_json + cleaned_json).strip()
        
        return text
    
    def _convert_office_with_libreoffice(self, file_path: str, file_type: str) -> List[str]:
        """Helper to convert any office-like document to images via PDF"""
        import subprocess
        import shutil
        from pathlib import Path

        file_path = Path(file_path)
        output_dir = self.temp_dir

        libreoffice_cmd = (
            shutil.which("soffice") or
            shutil.which("libreoffice")
        )

        if not libreoffice_cmd:
            raise RuntimeError(
                "LibreOffice not found. Install LibreOffice and ensure "
                "'soffice' is available in PATH."
            )

        subprocess.run(
            [
                libreoffice_cmd,
                "--headless",
                "--convert-to", "pdf",
                "--outdir", str(output_dir),
                str(file_path)
            ],
            check=True
        )

        pdf_path = output_dir / f"{file_path.stem}.pdf"

        if not pdf_path.exists():
            raise RuntimeError("LibreOffice PDF conversion failed")

        return self.convert_pdf_to_images(str(pdf_path))
    
    def convert_tiff_to_images(self, tiff_path: str) -> List[str]:
        """Convert TIFF (potentially multi-page) to images"""
        from PIL import Image
        
        print(f"Converting TIFF: {tiff_path}")
        img = Image.open(tiff_path)
        
        image_paths = []
        try:
            for i in range(img.n_frames):
                img.seek(i)
                image_path = self.temp_dir / f"tiff_page_{i+1}.jpg"
                img.convert('RGB').save(image_path, 'JPEG', quality=85)
                image_paths.append(str(image_path))
                print(f"  Converted page {i+1}")
        except EOFError:
            image_path = self.temp_dir / "tiff_page_1.jpg"
            img.convert('RGB').save(image_path, 'JPEG', quality=85)
            image_paths.append(str(image_path))
        
        return image_paths
    
    def convert_heic_to_image(self, heic_path: str) -> List[str]:
        """Convert HEIC (iPhone photos) to JPEG"""
        try:
            from pillow_heif import register_heif_opener
            from PIL import Image
            
            register_heif_opener()
            
            print(f"Converting HEIC: {heic_path}")
            img = Image.open(heic_path)
            
            image_path = self.temp_dir / "heic_converted.jpg"
            img.convert('RGB').save(image_path, 'JPEG', quality=85)
            
            return [str(image_path)]
            
        except ImportError:
            raise ImportError(
                "pillow-heif not installed. Install with:\n"
                "  pip install pillow-heif"
            )
    
    def convert_bmp_to_image(self, bmp_path: str) -> List[str]:
        """Convert BMP to JPEG"""
        from PIL import Image
        
        print(f"Converting BMP: {bmp_path}")
        img = Image.open(bmp_path)
        
        image_path = self.temp_dir / "bmp_converted.jpg"
        img.convert('RGB').save(image_path, 'JPEG', quality=85)
        
        return [str(image_path)]
    
    def convert_rtf_to_images(self, rtf_path: str) -> List[str]:
        """Convert RTF to images via LibreOffice"""
        return self._convert_office_with_libreoffice(rtf_path, 'rtf')
    
    def convert_odt_to_images(self, odt_path: str) -> List[str]:
        """Convert OpenDocument Text to images"""
        print(f"Converting ODT: {odt_path}")
        return self._convert_office_with_libreoffice(odt_path, 'odt')
    
    def convert_odp_to_images(self, odp_path: str) -> List[str]:
        """Convert OpenDocument Presentation to images"""
        print(f"Converting ODP: {odp_path}")
        return self._convert_office_with_libreoffice(odp_path, 'odp')
    
    def convert_ods_to_images(self, ods_path: str) -> List[str]:
        """Convert OpenDocument Spreadsheet to images"""
        print(f"Converting ODS: {ods_path}")
        return self._convert_office_with_libreoffice(ods_path, 'ods')
    
    def convert_csv_to_image(self, csv_path: str) -> List[str]:
        """Convert CSV to image"""
        import pandas as pd
        
        print(f"Converting CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        
        img = self._dataframe_to_image(df)
        image_path = self.temp_dir / "csv_image.jpg"
        img.save(image_path, 'JPEG', quality=95)
        
        return [str(image_path)]
    
    def convert_epub_to_images(self, epub_path: str) -> List[str]:
        """Convert EPUB to images"""
        try:
            import ebooklib
            from ebooklib import epub
            from bs4 import BeautifulSoup
            from PIL import Image, ImageDraw, ImageFont
            
            print(f"Converting EPUB: {epub_path}")
            book = epub.read_epub(epub_path)
            
            image_paths = []
            page_num = 1
            
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    text = soup.get_text()
                    
                    if text.strip():
                        img = self._text_to_image(text, f"Page {page_num}")
                        image_path = self.temp_dir / f"epub_page_{page_num}.jpg"
                        img.save(image_path, 'JPEG', quality=95)
                        image_paths.append(str(image_path))
                        page_num += 1
                        
                        if page_num > 50:
                            print("  Limiting to first 50 pages...")
                            break
            
            return image_paths
            
        except ImportError:
            raise ImportError(
                "ebooklib and beautifulsoup4 not installed. Install with:\n"
                "  pip install ebooklib beautifulsoup4"
            )
    
    def convert_html_to_images(self, html_path: str) -> List[str]:
        """Convert HTML to images"""
        try:
            import subprocess
            
            print(f"Converting HTML: {html_path}")
            image_path = self.temp_dir / "html_converted.jpg"
            
            subprocess.run([
                'wkhtmltoimage',
                '--quality', '85',
                html_path,
                str(image_path)
            ], check=True, capture_output=True)
            
            return [str(image_path)]
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            from bs4 import BeautifulSoup
            
            with open(html_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
            
            text = soup.get_text()
            img = self._text_to_image(text, Path(html_path).name)
            image_path = self.temp_dir / "html_text.jpg"
            img.save(image_path, 'JPEG', quality=95)
            
            return [str(image_path)]
    
    def convert_txt_to_image(self, txt_path: str) -> List[str]:
        """Convert plain text file to image"""
        print(f"Converting TXT: {txt_path}")
        
        with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        
        img = self._text_to_image(text, Path(txt_path).name)
        image_path = self.temp_dir / "txt_image.jpg"
        img.save(image_path, 'JPEG', quality=95)
        
        return [str(image_path)]
    
    def _text_to_image(self, text: str, title: str = ""):
        """Convert text to image"""
        from PIL import Image, ImageDraw, ImageFont
        
        line_height = 25
        max_chars_per_line = 100
        
        lines = []
        for paragraph in text.split('\n'):
            if not paragraph.strip():
                lines.append('')
                continue
            while len(paragraph) > max_chars_per_line:
                lines.append(paragraph[:max_chars_per_line])
                paragraph = paragraph[max_chars_per_line:]
            lines.append(paragraph)
        
        lines = lines[:200]
        
        width = 1200
        height = max(800, len(lines) * line_height + 100)
        
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        except:
            font = ImageFont.load_default()
            title_font = font
        
        y = 20
        if title:
            draw.text((20, y), title, fill='black', font=title_font)
            y += 50
        
        for line in lines:
            draw.text((20, y), line, fill='black', font=font)
            y += line_height
        
        return img
    
    def _dataframe_to_image(self, df):
        from PIL import Image, ImageDraw, ImageFont

        df = df.head(100)

        cell_padding = 10
        font_size = 16

        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()

        rows, cols = df.shape

        col_widths = []
        for col in df.columns:
            max_len = max(
                [len(str(col))] +
                [len(str(v)) for v in df[col].astype(str).values]
            )
            col_widths.append(max_len * 9)

        img_width = sum(col_widths) + cell_padding * cols
        img_height = (rows + 1) * (font_size + cell_padding * 2)

        img = Image.new("RGB", (img_width, img_height), "white")
        draw = ImageDraw.Draw(img)

        y = 10
        x = 10

        for i, col in enumerate(df.columns):
            draw.text((x, y), str(col), fill="black", font=font)
            x += col_widths[i] + cell_padding

        y += font_size + cell_padding * 2

        for _, row in df.iterrows():
            x = 10
            for i, value in enumerate(row):
                draw.text((x, y), str(value), fill="black", font=font)
                x += col_widths[i] + cell_padding
            y += font_size + cell_padding * 2

        return img
    
    def process_document(
        self,
        file_path: str,
        language: str = "English and Urdu",
        output_format: str = "markdown",
        combine_pages: bool = True
    ) -> dict:
        """
        Enhanced process_document with support for 30+ formats
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {"success": False, "error": f"File not found: {file_path}"}
        
        print(f"\n{'='*60}")
        print(f"Processing: {file_path.name}")
        print(f"Type: {file_path.suffix}")
        print(f"{'='*60}\n")
        
        image_paths = []
        ext = file_path.suffix.lower()
        
        try:
            if ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
                image_paths = [str(file_path)]
            
            elif ext in ['.tiff', '.tif']:
                image_paths = self.convert_tiff_to_images(str(file_path))
            
            elif ext == '.heic':
                image_paths = self.convert_heic_to_image(str(file_path))
            
            elif ext == '.bmp':
                image_paths = self.convert_bmp_to_image(str(file_path))
            
            elif ext == '.pdf':
                image_paths = self.convert_pdf_to_images(str(file_path))
            
            elif ext in ['.docx', '.doc']:
                image_paths = self.convert_docx_to_images(str(file_path))
            
            elif ext == '.rtf':
                image_paths = self.convert_rtf_to_images(str(file_path))
            
            elif ext in ['.pptx', '.ppt']:
                image_paths = self.convert_pptx_to_images(str(file_path))
            
            elif ext in ['.xlsx', '.xls']:
                image_paths = self.convert_xlsx_to_images(str(file_path))
            
            elif ext == '.csv':
                image_paths = self.convert_csv_to_image(str(file_path))
            
            elif ext == '.odt':
                image_paths = self.convert_odt_to_images(str(file_path))
            
            elif ext == '.odp':
                image_paths = self.convert_odp_to_images(str(file_path))
            
            elif ext == '.ods':
                image_paths = self.convert_ods_to_images(str(file_path))
            
            elif ext == '.epub':
                image_paths = self.convert_epub_to_images(str(file_path))
            
            elif ext in ['.html', '.htm']:
                image_paths = self.convert_html_to_images(str(file_path))
            
            elif ext == '.txt':
                image_paths = self.convert_txt_to_image(str(file_path))
            
            elif ext in ['.djvu', '.ps', '.eps', '.pages', '.key', '.numbers']:
                print(f"Converting {ext.upper()} via LibreOffice fallback...")
                image_paths = self._convert_office_with_libreoffice(str(file_path), ext[1:])
            
            else:
                all_supported = self.get_all_supported_formats()
                return {
                    "success": False,
                    "error": f"Unsupported file type: {ext}\nSupported: {', '.join(all_supported)}"
                }
        
        except Exception as e:
            return {"success": False, "error": f"Conversion failed: {str(e)}"}
        
        return self._ocr_images(image_paths, language, output_format, combine_pages)
    
    def _ocr_images(self, image_paths, language, output_format, combine_pages):
        """Helper to OCR images with parallel processing"""
        return self._ocr_processor.process_pages(
            image_paths=image_paths,
            language=language,
            output_format=output_format,
            clean_func=self._clean_page_text,
            combine_pages=combine_pages
        )