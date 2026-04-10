"""
Universal Document OCR - Process ANY document type
Cross-platform: Auto-detects LibreOffice on Windows, Linux, macOS
"""
from olmocr import OCRClient
from pathlib import Path
from typing import List, Optional
import tempfile
import shutil
import os
import subprocess
import logging
import platform
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class UniversalOCR:
    """Process any document type with OCR"""
    
    def __init__(self, base_url: str = os.getenv("MODEL_URL", 'https://ocr-ft-dgx.nimar.gov.pk')):
        self.ocr_client = OCRClient(base_url=base_url)
        self.temp_dir = Path(tempfile.mkdtemp())
        self.libreoffice_path = self._find_libreoffice()
        
        if self.libreoffice_path:
            logger.info(f"LibreOffice found at: {self.libreoffice_path}")
        else:
            logger.warning("LibreOffice not found - Office document conversion will be unavailable")
    
    def _find_libreoffice(self) -> Optional[str]:
        """
        Auto-detect LibreOffice executable on any platform
        Checks environment variable first, then common paths
        """
        # 1. Check environment variable (allows custom override)
        env_path = os.getenv("LIBREOFFICE_PATH")
        if env_path and os.path.exists(env_path):
            logger.info(f"Using LibreOffice from LIBREOFFICE_PATH: {env_path}")
            return env_path
        
        system = platform.system()
        
        # 2. Try 'soffice' or 'libreoffice' in PATH
        for cmd in ['soffice', 'libreoffice']:
            if shutil.which(cmd):
                logger.info(f"Found '{cmd}' in system PATH")
                return cmd
        
        # 3. Check platform-specific paths
        if system == "Windows":
            return self._find_libreoffice_windows()
        elif system == "Darwin":  # macOS
            return self._find_libreoffice_macos()
        else:  # Linux
            return self._find_libreoffice_linux()
        
        return None
    
    def _find_libreoffice_windows(self) -> Optional[str]:
        """Find LibreOffice on Windows"""
        possible_paths = [
            r"C:\Program Files\LibreOffice\program\soffice.exe",
            r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
            # Version-specific paths
            r"C:\Program Files\LibreOffice 7\program\soffice.exe",
            r"C:\Program Files (x86)\LibreOffice 7\program\soffice.exe",
            r"C:\Program Files\LibreOffice 24\program\soffice.exe",
            r"C:\Program Files (x86)\LibreOffice 24\program\soffice.exe",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # Try to find any LibreOffice installation
        program_files = [
            os.environ.get("ProgramFiles", r"C:\Program Files"),
            os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
        ]
        
        for pf in program_files:
            libreoffice_dir = Path(pf)
            if libreoffice_dir.exists():
                # Look for any LibreOffice folder
                for folder in libreoffice_dir.glob("LibreOffice*"):
                    soffice = folder / "program" / "soffice.exe"
                    if soffice.exists():
                        return str(soffice)
        
        return None
    
    def _find_libreoffice_macos(self) -> Optional[str]:
        """Find LibreOffice on macOS"""
        possible_paths = [
            "/Applications/LibreOffice.app/Contents/MacOS/soffice",
            os.path.expanduser("~/Applications/LibreOffice.app/Contents/MacOS/soffice"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def _find_libreoffice_linux(self) -> Optional[str]:
        """Find LibreOffice on Linux"""
        possible_paths = [
            "/usr/bin/libreoffice",
            "/usr/bin/soffice",
            "/usr/local/bin/libreoffice",
            "/usr/local/bin/soffice",
            "/opt/libreoffice/program/soffice",
            "/snap/bin/libreoffice",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def _get_libreoffice_install_instructions(self) -> str:
        """Get platform-specific installation instructions"""
        system = platform.system()
        
        if system == "Windows":
            return (
                "LibreOffice not found.\n\n"
                "Install LibreOffice for Windows:\n"
                "1. Download from: https://www.libreoffice.org/download/download-libreoffice/\n"
                "2. Run the installer\n"
                "3. Restart your application\n\n"
                "Or set custom path: set LIBREOFFICE_PATH=C:\\Path\\To\\soffice.exe"
            )
        elif system == "Darwin":
            return (
                "LibreOffice not found.\n\n"
                "Install LibreOffice for macOS:\n"
                "  brew install --cask libreoffice\n\n"
                "Or download from: https://www.libreoffice.org/download/\n\n"
                "Or set custom path: export LIBREOFFICE_PATH=/path/to/soffice"
            )
        else:
            return (
                "LibreOffice not found.\n\n"
                "Install LibreOffice for Linux:\n"
                "  Ubuntu/Debian: sudo apt-get install libreoffice\n"
                "  CentOS/RHEL: sudo yum install libreoffice\n"
                "  Fedora: sudo dnf install libreoffice\n\n"
                "Or set custom path: export LIBREOFFICE_PATH=/path/to/soffice"
            )
    
    def __del__(self):
        """Cleanup temp files"""
        if hasattr(self, 'temp_dir') and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def convert_pdf_to_images(self, pdf_path: str) -> List[str]:
        """Convert PDF to images (one per page)"""
        try:
            from pdf2image import convert_from_path
            
            logger.info(f"Converting PDF: {pdf_path}")
            images = convert_from_path(
                pdf_path,
                dpi=300,
                fmt='jpeg'
            )
            
            image_paths = []
            for i, image in enumerate(images):
                image_path = self.temp_dir / f"page_{i+1}.jpg"
                image.save(image_path, 'JPEG', quality=85)
                image_paths.append(str(image_path))
                logger.debug(f"Converted page {i+1}/{len(images)}")
            
            return image_paths
            
        except ImportError:
            system = platform.system()
            if system == "Windows":
                raise ImportError(
                    "pdf2image not installed.\n\n"
                    "Install with: pip install pdf2image\n"
                    "Also download Poppler for Windows:\n"
                    "https://github.com/oschwartz10612/poppler-windows/releases/\n"
                    "Extract and add to PATH or set POPPLER_PATH environment variable"
                )
            else:
                raise ImportError(
                    "pdf2image not installed.\n\n"
                    "pip install pdf2image\n"
                    "sudo apt-get install poppler-utils  # Linux\n"
                    "brew install poppler  # macOS"
                )
    
    def _convert_to_pdf_with_libreoffice(self, file_path: str, file_type: str) -> str:
        """Convert Office documents to PDF using LibreOffice"""
        if not self.libreoffice_path:
            raise RuntimeError(self._get_libreoffice_install_instructions())
        
        logger.info(f"Converting {file_type.upper()} to PDF using LibreOffice...")
        
        try:
            # Ensure absolute paths
            file_path = str(Path(file_path).absolute())
            output_dir = str(self.temp_dir.absolute())
            
            # Build command
            cmd = [
                self.libreoffice_path,
                '--headless',
                '--convert-to', 'pdf',
                '--outdir', output_dir,
                file_path
            ]
            
            logger.debug(f"Running: {' '.join(cmd)}")
            
            # Run conversion
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.stdout:
                logger.debug(f"LibreOffice output: {result.stdout}")
            
            # Find generated PDF
            input_filename = Path(file_path).stem
            pdf_path = self.temp_dir / f"{input_filename}.pdf"
            
            if pdf_path.exists():
                logger.info(f"PDF generated: {pdf_path}")
                return str(pdf_path)
            
            # Fallback: find any new PDF
            pdf_files = list(self.temp_dir.glob("*.pdf"))
            if pdf_files:
                # Get the most recently created one
                pdf_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                logger.info(f"Found PDF: {pdf_files[0]}")
                return str(pdf_files[0])
            
            raise FileNotFoundError(
                f"PDF not generated by LibreOffice for {file_type}\n"
                f"Check if the file is valid and not corrupted"
            )
            
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"LibreOffice conversion timed out after 120s for {file_type}.\n"
                f"The file might be too large or complex."
            )
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if e.stderr else str(e)
            raise RuntimeError(
                f"LibreOffice conversion failed for {file_type}:\n{error_msg}\n\n"
                f"The file might be corrupted or in an unsupported format."
            )
        except Exception as e:
            raise RuntimeError(f"Document conversion failed: {str(e)}")
    
    def convert_docx_to_images(self, docx_path: str) -> List[str]:
        """Convert Word document to images"""
        logger.info(f"Converting DOCX: {docx_path}")
        pdf_path = self._convert_to_pdf_with_libreoffice(docx_path, 'docx')
        return self.convert_pdf_to_images(pdf_path)
    
    def convert_pptx_to_images(self, pptx_path: str) -> List[str]:
        """Convert PowerPoint to images"""
        logger.info(f"Converting PPTX: {pptx_path}")
        pdf_path = self._convert_to_pdf_with_libreoffice(pptx_path, 'pptx')
        return self.convert_pdf_to_images(pdf_path)
    
    def convert_xlsx_to_images(self, xlsx_path: str) -> List[str]:
        """Convert Excel to images"""
        logger.info(f"Converting XLSX: {xlsx_path}")
        pdf_path = self._convert_to_pdf_with_libreoffice(xlsx_path, 'xlsx')
        return self.convert_pdf_to_images(pdf_path)
    
    def convert_rtf_to_images(self, rtf_path: str) -> List[str]:
        """Convert RTF to images"""
        logger.info(f"Converting RTF: {rtf_path}")
        pdf_path = self._convert_to_pdf_with_libreoffice(rtf_path, 'rtf')
        return self.convert_pdf_to_images(pdf_path)
    
    def convert_odt_to_images(self, odt_path: str) -> List[str]:
        """Convert OpenDocument Text to images"""
        logger.info(f"Converting ODT: {odt_path}")
        pdf_path = self._convert_to_pdf_with_libreoffice(odt_path, 'odt')
        return self.convert_pdf_to_images(pdf_path)
    
    def convert_odp_to_images(self, odp_path: str) -> List[str]:
        """Convert OpenDocument Presentation to images"""
        logger.info(f"Converting ODP: {odp_path}")
        pdf_path = self._convert_to_pdf_with_libreoffice(odp_path, 'odp')
        return self.convert_pdf_to_images(pdf_path)
    
    def convert_ods_to_images(self, ods_path: str) -> List[str]:
        """Convert OpenDocument Spreadsheet to images"""
        logger.info(f"Converting ODS: {ods_path}")
        pdf_path = self._convert_to_pdf_with_libreoffice(ods_path, 'ods')
        return self.convert_pdf_to_images(pdf_path)
    
    def process_document(
        self,
        file_path: str,
        language: str = "English and Urdu",
        output_format: str = "markdown",
        combine_pages: bool = True
    ) -> dict:
        """Process any document type with OCR"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {"success": False, "error": f"File not found: {file_path}"}
        
        logger.info(f"Processing: {file_path.name} ({file_path.suffix})")
        
        # Convert to images
        image_paths = []
        
        try:
            ext = file_path.suffix.lower()
            
            if ext == '.pdf':
                image_paths = self.convert_pdf_to_images(str(file_path))
            
            elif ext in ['.docx', '.doc']:
                image_paths = self.convert_docx_to_images(str(file_path))
            
            elif ext in ['.pptx', '.ppt']:
                image_paths = self.convert_pptx_to_images(str(file_path))
            
            elif ext in ['.xlsx', '.xls']:
                image_paths = self.convert_xlsx_to_images(str(file_path))
            
            elif ext == '.rtf':
                image_paths = self.convert_rtf_to_images(str(file_path))
            
            elif ext == '.odt':
                image_paths = self.convert_odt_to_images(str(file_path))
            
            elif ext == '.odp':
                image_paths = self.convert_odp_to_images(str(file_path))
            
            elif ext == '.ods':
                image_paths = self.convert_ods_to_images(str(file_path))
            
            elif ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff', '.tif']:
                image_paths = [str(file_path)]
            
            else:
                return {
                    "success": False,
                    "error": f"Unsupported file type: {ext}"
                }
        
        except Exception as e:
            logger.error(f"Conversion failed: {str(e)}")
            return {"success": False, "error": str(e)}
        
        # OCR processing
        logger.info(f"OCR Processing: {len(image_paths)} page(s)")
        
        page_results = []
        total_tokens = 0
        
        for i, image_path in enumerate(image_paths):
            logger.info(f"Processing page {i+1}/{len(image_paths)}...")
            
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
            
            if result["success"]:
                page_results.append({
                    "page": i + 1,
                    "text": result["text"],
                    "tokens": result.get("usage", {})
                })
                total_tokens += result.get("usage", {}).get("total_tokens", 0)
            else:
                page_results.append({
                    "page": i + 1,
                    "text": "",
                    "error": result.get("error")
                })
        
        # Combine results
        if combine_pages:
            combined_text = "\n\n---\n\n".join([
                f"# Page {r['page']}\n\n{r['text']}" 
                for r in page_results if r['text']
            ])
            
            return {
                "success": True,
                "text": combined_text,
                "pages": len(image_paths),
                "total_tokens": total_tokens,
                "page_results": page_results
            }
        else:
            return {
                "success": True,
                "pages": len(image_paths),
                "total_tokens": total_tokens,
                "page_results": page_results
            }