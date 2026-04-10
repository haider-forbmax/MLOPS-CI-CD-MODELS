import logging
import requests
from pathlib import Path
from config import Config
from fastapi import HTTPException
from typing import Optional

logger = logging.getLogger(__name__)


class WhisperAPIClient:
    """Client for Whisper API transcription"""
    
    def __init__(self):
        """Initialize Whisper API client with config"""
        self.base_url = Config.MODEL_URL.rstrip('/')
        self.api_key = Config.API_KEY
        self.timeout = Config.TRITON_TIMEOUT
        self.model_name = Config.MODEL_NAME
        self.temperature = Config.TEMPERATURE
        self.response_format = Config.RESPONSE_FORMAT
        
        # Create session with headers
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}"
        })
        
        logger.info(f"Whisper API Client initialized: {self.base_url}")
    
    def health_check(self) -> bool:
        """Check if server is healthy"""
        try:
            response = self.session.get(
                f"{self.base_url}/health",
                timeout=10
            )
            is_healthy = response.status_code == 200
            logger.info(f"Health check: {'OK' if is_healthy else 'Failed'}")
            return is_healthy
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def transcribe(
        self,
        file_path: str,
        language: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> str:
        """Transcribe audio file using Whisper API
        
        Args:
            file_path: Path to audio file
            language: Optional language code (e.g., 'ur', 'en')
            temperature: Optional sampling temperature
            
        Returns:
            Transcription text
        """
        try:
            # Read audio file
            with open(file_path, "rb") as f:
                audio_data = f.read()
            
            # Get filename
            filename = Path(file_path).name
            
            # Prepare multipart form data
            files = {
                'file': (filename, audio_data, 'audio/wav')
            }
            
            # Prepare form data
            data = {
                'model': self.model_name,
                'response_format': self.response_format,
                'temperature': str(temperature if temperature is not None else self.temperature),
            }
            
            # Add language if specified
            if language:
                data['language'] = language
            
            # Make API request
            logger.debug(f"Transcribing {filename} with model {self.model_name}")
            
            response = self.session.post(
                f"{self.base_url}/v1/audio/transcriptions",
                files=files,
                data=data,
                timeout=self.timeout
            )
            
            # Check response status
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            transcription = result.get('text', '').strip()
            
            logger.debug(f"Transcription successful for {filename}: {len(transcription)} chars")
            
            return transcription
            
        except requests.exceptions.Timeout:
            logger.error(f"Request timeout for {file_path}")
            raise HTTPException(status_code=504, detail="Transcription request timeout")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response body: {e.response.text}")
                raise HTTPException(
                    status_code=e.response.status_code,
                    detail=f"API error: {e.response.text}"
                )
            raise HTTPException(status_code=500, detail=f"API request failed: {str(e)}")
            
        except FileNotFoundError:
            logger.error(f"Audio file not found: {file_path}")
            raise HTTPException(status_code=404, detail="Audio file not found")
            
        except Exception as e:
            logger.error(f"Unexpected error during transcription: {e}")
            raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


# Create global client instance
_client = None


def get_client() -> WhisperAPIClient:
    """Get or create global client instance"""
    global _client
    if _client is None:
        _client = WhisperAPIClient()
    return _client


def transcribe_audio(
    file_path: str,
    output_file: str = None,
    language: Optional[str] = None
) -> str:
    """Transcribe audio file (maintains backward compatibility)
    
    Args:
        file_path: Path to audio file
        output_file: Optional path to save transcription
        language: Optional language code
        
    Returns:
        Transcription text
    """
    try:
        client = get_client()
        transcription = client.transcribe(file_path, language=language)
        
        # Save to file if requested
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(transcription)
            logger.info(f"Transcription saved to {output_file}")
        
        return transcription
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error in transcribe_audio: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")