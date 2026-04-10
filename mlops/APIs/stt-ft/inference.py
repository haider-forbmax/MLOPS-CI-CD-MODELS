# inference.py
import logging
import os
import time
from pathlib import Path
# from openai import OpenAI
from fastapi import HTTPException
from config import Config
import traceback
import requests

logger = logging.getLogger(__name__)


class WhisperAPIClient:
    """Client for vLLM Whisper API using OpenAI SDK"""

    def __init__(self):
        # self.base_url = Config.MODEL_URL.rsplit('/v1/', 1)[0] + '/v1'  # Get base URL
        # self.model_name = Config.MODEL_NAME
        # self.model_name = Config.URDU_MODEL_NAME
        # self.model_name = Config.ENGLISH_MODEL_NAME
        self.response_format = Config.RESPONSE_FORMAT
        self.temperature = Config.TEMPERATURE
        self.language = Config.LANGUAGE
        self.timeout = Config.REQUEST_TIMEOUT
        
        # Initialize OpenAI client pointing to vLLM server
        # self.client = OpenAI(
        #     api_key="EMPTY",  # vLLM doesn't require API key
        #     base_url=self.base_url,
        #     timeout=self.timeout,
        #     http_client=None if Config.SSL_VERIFY else self._get_insecure_client()
        # )
        
        logger.info(f"Whisper API client initialized")
        # logger.info(f"Base URL: {self.base_url}")
        # logger.info(f"Model: {self.model_name}, Format: {self.response_format}, Language: {self.language}")

    def _get_insecure_client(self):
        """Create HTTP client that doesn't verify SSL"""
        import httpx
        return httpx.Client(verify=False)

    def health_check(self) -> dict:
        """
        Check health of all dependent model APIs
        """
        services = {
            "language_detector": f"{Config.LANGUAGE_DETECTOR_URL}/health",
            "english_model": f"{Config.ENGLISH_MODEL_URL}/health",
            "urdu_model": f"{Config.URDU_MODEL_URL}/health",
        }

        results = {}

        for service_name, url in services.items():
            try:
                response = requests.get(
                    url,
                    timeout=3,
                    verify=Config.SSL_VERIFY
                )
                results[service_name] = response.status_code < 500
            except Exception as e:
                logger.error(f"{service_name} health check failed: {e}")
                results[service_name] = False

        return results

    def transcribe(self, file_path: str, language: str) -> dict:
        """
        Send audio file to vLLM Whisper API for transcription
        
        Returns:
            dict with 'text' and optionally 'segments' if verbose_json format
        """
        try:
            if language in Config.Supported_Language:
                print(language)
                file_path = Path(file_path)
                if not file_path.exists():
                    raise FileNotFoundError(f"Audio file not found: {file_path}")
                
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                logger.info(f"Transcribing audio: {file_path.name} ({file_size_mb:.2f} MB)")

                # Transcribe using OpenAI client
                start_time = time.time()
                url = (
                    f"{Config.URDU_MODEL_URL}/v1/audio/transcriptions"
                    if language in ["ur","hi"]
                    else f"{Config.ENGLISH_MODEL_URL}/v1/audio/transcriptions"
                )

                with open(file_path, "rb") as audio_file:
                    files = {
                        "file": (file_path.name, audio_file, "audio/wav")
                    }
                    # data = {
                    #     "model": (Config.URDU_MODEL_NAME
                    # if language in ["ur","hi"]
                    # else Config.ENGLISH_MODEL_NAME),
                    #     "language": language
                    # }

                    response = requests.post(
                        url,
                        files=files
                        # data=data
                    )

                response.raise_for_status()      # 🔥 FAIL FAST
                return response.json()           # 🔥 PARSED RESULT

            else:
                raise HTTPException(status_code=400, detail= {"message":f"unsupported Language supported languages are {Config.Supported_Language}"})

        except FileNotFoundError as e:
            logger.error(str(e))
            raise HTTPException(status_code=404, detail=str(e))
        
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            traceback.print_exc()
            
            # Handle OpenAI API errors
            if hasattr(e, 'status_code'):
                raise HTTPException(
                    status_code=e.status_code,
                    detail=f"Whisper API error: {str(e)}"
                )
            
            raise HTTPException(
                status_code=500,
                detail=f"Transcription failed: {str(e)}"
            )


# Simple singleton pattern
_client = None


def get_client() -> WhisperAPIClient:
    """Get or create Whisper API client"""
    global _client
    if _client is None:
        logger.info("Creating new Whisper API client")
        _client = WhisperAPIClient()
    return _client


def transcribe_audio(file_path: str, output_file: str = None) -> dict:
    """
    Transcribe audio file using Whisper API
    
    Returns:
        dict with transcription results
    """
    try:
        # Verify file exists before attempting transcription
        if not os.path.exists(file_path):
            logger.error(f"Audio file not found: {file_path}")
            raise HTTPException(
                status_code=404,
                detail=f"Audio file not found: {file_path}"
            )
        
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            logger.error(f"Audio file is empty: {file_path}")
            raise HTTPException(
                status_code=422,
                detail="Audio file is empty"
            )
        
        logger.info(f"Audio file verified: {file_path} ({file_size:,} bytes)")
        
        # Transcribe
        client = get_client()
        with open(file_path, "rb") as f:
            files = {
                "file": (file_path, f)
            }

            response = requests.post(
                f"{Config.LANGUAGE_DETECTOR_URL}/detect",
                files=files
            )
        response.raise_for_status()
        data = response.json()
        language = data["language"]

        result = client.transcribe(file_path,language)

        # Save transcription if output file specified
        if output_file:
            text = result.get('text', '')
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(text)
            logger.info(f"Transcription saved to {output_file}")

        return result

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Error in transcribe_audio: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {str(e)}"
        )