import logging
import requests
from fastapi import FastAPI, HTTPException, status
from fastapi import FastAPI, Header, HTTPException

from config import Config
from services import TranslationService, SummarizationService
from models import (
    TranslationRequest, TranslationResponse,
    SummarizationRequest, SummarizationResponse, HealthResponse)
from models import TokenValidationRequest, TokenValidationResponse
from utils import verify_api_key, count_tokens, truncate_to_token_limit

config = Config()
app = FastAPI(
    title=config.API_TITLE,
    description=f"{config.API_TITLE}",
    version=config.API_VERSION)



log_level = getattr(logging, Config.LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(Config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)




@app.get("/")
def root():
    return {
        "message": f"Welcome to {config.API_TITLE}",
        "docs": "/docs",
        "health": "/health",
        "config": "/config",
        "translate": "/v1/translate",
        "summarize": "/v1/summarize",
        "validate_tokens": "/v1/validate-tokens",
        'input_token_limit': config.INPUT_TOKENS_LIMIT,
        'context_window': config.CONTEXT_WINDOW
        # "model": config.MODEL_NAME,
        # "base_url": config.BASE_URL,
    }


@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Pings the GPT-OSS model list endpoint to verify health.
    """
    try:
        resp = requests.get(config.HEALTH_ENDPOINT, timeout=5)
        resp.raise_for_status()
        models = [m["id"] for m in resp.json().get("data", [])]
        return HealthResponse(
            status="healthy",
            available_models=models,
            checked_url=config.HEALTH_ENDPOINT,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {e}")


# Translation endpoint
@app.post(
    "/v1/translate",
    response_model=TranslationResponse,
    summary="Translate between English and Urdu")
def translate_text(
    request: TranslationRequest,
    authorization: str = Header(None)
):
    """
    Translate text between English and Urdu.
    
    - **Urdu → English**: Automatically detects Urdu and translates to English
    - **English → Urdu**: Automatically detects English and translates to Urdu
    
    No need to specify source or target language - it's automatic!
    """
    try:
        try:
            if not authorization:
                logger.warning("Missing Authorization header")
                raise HTTPException(
                    status_code=401,
                    detail="Missing Authorization header"
            )
        
        # Extract API key from Authorization header (expecting "Bearer <api_key>")
            auth_parts = authorization.split()
            if len(auth_parts) != 2 or auth_parts[0].lower() != "bearer":
                logger.warning("Invalid Authorization header format")
                raise HTTPException(
                    status_code=401,
                    detail="Invalid Authorization header format. Expected: Bearer <api_key>"
                )
            
            api_key = auth_parts[1]
            if not verify_api_key(api_key):
                logger.warning("Invalid API key provided")
                raise HTTPException(
                    status_code=401,
                    detail="Invalid API key"
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error validating API key: {str(e)}")
            raise HTTPException(
                status_code=401,
                detail="Invalid Authorization header"
            )

        # api_key: str = Depends(verify_api_key)
        logger.info("Translation request received")
        
        # Call translation service (auto-detects and translates)
        result = TranslationService.translate_text(text=request.text)
        
        # Determine target language based on detected language
        target_lang = "English" if result.detected_language == "Urdu" else "Urdu"
        
        # Build response
        response = TranslationResponse(
            original_text=request.text,
            translated_text=result.translated_text,
            detected_language=result.detected_language,
            target_language=target_lang
        )
        
        logger.info(f"Translation completed: {result.detected_language} → {target_lang}")
        return response
        
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Translation failed: {str(e)}"
        )


# Summarization endpoint
@app.post(
    "/v1/summarize",
    response_model=SummarizationResponse,
    summary="Summary of Text")
def summarize_text(
    request: SummarizationRequest,
    authorization: str = Header(None)):
    """
    Generate an English summary of the text.
    
    - **Input**: English or Urdu text
    - **Output**: Always English summary
    - **max_words**: Control summary length (default: 150 words)
    
    The summary is always provided in English regardless of input language.
    """
    try:
        try:
            if not authorization:
                logger.warning("Missing Authorization header")
                raise HTTPException(
                    status_code=401,
                    detail="Missing Authorization header"
            )
        
        # Extract API key from Authorization header (expecting "Bearer <api_key>")
            auth_parts = authorization.split()
            if len(auth_parts) != 2 or auth_parts[0].lower() != "bearer":
                logger.warning("Invalid Authorization header format")
                raise HTTPException(
                    status_code=401,
                    detail="Invalid Authorization header format. Expected: Bearer <api_key>"
                )
            
            api_key = auth_parts[1]
            if not verify_api_key(api_key):
                logger.warning("Invalid API key provided")
                raise HTTPException(
                    status_code=401,
                    detail="Invalid API key"
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error validating API key: {str(e)}")
            raise HTTPException(
                status_code=401,
                detail="Invalid Authorization header"
            )
        logger.info("Summarization request received")
        
        # Call summarization service
        result = SummarizationService.summarize_text(
            text=request.text,
            # max_words=request.max_words
        )
        
        # Calculate word count
        summary_word_count = len(result.summary.split())
        
        # Build response
        response = SummarizationResponse(
            original_text=request.text,
            summary=result.summary,
            # detected_language=result.detected_language,
            # word_count=summary_word_count
        )
        
        logger.info(f"Summarization completed: {result.detected_language} → English ({summary_word_count} words)")
        return response
        
    except Exception as e:
        logger.error(f"Summarization error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Summarization failed: {str(e)}"
        )



@app.post(
    "/v1/validate-tokens",
    response_model=TokenValidationResponse,
    summary="Validate token count before translation or summarization"
)
def validate_tokens(request: TokenValidationRequest, authorization: str = Header(None)):
    """
    Validate the number of tokens in input text.
    - Returns token count
    - Shows if within or over limit
    - Provides truncated preview if needed
    """
    try:
        if not authorization:
            logger.warning("Missing Authorization header")
            raise HTTPException(
                status_code=401,
                detail="Missing Authorization header"
        )
    
    # Extract API key from Authorization header (expecting "Bearer <api_key>")
        auth_parts = authorization.split()
        if len(auth_parts) != 2 or auth_parts[0].lower() != "bearer":
            logger.warning("Invalid Authorization header format")
            raise HTTPException(
                status_code=401,
                detail="Invalid Authorization header format. Expected: Bearer <api_key>"
            )
        
        api_key = auth_parts[1]
        if not verify_api_key(api_key):
            logger.warning("Invalid API key provided")
            raise HTTPException(
                status_code=401,
                detail="Invalid API key"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating API key: {str(e)}")
        raise HTTPException(
            status_code=401,
            detail="Invalid Authorization header"
        )
    
    try:
        token_count = count_tokens(request.text)
        within_limit = token_count <= config.INPUT_TOKENS_LIMIT

        truncated_preview = None
        if not within_limit:
            truncated_preview = truncate_to_token_limit(request.text, config.INPUT_TOKENS_LIMIT)[:500]  # only show first 500 chars for preview

        logger.info(f"Token validation: {token_count} tokens (limit {config.INPUT_TOKENS_LIMIT})")

        return TokenValidationResponse(
            token_count=token_count,
            INPUT_TOKENS_LIMIT=config.INPUT_TOKENS_LIMIT,
            within_limit=within_limit,
            truncated_preview=truncated_preview
        )
    except Exception as e:
        logger.error(f"Token validation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Token validation failed: {str(e)}")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)