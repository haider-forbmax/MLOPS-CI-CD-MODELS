import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype
from fastapi import HTTPException
from transformers import AutoFeatureExtractor, AutoConfig
from emotion_inference.config import Config
import librosa
import numpy as np
import logging

logger = logging.getLogger(__name__)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=-1, keepdims=True)


try:
    triton_client = httpclient.InferenceServerClient(url=Config.TRITON_URL)
    feature_extractor = AutoFeatureExtractor.from_pretrained(Config.MODEL_ID)
    config = AutoConfig.from_pretrained(Config.MODEL_ID)
    id2label = config.id2label
    logger.info("Triton client and feature extractor initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Triton client or feature extractor: {e}")
    triton_client = None
    feature_extractor = None
    id2label = None

def classify_emotion_triton(audio_file_path: str):
    """
    Classify emotion using Triton inference server.
    
    Args:
        audio_file_path: Path to the audio file
        
    Returns:
        List containing emotion prediction with label and score
    """
    logger.info("Triton inference called")
    try:
        # Load and preprocess audio
        try:
            audio, sr = librosa.load(audio_file_path, sr=16000)
        except Exception as e:
            logger.error(f"Failed to load audio file: {e}")
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Unable to load the audio file. The file may be corrupted or in an invalid format",
                    "error": f"Audio loading failed: {str(e)}"
                }
            )
        
        # Check if audio is valid
        if len(audio) == 0:
            logger.error("Audio file is empty")
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "The audio file is empty or contains no valid audio data",
                    "error": "No valid audio chunks found"
                }
            )

        # Extract features
        try:
            inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="np")
            input_features = inputs["input_features"].astype(np.float32)
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Failed to process audio features",
                    "error": f"Feature extraction error: {str(e)}"
                }
            )

        # Prepare Triton input
        try:
            infer_input = httpclient.InferInput(
                "input_features", 
                input_features.shape, 
                np_to_triton_dtype(np.float32)
            )
            infer_input.set_data_from_numpy(input_features, binary_data=True)
            
            # Request output
            output = httpclient.InferRequestedOutput("logits", binary_data=True)
        except Exception as e:
            logger.error(f"Failed to prepare Triton input: {e}")
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Failed to prepare inference request",
                    "error": f"Triton input preparation failed: {str(e)}"
                }
            )
        
        # Send inference request
        try:
            response = triton_client.infer(
                model_name=Config.MODEL_NAME, 
                inputs=[infer_input], 
                outputs=[output]
            )
        except Exception as e:
            logger.error(f"Triton inference request failed: {e}")
            raise HTTPException(
                status_code=503,
                detail={
                    "message": "Emotion recognition service is temporarily unavailable. Please try again later",
                    "error": f"Triton inference failed: {str(e)}"
                }
            )
        
        # Process results
        try:
            logits = response.as_numpy("logits")
            probabilities = _softmax(logits)
            predicted_id = probabilities.argmax(-1).item()
            predicted_emotion = id2label[predicted_id]
            confidence = probabilities[0][predicted_id].item()
            
            logger.info(f"Emotion detected: {predicted_emotion} with confidence {confidence:.2f}")
            return predicted_emotion, confidence
            
        except Exception as e:
            logger.error(f"Failed to process inference results: {e}")
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Failed to process emotion detection results",
                    "error": f"Result processing failed: {str(e)}"
                }
            )
        
    except HTTPException:
        # Re-raise HTTPExceptions as-is
        raise
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"Unexpected error in classify_emotion_triton: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "message": "An unexpected error occurred during emotion detection",
                "error": str(e)
            }
        )
