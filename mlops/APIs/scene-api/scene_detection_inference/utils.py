from fastapi import HTTPException
import traceback
from PIL import Image
import base64, requests, io
from contextlib import contextmanager
from scene_detection_inference.schemas import *
from scene_detection_inference.florence import test_single_prediction
import logging

logger = logging.getLogger(__name__)

@contextmanager
def managed_image(image):
    """Context manager to ensure proper image cleanup"""
    try:
        yield image
    finally:
        try:
            if image and hasattr(image, 'close'):
                image.close()
                logger.debug("Image resource cleaned up successfully")
        except Exception as e:
            logger.warning(f"Error closing image: {str(e)}")

def process_image_with_florence(image):
    logger.info("Starting Florence model processing")
    try:
        # Open the image file        
        # Task prompt for Florence model
        task_prompt = '<DETAILED_CAPTION>'
        
        # Run the Florence model on the image
        results = test_single_prediction(image, task_prompt)
        
        if not results or "<DETAILED_CAPTION>" not in results:
            logger.error("Florence model returned invalid results")
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Failed to generate image caption. Please try again.",
                    "error": "Florence model returned invalid or empty results"
                }
            )
        
        caption = results["<DETAILED_CAPTION>"]
        logger.info(f"Florence processing successful, caption length: {len(caption)}")

        return caption
    except HTTPException:
        raise
    except requests.exceptions.Timeout:
        logger.error("Florence API timeout")
        raise HTTPException(
            status_code=504,
            detail={
                "message": "Image processing service timed out. Please try again later.",
                "error": "Timeout when calling Florence inference API"
            }
        )
    except requests.exceptions.RequestException as e:
        logger.error(f"Florence API request failed: {str(e)}")
        raise HTTPException(
            status_code=502,
            detail={
                "message": "Unable to connect to image processing service. Please try again later.",
                "error": f"Florence API request failed: {str(e)}"
            }
        )
    except Exception as e:
        logger.error(f"Florence processing failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "message": "An unexpected error occurred during image processing.",
                "error": f"Florence processing failed: {str(e)}"
            }
        )

def load_image(image_input:ImageData) -> Image.Image:
    '''Load image fro, base64 or URL '''
    image = None
    buffer = None
    response = None
    try:
            if image_input.type == "base64":
                if not image_input.data:
                    raise HTTPException(
                    status_code=400,
                    detail={
                        "message": "Base64 image data is required but was not provided.",
                        "error": "Missing base64 data in request"
                    }
                    )

                try:
                    image_data = base64.b64decode(image_input.data)
                    buffer = io.BytesIO(image_data)
                    image = Image.open(buffer).convert('RGB')
                except base64.binascii.Error:
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "message": "Invalid base64 image data provided.",
                            "error": "Base64 decoding failed"
                        }
                    )
                except Exception as e:
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "message": "Unable to process the provided image data.",
                            "error": f"Failed to open base64 image: {str(e)}"
                        }
                    )
            elif image_input.type == "url":
                if not image_input == "url":
                    raise HTTPException(
                            status_code=400,
                            detail={
                                "message": "Image URL is required but was not provided.",
                                "error": "Missing URL in request"
                            }
                        )
                try:
                    response = requests.get(image_input.url)
                    response.raise_for_status()
                except requests.exceptions.Timeout:
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "message": "Image download timed out. Please check the URL and try again.",
                            "error": f"Timeout when fetching image from URL: {image_input.url}"
                        }
                    )
                except requests.exceptions.HTTPError as e:
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "message": f"Unable to download image from the provided URL (HTTP {e.response.status_code}).",
                            "error": f"HTTP error when fetching image: {str(e)}"
                        }
                    )
                except requests.exceptions.RequestException as e:
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "message": "Unable to download image. Please check the URL and try again.",
                            "error": f"Request failed for URL {image_input.url}: {str(e)}"
                        }
                    )
                
                try:
                    buffer = io.BytesIO(response.content)
                    image = Image.open(buffer).convert('RGB')
                except Exception as e:
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "message": "Downloaded file is not a valid image.",
                            "error": f"Failed to open image from URL: {str(e)}"
                        }
                    )
            else:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "message": "Unsupported image type. Please use 'base64' or 'url'.",
                        "error": f"Invalid image type: {image_input.type}"
                    }
                )
            
            return image

    except HTTPException:
        # Clean up resources on error
        if image:
            try:
                image.close()
            except:
                pass
        if buffer:
            try:
                buffer.close()
            except:
                pass
        raise
        
    except Exception as e:
        # Clean up resources on unexpected error
        if image:
            try:
                image.close()
            except:
                pass
        if buffer:
            try:
                buffer.close()
            except:
                pass
        
        logger.error(f"Unexpected error loading image: {str(e)}", exc_info=True)
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail={
                "message": "An unexpected error occurred while loading the image.",
                "error": f"Unexpected image loading error: {str(e)}"
            }
        )
    
    finally:
        # Close response if it exists (for URL case)
        if response:
            try:
                response.close()
                logger.debug("Response connection closed")
            except Exception as e:
                logger.warning(f"Error closing response: {str(e)}")

