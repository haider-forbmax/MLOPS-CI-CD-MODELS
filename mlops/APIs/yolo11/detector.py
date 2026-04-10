import numpy as np
import base64, io, requests, cv2, time
import ast
import json
import logging
from PIL import Image, ImageDraw, ImageFont
from fastapi import HTTPException
import tritonclient.http as httpclient
from config import Config
from typing import Dict, List, Any
from schemas import (
    ImageInput, DetectionRequest, DetectionResponse, BoundingBox, Detection,
    ImageInfo, Usage
)
config = Config()
logger = logging.getLogger(__name__)


class YOLO11Detector:
    def __init__(self):
        self.triton_url = config.TRITON_SERVER_URL
        self.input_size = tuple(config.DEFAULT_INPUT_SIZE)
        self.model_classes_cache: Dict[str, Dict[str, str]] = {}
        
    def get_triton_client(self):
        """Get Triton client connection"""
        try:
            return httpclient.InferenceServerClient(url=self.triton_url, connection_timeout=config.TRITON_TIMEOUT)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to connect to Triton server: {str(e)}")
    
    def _parameter_string_value(self, parameters: Any, key_name: str) -> str:
        """Read parameter value from dict-style or list-style Triton responses."""
        if isinstance(parameters, dict):
            param = parameters.get(key_name)
            if isinstance(param, dict):
                return param.get("string_value") or param.get("stringValue", "")
            if isinstance(param, str):
                return param
            return ""

        if isinstance(parameters, list):
            for item in parameters:
                if not isinstance(item, dict):
                    continue
                if item.get("key") != key_name:
                    continue
                value = item.get("value", {})
                if isinstance(value, dict):
                    return value.get("string_value") or value.get("stringValue", "")
                if isinstance(value, str):
                    return value
            return ""

        return ""

    def _parse_metadata_blob(self, blob: str) -> Dict[str, Any]:
        """Parse metadata blob that may be Python-literal or JSON."""
        if not blob:
            return {}

        for parser in (ast.literal_eval, json.loads):
            try:
                parsed = parser(blob)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue
        return {}

    def extract_model_classes(self, triton_client: httpclient.InferenceServerClient, model_name: str) -> Dict[str, str]:
        """Extract classes from Triton model config parameters.metadata only."""
        cache_key = model_name or config.TRITON_MODEL_NAME
        if cache_key in self.model_classes_cache:
            return self.model_classes_cache[cache_key]

        try:
            model_config = triton_client.get_model_config(model_name=cache_key)
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Failed to fetch model config for '{cache_key}': {str(e)}"
            )

        payload = model_config.get("config", model_config) if isinstance(model_config, dict) else {}
        parameters = payload.get("parameters", {})
        metadata_blob = self._parameter_string_value(parameters, "metadata")
        metadata = self._parse_metadata_blob(metadata_blob)
        names = metadata.get("names", {})

        if not isinstance(names, dict) or not names:
            raise HTTPException(
                status_code=500,
                detail=f"'names' not found in Triton model config metadata for model '{cache_key}'"
            )

        class_map = {str(k): str(v) for k, v in names.items()}
        self.model_classes_cache[cache_key] = class_map
        logger.debug(f"Loaded {len(class_map)} classes from Triton model config for '{cache_key}'")
        return class_map

    def load_image(self, image_input: ImageInput) -> Image.Image:
        """Load image from base64 or URL"""
        try:
            if image_input.type == "base64":
                if not image_input.data:
                    raise ValueError("Base64 data is required")
                image_data = base64.b64decode(image_input.data)
                
                # Check image size
                if len(image_data) > config.MAX_IMAGE_SIZE:
                    raise ValueError(f"Image size exceeds maximum allowed size of {config.MAX_IMAGE_SIZE} bytes")
                
                image = Image.open(io.BytesIO(image_data)).convert('RGB')
            elif image_input.type == "url":
                if not image_input.url:
                    raise ValueError("URL is required")
                try:
                    response = requests.get(image_input.url, timeout=config.REQUEST_TIMEOUT)
                    response.raise_for_status()
                except Exception as e:
                    raise ValueError("URL is invalid or unreachable")
                
                if len(response.content) > config.MAX_IMAGE_SIZE:
                    raise ValueError(f"Image size exceeds maximum allowed size of {config.MAX_IMAGE_SIZE} bytes")
                
                image = Image.open(io.BytesIO(response.content)).convert('RGB')
            else:
                raise ValueError("Unsupported image type. Use 'base64' or 'url'")
            
            return image
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to load image: {str(e)}")
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for YOLO11 model - letterbox resize"""
        # Get original dimensions
        orig_w, orig_h = image.size
        target_w, target_h = self.input_size
        
        # Calculate scale to fit image within target size while maintaining aspect ratio
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        
        # Resize image
        resized_image = image.resize((new_w, new_h), Image.BILINEAR)
        
        # Create a blank canvas with target size (letterbox)
        canvas = Image.new('RGB', self.input_size, (114, 114, 114))
        
        # Calculate padding to center the image
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        
        # Paste resized image onto canvas
        canvas.paste(resized_image, (pad_x, pad_y))
        
        # Convert to numpy array and normalize
        img_data = np.array(canvas).astype(np.float32)
        img_data = img_data.transpose(2, 0, 1)  # HWC to CHW
        img_data = np.expand_dims(img_data, axis=0)  # Add batch dimension
        img_data /= 255.0  # Normalize to [0, 1]
        
        return img_data
    
    def apply_nms(self, detections: np.ndarray, nms_threshold: float = 0.45) -> np.ndarray:
        """Apply Non-Maximum Suppression to remove overlapping boxes"""
        if len(detections) == 0:
            return detections
        
        # Extract coordinates and confidence scores
        boxes = []
        scores = []
        
        for detection in detections:
            x, y, w, h, confidence = detection[:5]
            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x + w / 2
            y2 = y + h / 2
            boxes.append([x1, y1, x2, y2])
            scores.append(confidence)
        
        boxes = np.array(boxes)
        scores = np.array(scores)
        
        # Apply NMS using OpenCV
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.0, nms_threshold)
        
        if len(indices) > 0:
            indices = indices.flatten()
            return detections[indices]
        else:
            return np.array([])
    
    def postprocess_detections(self, output: np.ndarray, original_size: tuple, 
                             confidence_threshold: float, nms_threshold: float,
                             model_classes: Dict[str, str]) -> np.ndarray:
        """
        Postprocess detection results. 
        Supports both Standard (Raw) and End-to-End (E2E) YOLO outputs.
        """
        # 1. Handle Batch Dimension
        if output.ndim == 3:
            output = output[0]  # [369, 8400] or [300, 6]

        num_classes = len(model_classes)
        expected_dim_standard = 4 + num_classes
        
        # --- CASE A: End-to-End Model (e.g., YOLO26, YOLOv10) ---
        # Shape is usually [300, 6] -> [x1, y1, x2, y2, score, class_id]
        if output.shape[-1] == 6:
            logger.debug(f"Detected E2E model output format: {output.shape}")
            
            # Filter by confidence
            confidences = output[:, 4]
            mask = confidences > confidence_threshold
            detections = output[mask]
            
            if len(detections) == 0:
                return np.array([])

            # Convert [x1, y1, x2, y2] to [cx, cy, w, h] for internal API consistency
            x1, y1, x2, y2 = detections[:, 0], detections[:, 1], detections[:, 2], detections[:, 3]
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w / 2
            cy = y1 + h / 2
            
            final_detections = np.column_stack([cx, cy, w, h, detections[:, 4], detections[:, 5]])
            
            # Scale to original image (E2E models usually output absolute coordinates relative to input size)
            return self._rescale_boxes(final_detections, original_size)

        # --- CASE B: Standard Model (e.g., YOLO11 Standard) ---
        # Shape is usually [369, 8400] -> [cx, cy, w, h, class0, class1...]
        else:
            logger.debug(f"Detected Standard model output format: {output.shape}")
            
            # Transpose if output is [369, 8400]
            if output.shape[0] == expected_dim_standard:
                output = output.T
            
            if output.shape[1] != expected_dim_standard:
                logger.error(f"Shape mismatch: Expected {expected_dim_standard} dims but got {output.shape[1]}")
                return np.array([])

            boxes = output[:, :4]
            class_scores = output[:, 4:]
            
            class_ids = np.argmax(class_scores, axis=1)
            confidences = np.max(class_scores, axis=1)
            
            mask = confidences > confidence_threshold
            if not np.any(mask):
                return np.array([])
            
            final_detections = np.column_stack([
                boxes[mask],
                confidences[mask],
                class_ids[mask]
            ])
            
            # Rescale and then apply NMS (Standard models require manual NMS)
            rescaled = self._rescale_boxes(final_detections, original_size)
            return self.apply_nms(rescaled, nms_threshold)

    def _rescale_boxes(self, detections: np.ndarray, original_size: tuple) -> np.ndarray:
        """Helper to scale [cx, cy, w, h] boxes back to original image size"""
        orig_w, orig_h = original_size
        target_w, target_h = self.input_size
        
        scale = min(target_w / orig_w, target_h / orig_h)
        pad_x = (target_w - int(orig_w * scale)) // 2
        pad_y = (target_h - int(orig_h * scale)) // 2
        
        # In-place scaling
        detections[:, 0] = (detections[:, 0] - pad_x) / scale  # cx
        detections[:, 1] = (detections[:, 1] - pad_y) / scale  # cy
        detections[:, 2] /= scale  # w
        detections[:, 3] /= scale  # h
        
        return detections
    
    def get_class_info(self, detection: np.ndarray, model_classes: Dict[str, str]) -> tuple:
        """Extract class ID and name from detection"""
        # For YOLO11 format: [x, y, w, h, confidence, class_id]
        if detection.shape[0] >= 6:
            class_id = int(detection[5])
        else:
            class_id = 0
        
        class_name = model_classes.get(str(class_id), f"class_{class_id}")
        return class_id, class_name
    
    def draw_bounding_boxes_labels(self, image: Image.Image, detections: np.ndarray, 
                           model_classes: Dict[str, str]) -> Image.Image:
        """Draw bounding boxes on image with object detections"""
        if len(detections) == 0:
            return image
        
        # Create a copy of the image to draw on
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)
        
        # Try to load a font, fallback to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            try:
                font = ImageFont.load_default()
            except:
                font = None
        
        colors = ["red", "blue", "green", "yellow", "purple", "orange", "cyan", "magenta",
                  "lime", "pink", "teal", "lavender", "brown", "beige", "maroon"]
        
        for i, detection in enumerate(detections):
            x, y, w, h, confidence = detection[:5]
            class_id, class_name = self.get_class_info(detection, model_classes)
            
            # Calculate bounding box coordinates
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)
            
            # Choose color based on class
            color = colors[class_id % len(colors)]
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Draw label
            text = f"{class_name}"
            
            # Calculate text position
            if font:
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            else:
                text_width, text_height = len(text) * 8, 15  # Approximate default font size
            
            # Position text above the box if there's space, otherwise below
            text_y = y1 - text_height - 5 if y1 - text_height - 5 > 0 else y2 + 5
            text_x = x1
            
            # Draw background rectangle for text
            draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height], 
                         fill=color, outline=color)
            
            # Draw text
            draw.text((text_x, text_y), text, fill="white", font=font)
        
        return annotated_image
    

    def draw_bounding_boxes(self, image: Image.Image, detections: np.ndarray, 
                           model_classes: Dict[str, str]) -> Image.Image:
        """Draw bounding boxes on image with object detections"""
        if len(detections) == 0:
            return image
        
        # Create a copy of the image to draw on
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)
        
        colors = ["red", "blue", "green", "yellow", "purple", "orange", "cyan", "magenta",
                  "lime", "pink", "teal", "lavender", "brown", "beige", "maroon"]
        
        for i, detection in enumerate(detections):
            x, y, w, h, confidence = detection[:5]
            class_id, class_name = self.get_class_info(detection, model_classes)
            
            # Calculate bounding box coordinates
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)
            
            # Choose color based on class
            color = colors[class_id % len(colors)]
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        return annotated_image
    
    def crop_object(self, image: Image.Image, detection: np.ndarray) -> str:
        """Crop detected object and return as base64"""
        x, y, w, h = detection[:4]
        x1 = max(0, int(x - w / 2))
        y1 = max(0, int(y - h / 2))
        x2 = min(image.width, int(x + w / 2))
        y2 = min(image.height, int(y + h / 2))
        
        # Check if the bounding box is valid (has area > 0)
        if x1 >= x2 or y1 >= y2 or w <= 0 or h <= 0:
            # Return None for invalid bounding boxes
            return None
            
        # Ensure minimum crop size to avoid empty images
        min_size = 1
        if (x2 - x1) < min_size or (y2 - y1) < min_size:
            return None
        
        cropped = image.crop((x1, y1, x2, y2))
        
        # Double check the cropped image has valid dimensions
        if cropped.size[0] == 0 or cropped.size[1] == 0:
            return None
            
        buffer = io.BytesIO()
        cropped.save(buffer, format='JPEG', quality=85)
        cropped_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return cropped_base64
    
    def image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL image to base64"""
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=85)
        return base64.b64encode(buffer.getvalue()).decode()
