import numpy as np
import base64, io, requests, cv2, time
import ast
import json
import logging
from PIL import Image, ImageDraw, ImageFont
from fastapi import HTTPException
import tritonclient.http as httpclient
from config import Config
from typing import Dict, List, Any, Tuple
from schemas import (
    ImageInput, DetectionRequest, DetectionResponse, BoundingBox, Detection,
    ImageInfo, Usage
)
config = Config()
logger = logging.getLogger(__name__)



class YOLO5Detector:
    def __init__(self):
        self.triton_url = config.TRITON_SERVER_URL
        self.input_size = tuple(config.DEFAULT_INPUT_SIZE)
        self.class_cache_ttl_seconds = 600
        self.model_classes_cache: Dict[str, Tuple[float, Dict[str, str]]] = {}
        
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
        """Parse metadata blob that may be Python-literal or strict JSON."""
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
        """
        Fetch class mapping from Triton model config metadata.
        Cache result for 10 minutes per model.
        """
        cache_key = model_name or config.TRITON_MODEL_NAME
        now = time.time()

        if cache_key in self.model_classes_cache:
            ts, cached_classes = self.model_classes_cache[cache_key]
            if (now - ts) < self.class_cache_ttl_seconds:
                return cached_classes

        try:
            model_config = triton_client.get_model_config(model_name=cache_key)
        except Exception as e:
            if cache_key in self.model_classes_cache:
                logger.warning(f"Using stale cached classes for '{cache_key}' due to fetch error: {str(e)}")
                return self.model_classes_cache[cache_key][1]
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
            if cache_key in self.model_classes_cache:
                logger.warning(f"Using stale cached classes for '{cache_key}' because metadata.names is missing")
                return self.model_classes_cache[cache_key][1]
            raise HTTPException(
                status_code=500,
                detail=f"'names' not found in Triton model config metadata for model '{cache_key}'"
            )

        class_map = {str(k): str(v) for k, v in names.items()}
        self.model_classes_cache[cache_key] = (now, class_map)
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
        """Preprocess image for YOLO11 model"""
        # Resize image
        resized_image = image.resize(self.input_size, Image.BILINEAR)
        
        # Convert to numpy array and normalize
        img_data = np.array(resized_image).astype(np.float32)
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
        """Postprocess YOLO5 model output to get valid detections"""
        # YOLO5 output format: typically [num_detections, 85] or [num_detections, num_classes+5]
        # Format: [x_center, y_center, width, height, confidence, class_scores...]
        
        if output.ndim == 3:
            output = output[0]  # Remove batch dimension
        
        # Handle different orientations
        if output.shape[1] < output.shape[0] and output.shape[1] >= 14:  # 5 + 9 classes
            detections = output  # Already correct format [num_detections, 14]
        elif output.shape[0] >= 14 and output.shape[1] < output.shape[0]:
            detections = output.T  # Transpose to [num_detections, 14]
        else:
            # Handle case where we might have different format
            detections = output if output.shape[1] >= 14 else output.T
        
        if len(detections) == 0:
            return np.array([])
        
        # Extract boxes and class scores
        boxes = detections[:, :4]  # [x_center, y_center, width, height]
        obj_confidence = detections[:, 4]  # objectness confidence
        
        # Handle class scores - for 9 classes, we expect columns 5-13
        if detections.shape[1] >= 14:  # 5 + 9 classes
            class_scores = detections[:, 5:14]  # 9 class scores
        else:
            # Fallback: assume remaining columns are class scores
            class_scores = detections[:, 5:]
        
        # Get class predictions
        class_ids = np.argmax(class_scores, axis=1)
        class_confidences = np.max(class_scores, axis=1)
        
        # Combine objectness and class confidence
        final_confidences = obj_confidence * class_confidences
        
        # Filter by confidence threshold
        valid_mask = final_confidences > confidence_threshold
        if not np.any(valid_mask):
            return np.array([])
        
        valid_boxes = boxes[valid_mask]
        valid_confidences = final_confidences[valid_mask]
        valid_class_ids = class_ids[valid_mask]
        
        # Scale bounding boxes to original image size
        scale_x = original_size[0] / self.input_size[0]
        scale_y = original_size[1] / self.input_size[1]
        
        valid_boxes[:, 0] *= scale_x  # x_center
        valid_boxes[:, 1] *= scale_y  # y_center
        valid_boxes[:, 2] *= scale_x  # width
        valid_boxes[:, 3] *= scale_y  # height
        
        # Combine results into detection format [x, y, w, h, confidence, class_id]
        final_detections = np.column_stack([
            valid_boxes,
            valid_confidences,
            valid_class_ids
        ])
        
        # Apply NMS
        final_detections = self.apply_nms(final_detections, nms_threshold)
        
        return final_detections
    
    def get_class_info(self, detection: np.ndarray, model_classes: Dict[str, str]) -> tuple:
        """Extract class ID and name from detection"""
        # For YOLO11 format: [x, y, w, h, confidence, class_id]
        if detection.shape[0] >= 6:
            class_id = int(detection[5])
        else:
            class_id = 0
        class_id = max(0, min(class_id, 8))
        class_name = model_classes.get(str(class_id), f"unknown_class_{class_id}")
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
                  "lime", "pink", "teal", "lavender", "brown", "beige", "maroon", "mint"]
        
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
            
            # Draw label with confidence score
            # text = f"{class_name}: {confidence:.2f}"
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
                  "lime", "pink", "teal", "lavender", "brown", "beige", "maroon", "mint"]
        
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
