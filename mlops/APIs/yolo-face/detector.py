import numpy as np
import base64, io, requests, cv2, time
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



class YOLO11Detector:
    def __init__(self):
        self.triton_url = config.TRITON_SERVER_URL
        self.input_size = tuple(config.DEFAULT_INPUT_SIZE)
        self.classes = config.CLASSES
        
    def get_triton_client(self):
        """Get Triton client connection"""
        try:
            return httpclient.InferenceServerClient(url=self.triton_url, connection_timeout=config.TRITON_TIMEOUT)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to connect to Triton server: {str(e)}")
    
    def extract_model_classes(self, model_name: str) -> Dict[str, str]:
        """Use classes from environment configuration"""
        return config.CLASSES

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
        """Postprocess YOLO face model output to get valid detections"""
        # Face detection model output format: [1, num_detections, 16]
        # where 16 is [x, y, w, h, confidence, class_prob1, class_prob2, ...]
        if output.ndim == 3:
            detections = output[0]  # Remove batch dimension, shape: [num_detections, 16]
        else:
            detections = output

        if len(detections) == 0:
            return np.array([])

        # Filter detections based on confidence threshold (confidence is at index 4)
        valid_detections = detections[detections[:, 4] > confidence_threshold]

        if len(valid_detections) == 0:
            return np.array([])

        # Scale bounding boxes to original image size
        scale_x = original_size[0] / self.input_size[0]
        scale_y = original_size[1] / self.input_size[1]

        valid_detections[:, 0] *= scale_x  # x_center
        valid_detections[:, 1] *= scale_y  # y_center
        valid_detections[:, 2] *= scale_x  # width
        valid_detections[:, 3] *= scale_y  # height

        # For face detection, we only need [x, y, w, h, confidence]
        # Extract the first 5 columns and add class_id as 0 (face class)
        final_detections = np.column_stack([
            valid_detections[:, :5],  # [x, y, w, h, confidence]
            np.zeros(len(valid_detections))  # class_id = 0 for face
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