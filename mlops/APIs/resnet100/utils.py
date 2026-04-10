import numpy as np
import base64, io, requests
from PIL import Image
from fastapi import HTTPException
import tritonclient.http as httpclient
from config import Config
from schemas import ImageInput

config = Config()

class ResNetDetector:
    def __init__(self):
        self.triton_url = config.TRITON_SERVER_URL
        # ResNet model expects 112x112 input (hardcoded as this is model-specific)
        self.input_size = (112, 112)

    def get_triton_client(self):
        """Get Triton client connection"""
        try:
            return httpclient.InferenceServerClient(url=self.triton_url, connection_timeout=config.TRITON_TIMEOUT)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to connect to Triton server: {str(e)}")

    def load_image(self, image_input: ImageInput) -> Image.Image:
        """Load image from base64 or URL"""
        try:
            if image_input.type == "base64":
                if not image_input.data:
                    raise ValueError("Base64 data is required")
                # Handle "data:image/png;base64,xxxx" format
                if "," in image_input.data:
                    image_input.data = image_input.data.split(",")[1]
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
        """Preprocess image for ResNet model"""
        # Resize image to model input dimensions
        resized_image = image.resize(self.input_size, Image.BILINEAR)

        # Convert to numpy array and normalize
        img_data = np.array(resized_image).astype(np.float32)
        img_data = (img_data - 127.5) / 128.0
        img_data = img_data.transpose(2, 0, 1)  # HWC to CHW
        img_data = np.expand_dims(img_data, axis=0)  # Add batch dimension

        return img_data

    def run_triton_inference(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference using Triton client"""
        try:
            triton_client = self.get_triton_client()

            inputs = []
            inputs.append(httpclient.InferInput(config.MODEL_INPUT_NAME, input_data.shape, "FP32"))
            inputs[0].set_data_from_numpy(input_data)

            outputs = []
            outputs.append(httpclient.InferRequestedOutput(config.MODEL_OUTPUT_NAME))

            # Run inference
            results = triton_client.infer(config.TRITON_MODEL_NAME, inputs=inputs, outputs=outputs)

            # Get output data
            output_data = results.as_numpy(config.MODEL_OUTPUT_NAME)
            return output_data.flatten()  # Return flattened embedding vector

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Triton inference failed: {str(e)}")