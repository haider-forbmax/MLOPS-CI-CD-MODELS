import os
import re
import numpy as np
from PIL import Image
from utils.utils import initialize_ocr_reader, urdu_ocr, separate_urdu_english
from config import Config


def remove_special_characters_regex(input_string):
    pattern = r'[^a-zA-Z0-9\s]'
    return re.sub(pattern, '', input_string)

def sort_boxes(bounds):
    return sorted(bounds, key=lambda b: (-b[0][0][1], -b[0][0][0]))


def pil_to_cv2(img):
    return np.array(img)


def crop_image(img, bbox):
    (tl, tr, br, bl) = bbox
    left = max(0, min(tl[0], bl[0]))
    top = max(0, min(tl[1], tr[1]))
    right = min(img.width, max(tr[0], br[0]))
    bottom = min(img.height, max(bl[1], br[1]))
    return img.crop((left, top, right, bottom))


def process_image(input_image, reader=None):
    if not os.path.exists(input_image):
        return f"File not found: {input_image}"

    if reader is None:
        reader = initialize_ocr_reader()

    img_pil = Image.open(input_image)
    if img_pil is None:
        return f"Failed to read the image: {input_image}"

    img_np = pil_to_cv2(img_pil)
    result = reader.readtext(img_np)
    final_text = ""

    if result:
        english_text, urdu_text, result = separate_urdu_english(result)
        english_text = remove_special_characters_regex(english_text)
        if english_text:
            final_text += english_text

        if urdu_text:
            urdu_text = ""
            sorted_result = sort_boxes(result)
            directory = os.path.join(Config.BASE_FOLDER, Config.CROPS)
            print(f"Saving cropped images to: {os.path.abspath(directory)}")
            os.makedirs(directory, exist_ok=True)

            ocr_results = []
            for i, (bbox, text, confidence) in enumerate(sorted_result):
                cropped_img = crop_image(img_pil, bbox)
                crop_image_path = os.path.join(directory, f"image_{i+1}.png")
                cropped_img.save(crop_image_path)
                print(f"Cropped image saved to: {crop_image_path}")
                try:
                    ocr_result = urdu_ocr(crop_image_path)
                    if "Error" in ocr_result:
                        print(f"Error during OCR for {crop_image_path}: {ocr_result}")
                        urdu_text += f" Error OCR: {ocr_result}"
                    else:
                        ocr_result = ocr_result['ocr_result']
                        ocr_results.append(ocr_result)
                except Exception as e:
                    print(f"An error occurred during Urdu OCR processing: {e}")
                    return "Urdu OCR API not live, check it!!!!"

            urdu_text = " ".join(reversed(ocr_results))
        final_text += urdu_text

    return final_text
