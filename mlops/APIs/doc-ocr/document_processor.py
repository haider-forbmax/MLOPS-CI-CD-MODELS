import os
import shutil
from utils.utils import make_pdf
from config import Config
from ocr_processor import process_image
from utils.utils import *

temp_paths=create_temp_folders(Config.BASE_FOLDER , Config.SUBFOLDERS)

def extract_text_and_tables_from_pdf(pdf_path):
    import pdfplumber

    with pdfplumber.open(pdf_path) as pdf:
        text = ''
        tables = []
        for page in pdf.pages:
            text += page.extract_text()
            tables += page.extract_tables()
            
    result_dict = {
        'text': text,
        'tables': {}
    }

    non_empty_tables = [table for table in tables if any(any(cell is not None and cell != '' for cell in row) for row in table)]

    for i, table in enumerate(non_empty_tables, start=1):
        table_key = f"Table{i}"
        result_dict['tables'][table_key] = table

    return result_dict

def extract_images_from_pdf_pages(pdf_path, output_folder):
    import fitz

    base_filename = os.path.splitext(os.path.basename(pdf_path))[0]
    file_folder = os.path.join(output_folder, base_filename)
    os.makedirs(file_folder, exist_ok=True)

    images_list = []

    # Open the PDF
    pdf_document = fitz.open(pdf_path)

    # Iterate through each page
    for page_number in range(pdf_document.page_count):
        page = pdf_document[page_number]

        # Get images from the page
        images = page.get_images(full=True)

        # Iterate through each image on the page
        for img_index, img_info in enumerate(images):
            image_index = img_info[0]
            base_image = pdf_document.extract_image(image_index)
            image_bytes = base_image["image"]

            # Generate a unique filename for each image
            image_filename = f"image_page_{page_number + 1}_{img_index + 1}.png"
            image_path = os.path.join(file_folder, image_filename)
            images_list.append(image_path)

            # Save the image to the specified folder
            with open(image_path, "wb") as image_file:
                image_file.write(image_bytes)

    # Close the PDF document
    pdf_document.close()

    return images_list, file_folder

def process_pdf_file(file_path, output_folder=temp_paths['temp_pdf']):
    # text and table data
    results = extract_text_and_tables_from_pdf(file_path)
    images_list,file_folder=extract_images_from_pdf_pages(file_path, output_folder)
    images_data = {}
    for filename in images_list:
        # print(filename)
        pdf_image_result = process_image(filename)
        key = os.path.splitext(os.path.basename(filename))[0]
        images_data[key] = pdf_image_result
        
    
    results['images'] = images_data
    return results

def process_document(file_path):
    file_extension = file_path.lower().split('.')[-1]
    if file_extension == 'pdf':
        results=process_pdf_file(file_path=file_path)
        return results
    
    elif file_extension in Config.IMAGES_TYPE:
        result=process_image(file_path)
        results = {'text': '', 'tables': {}, 'images': {'image1': result}}
        return results
    
    elif file_extension=="txt":
        result=read_text_file(file_path)
        results = {'text': result, 'tables': {}, 'images': {'image1': ""}}

        return results
    elif file_extension in Config.NOT_SUPPORTED_FORMAT:
        return "This format is not supported!"
        
    else:
        file_folder=os.path.splitext(os.path.basename(file_path))[0]
        file_folder=os.path.join(Config.BASE_FOLDER,"convert",file_folder)
        
        file_path=make_pdf(file_path,file_extension,file_folder)
        results=process_pdf_file(file_path=file_path)
        shutil.rmtree(file_folder, ignore_errors=True)
        return results
