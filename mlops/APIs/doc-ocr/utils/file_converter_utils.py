import os
import subprocess
import shutil
from config import Config


def doc_to_pdf(input_file, output_dir=os.path.join('temp', 'convert')):
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_pdf_file = os.path.join(output_dir, os.path.splitext(os.path.basename(input_file))[0] + '.pdf')

        if input_file.lower().endswith('.docx'):
            subprocess.run(['pandoc', '--from=docx', input_file, '-o', output_pdf_file])
        elif input_file.lower().endswith('.doc'):
            subprocess.run(['libreoffice', '--headless', '--convert-to', 'pdf', '--outdir', output_dir, input_file])
        else:
            return "Error: Unsupported file format."

        return output_pdf_file
    except Exception as e:
        return f"Error converting document to PDF: {str(e)}"


def _get_libreoffice_binary():
    return shutil.which("libreoffice") or shutil.which("soffice")





# Testnet

import json
from typing import Any, Dict, List


def value_to_blocks(key: str, value: Any) -> List[Dict]:
    """Convert a JSON key-value pair to Pandoc block(s)."""
    blocks = []

    if isinstance(value, dict):
        # Create a header line
        blocks.append({"t": "Para", "c": [{"t": "Str", "c": f"{key}:"}]})
        # Recursively add nested key-value pairs
        for subkey, subvalue in value.items():
            sub_blocks = value_to_blocks(subkey, subvalue)
            blocks.extend(sub_blocks)

    elif isinstance(value, list):
        blocks.append({"t": "Para", "c": [{"t": "Str", "c": f"{key}:"}]})
        items = []
        for item in value:
            # Handle nested list/dict items recursively
            if isinstance(item, (dict, list)):
                item_blocks = value_to_blocks("", item)
                items.append(item_blocks)
            else:
                items.append([{"t": "Plain", "c": [{"t": "Str", "c": str(item)}]}])
        blocks.append({"t": "BulletList", "c": [items]})

    else:
        para = [
            {"t": "Str", "c": f"{key}:"} if key else None,
            {"t": "Space"} if key else None,
            {"t": "Str", "c": str(value)}
        ]
        # Filter out None values (for list items without keys)
        para = [p for p in para if p]
        blocks.append({"t": "Para", "c": para})

    return blocks


def json_to_pandoc_ast(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a normal JSON dict to a Pandoc JSON AST."""
    blocks = []
    for key, value in data.items():
        blocks.extend(value_to_blocks(key, value))

    pandoc_json = {
        "pandoc-api-version": [1, 23, 1],
        "meta": {},
        "blocks": blocks
    }
    return pandoc_json


def convert_json_file_to_pandoc(input_path: str, output_path: str):
    """Convert a JSON file to Pandoc-compatible JSON."""
    with open(input_path, "r", encoding="utf-8") as infile:
        data = json.load(infile)

    pandoc_ast = json_to_pandoc_ast(data)

    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(pandoc_ast, outfile, indent=2, ensure_ascii=False)

    print(f"✅ Pandoc JSON created at: {output_path}")












def convert_to_pdf(input_file, output_dir=os.path.join('temp', 'convert'), config=Config):
    try:
        libreoffice_bin = _get_libreoffice_binary()
        if not libreoffice_bin:
            raise RuntimeError("LibreOffice is required for document conversion.")

        if not os.path.exists(input_file):
            raise RuntimeError(f"Input file not found: {input_file}")
        
        os.makedirs(output_dir, exist_ok=True)
        output_pdf_file = os.path.join(output_dir, os.path.splitext(os.path.basename(input_file))[0] + '.pdf')

        subprocess.run(
            [
                libreoffice_bin,
                '--headless',
                '--convert-to', 'pdf',
                '--outdir', output_dir,
                input_file
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        
        if not os.path.exists(output_pdf_file):
            raise RuntimeError("PDF conversion failed - output file not created")
        
        return output_pdf_file

    except Exception as e:
        raise RuntimeError(f"Error converting document to PDF: {e}")


def excel_to_pdf(input_file, output_dir=os.path.join('temp', 'convert')):
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_pdf_file = os.path.join(output_dir, os.path.splitext(os.path.basename(input_file))[0] + '.pdf')
        subprocess.run(['libreoffice', '--headless', '--convert-to', 'pdf', '--outdir', output_dir, input_file])
        return output_pdf_file
    except Exception as e:
        return f"Error converting Excel document to PDF: {str(e)}"


def ppt_to_pdf(input_file, output_dir=os.path.join('temp', 'convert')):
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_pdf_file = os.path.join(output_dir, os.path.splitext(os.path.basename(input_file))[0] + '.pdf')
        subprocess.run(['libreoffice', '--headless', '--convert-to', 'pdf', '--outdir', output_dir, input_file])
        return output_pdf_file
    except Exception as e:
        return f"Error converting PowerPoint document to PDF: {str(e)}"


def libreoffice_convert_to_pdf(input_file, output_dir=os.path.join('temp', 'convert')):
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_pdf_file = os.path.join(output_dir, os.path.splitext(os.path.basename(input_file))[0] + '.pdf')
        subprocess.run(['libreoffice', '--headless', '--convert-to', 'pdf', '--outdir', output_dir, input_file])
        return output_pdf_file
    except Exception as e:
        return "Error converting document to PDF"
