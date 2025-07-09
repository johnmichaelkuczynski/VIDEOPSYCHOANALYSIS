#!/usr/bin/env python3
"""
Robust document parser for PDF, DOC, and DOCX files
Handles both text-based and image-based documents with OCR fallback
"""

import os
import sys
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import docx
import json
from pathlib import Path

def extract_text(file_path):
    """Universal text extraction for supported document types"""
    try:
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdf':
            return extract_text_from_pdf(file_path)
        elif ext in ['.doc', '.docx']:
            return extract_text_from_doc(file_path)
        elif ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            clean_text = clean_extracted_text(text)
            return {
                "text": clean_text,
                "length": len(clean_text),
                "method": "direct text read"
            }
        else:
            return {"error": f"Unsupported file type: {ext}. Only PDF, DOC, DOCX, TXT supported."}
    except Exception as e:
        return {"error": f"Failed to extract text: {str(e)}"}

def extract_text_from_pdf(file_path):
    """Extract text from PDF with OCR fallback for image-based PDFs"""
    try:
        doc = fitz.open(file_path)
        full_text = ""
        pages_processed = 0
        ocr_pages = 0
        
        for page_num, page in enumerate(doc):
            # Try to extract text directly first
            text = page.get_text()
            if text.strip():
                full_text += text + "\n"
            else:
                # No text found, try OCR
                try:
                    pix = page.get_pixmap()
                    img_data = pix.tobytes("png")
                    img = Image.open(io.BytesIO(img_data))
                    ocr_text = pytesseract.image_to_string(img)
                    if ocr_text.strip():
                        full_text += ocr_text + "\n"
                        ocr_pages += 1
                except Exception as ocr_error:
                    print(f"OCR failed for page {page_num}: {ocr_error}", file=sys.stderr)
            
            pages_processed += 1
        
        doc.close()
        
        # Clean the text
        clean_text = clean_extracted_text(full_text)
        
        return {
            "text": clean_text,
            "pages_processed": pages_processed,
            "ocr_pages": ocr_pages,
            "length": len(clean_text),
            "method": "PyMuPDF + OCR"
        }
        
    except Exception as e:
        return {"error": f"PDF extraction failed: {str(e)}"}

def extract_text_from_doc(file_path):
    """Extract text from DOC/DOCX files with fallback methods"""
    try:
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.docx':
            # Try python-docx first (most reliable for DOCX)
            try:
                doc = docx.Document(file_path)
                paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
                text = '\n'.join(paragraphs)
                
                if text.strip():
                    clean_text = clean_extracted_text(text)
                    return {
                        "text": clean_text,
                        "paragraphs": len(paragraphs),
                        "length": len(clean_text),
                        "method": "python-docx"
                    }
            except Exception as docx_error:
                print(f"python-docx failed: {docx_error}", file=sys.stderr)
        
        # Fallback to textract for both DOC and DOCX
        try:
            import textract
            raw_text = textract.process(file_path).decode('utf-8')
            clean_text = clean_extracted_text(raw_text)
            
            return {
                "text": clean_text,
                "length": len(clean_text),
                "method": "textract"
            }
        except Exception as textract_error:
            return {"error": f"Textract extraction failed: {str(textract_error)}"}
            
    except Exception as e:
        return {"error": f"Document extraction failed: {str(e)}"}

def clean_extracted_text(text):
    """Clean and normalize extracted text"""
    if not text:
        return ""
    
    # Remove non-printable characters except newlines and tabs
    clean_text = ''.join(c for c in text if c.isprintable() or c in '\n\t ')
    
    # Normalize whitespace
    lines = clean_text.split('\n')
    clean_lines = [line.strip() for line in lines]
    
    # Remove excessive blank lines
    result_lines = []
    prev_blank = False
    for line in clean_lines:
        if line:
            result_lines.append(line)
            prev_blank = False
        elif not prev_blank:
            result_lines.append('')
            prev_blank = True
    
    return '\n'.join(result_lines).strip()

def main():
    """Command line interface for document parsing"""
    if len(sys.argv) != 2:
        print("Usage: python document_parser.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(json.dumps({"error": f"File not found: {file_path}"}))
        sys.exit(1)
    
    result = extract_text(file_path)
    print(json.dumps(result))

if __name__ == "__main__":
    main()