#!/usr/bin/env python3
"""
Debug script to test PDF processing step by step
"""

import os
from pathlib import Path
import base64
import io
from dotenv import load_dotenv
load_dotenv()

def test_pdf_processing(pdf_path):
    """Test PDF processing step by step"""

    print(f"Testing PDF processing for: {pdf_path}")

    # Step 1: Check if file exists
    if not os.path.exists(pdf_path):
        print(f"ERROR: File not found: {pdf_path}")
        return False

    print(f"SUCCESS: File exists: {Path(pdf_path).name}")

    # Step 2: Test pdf2image
    try:
        from pdf2image import convert_from_path
        print("SUCCESS: pdf2image imported successfully")
    except ImportError as e:
        print(f"ERROR: pdf2image import failed: {e}")
        return False

    # Step 3: Try converting PDF to images
    try:
        print("Converting PDF to images...")
        images = convert_from_path(pdf_path, dpi=150, fmt='RGB')
        print(f"SUCCESS: Converted to {len(images)} images")

        if not images:
            print("ERROR: No images generated from PDF")
            return False

    except Exception as e:
        print(f"ERROR: PDF conversion failed: {e}")
        return False

    # Step 4: Test image processing
    try:
        first_image = images[0]
        print(f"SUCCESS: First image size: {first_image.size}")

        # Convert to base64
        buffer = io.BytesIO()
        first_image.save(buffer, format='PNG', quality=95)
        img_bytes = buffer.getvalue()
        image_b64 = base64.b64encode(img_bytes).decode('utf-8')
        print(f"SUCCESS: Base64 conversion successful: {len(image_b64)} characters")

    except Exception as e:
        print(f"ERROR: Image processing failed: {e}")
        return False

    # Step 5: Test Nova Lite API call
    try:
        from aws_config import setup_aws_environment
        aws_config, aws_utils = setup_aws_environment()

        print("Testing Nova Lite vision call...")
        prompt = "Analyze this page and extract all text content. What do you see?"

        response = aws_utils.safe_bedrock_vision_call(image_b64, prompt)

        if response.startswith("Error:"):
            print(f"ERROR: Nova Lite vision call failed: {response}")
            return False
        else:
            print(f"SUCCESS: Nova Lite response: {response[:200]}...")
            print(f"Response length: {len(response)} characters")

    except Exception as e:
        print(f"ERROR: Nova Lite test failed: {e}")
        return False

    print("All tests passed!")
    return True

if __name__ == "__main__":
    # Test with a PDF file
    # Replace with actual PDF path you're testing
    pdf_path = "test.pdf"
    test_pdf_processing(pdf_path)