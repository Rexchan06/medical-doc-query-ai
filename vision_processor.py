#!/usr/bin/env python3
"""
PERSON 4: Document Processing Expert
PDF processing, image conversion, and AI vision analysis

RESPONSIBILITIES:
- PDF to image conversion for vision analysis
- Claude Vision integration for chart/graph analysis
- Medical image and chart interpretation
- Advanced document structure analysis
- Integration with Person 2's text processing

DELIVERABLES:
- vision_processor.py (this file)
- Working PDF → image → AI description pipeline
- Medical chart and graph analysis
- Visual medical content extraction
"""

import os
import base64
import io
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from pathlib import Path
import tempfile

# Image processing libraries
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

# PDF processing
try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: pdf2image not installed. PDF to image conversion not available.")
    PDF2IMAGE_AVAILABLE = False

# Import Person 1's AWS configuration
try:
    from aws_config import AWSConfigManager, AWSUtilities
except ImportError:
    print("❌ Error: aws_config.py not found. Make sure Person 1 has completed AWS setup.")
    exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalVisionProcessor:
    """
    Advanced vision processing for medical documents
    Converts PDFs to images and analyzes them with AI vision
    """
    
    def __init__(self):
        """Initialize the vision processor"""
        logger.info("👁️ Initializing Medical Vision Processor...")
        
        # Initialize AWS utilities
        try:
            from aws_config import setup_aws_environment
            self.aws_config, self.aws_utils = setup_aws_environment()
        except Exception as e:
            logger.error(f"❌ Failed to initialize AWS: {e}")
            raise
        
        # Vision processing configuration
        self.max_pages_to_analyze = 10  # Limit for demo performance
        self.image_quality = 150  # DPI for PDF conversion
        self.max_image_size = (1024, 1024)  # Resize large images
        
        # Chart detection patterns
        self.chart_indicators = [
            'chart', 'graph', 'plot', 'axis', 'legend', 'scale',
            'data', 'trend', 'measurement', 'reading', 'level',
            'glucose', 'pressure', 'heart rate', 'temperature'
        ]
        
        # Medical image types
        self.medical_image_types = [
            'x-ray', 'mri', 'ct scan', 'ultrasound', 'ecg', 'ekg',
            'lab results', 'pathology', 'radiology', 'scan'
        ]
        
        logger.info("✅ Medical Vision Processor ready!")
    
    def pdf_to_images(self, pdf_path: str, max_pages: Optional[int] = None) -> List[Image.Image]:
        """
        Convert PDF pages to PIL Images for vision analysis
        Optimized for medical document processing
        """
        if not PDF2IMAGE_AVAILABLE:
            logger.error("❌ pdf2image not available. Cannot convert PDF to images.")
            return []
        
        logger.info(f"📄 Converting PDF to images: {Path(pdf_path).name}")
        
        try:
            max_pages = max_pages or self.max_pages_to_analyze
            
            # Convert PDF to images
            images = convert_from_path(
                pdf_path,
                dpi=self.image_quality,
                first_page=1,
                last_page=max_pages,
                fmt='RGB'  # Ensure RGB format
            )
            
            # Process and optimize images
            processed_images = []
            for i, image in enumerate(images):
                processed_image = self._optimize_image_for_vision(image)
                processed_images.append(processed_image)
                
                logger.info(f"📄 Processed page {i+1}: {processed_image.size}")
            
            logger.info(f"✅ Converted {len(processed_images)} pages to images")
            return processed_images
            
        except Exception as e:
            logger.error(f"❌ PDF to image conversion failed: {e}")
            return []
    
    def _optimize_image_for_vision(self, image: Image.Image) -> Image.Image:
        """
        Optimize image for Claude Vision analysis
        Enhances quality while keeping size manageable
        """
        try:
            # Resize if too large
            if image.size[0] > self.max_image_size[0] or image.size[1] > self.max_image_size[1]:
                image = image.resize(self.max_image_size, Image.Resampling.LANCZOS)
            
            # Enhance contrast for better text/chart recognition
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)
            
            # Enhance sharpness for better text recognition
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.1)
            
            # Convert to RGB if not already
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
            
        except Exception as e:
            logger.warning(f"⚠️ Image optimization failed: {e}")
            return image
    
    def image_to_base64(self, image: Image.Image, format: str = 'PNG') -> str:
        """
        Convert PIL Image to base64 string for Claude Vision API
        """
        try:
            buffer = io.BytesIO()
            image.save(buffer, format=format, quality=95)
            img_bytes = buffer.getvalue()
            return base64.b64encode(img_bytes).decode('utf-8')
        except Exception as e:
            logger.error(f"❌ Image to base64 conversion failed: {e}")
            return ""
    
    def should_analyze_page(self, image: Image.Image) -> bool:
        """
        Quick filter to determine if page has visual content worth analyzing
        Saves API costs by skipping text-only pages
        """
        try:
            img_array = np.array(image)

            # Calculate color variance - pure text pages have low variance
            variance = np.var(img_array)

            # Skip mostly uniform pages (pure text)
            if variance < 800:
                logger.info(f"🚫 Skipping page - low visual complexity (variance: {variance:.0f})")
                return False

            logger.info(f"✅ Page has visual content - analyzing (variance: {variance:.0f})")
            return True

        except Exception as e:
            logger.warning(f"⚠️ Page analysis check failed: {e}")
            # Default to analyzing if check fails
            return True
    
    def analyze_medical_image(self, image: Image.Image, page_number: int = 1) -> Dict[str, Any]:
        """
        Simplified medical image analysis using Claude Vision
        """
        logger.info(f"🔍 Analyzing medical image (page {page_number})...")

        try:
            # Convert image to base64
            image_b64 = self.image_to_base64(image)
            if not image_b64:
                return {'error': 'Failed to convert image to base64'}

            # Create comprehensive prompt
            vision_prompt = self._create_vision_prompt(page_number)

            # Analyze with Claude Vision
            vision_response = self.aws_utils.NWsafe_bedrock_vision_call(
                image_b64,
                vision_prompt
            )

            # Simple result structure
            analysis_result = {
                'page_number': page_number,
                'vision_response': vision_response,
                'timestamp': datetime.now().isoformat()
            }

            logger.info(f"✅ Vision analysis complete for page {page_number}")
            return analysis_result

        except Exception as e:
            logger.error(f"❌ Medical image analysis failed for page {page_number}: {e}")
            return {
                'page_number': page_number,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _create_vision_prompt(self, page_number: int) -> str:
        """
        Create comprehensive medical analysis prompt
        """
        return f"""You are analyzing page {page_number} of a medical document. Provide a comprehensive analysis:

1. **Content Overview**: Describe what type of medical content is present (text, charts, images, tables, etc.)
2. **Key Medical Information**: Extract all important patient data, diagnoses, treatments, medications
3. **Quantitative Data**: Extract ALL numerical values, measurements, lab results with units
4. **Visual Elements**: If charts/graphs are present, describe data trends, patterns, and key findings
5. **Medical Images**: If medical images are present (X-rays, scans, etc.), describe findings and abnormalities
6. **Clinical Significance**: Assess medical importance and highlight any abnormal, critical, or concerning findings
7. **Action Items**: Note any recommendations, follow-ups, treatment plans, or alerts

Be thorough and specific with medical details, measurements, and clinical interpretations."""
    
    
    
    
    
    def process_document_vision(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Main function: Complete vision processing pipeline for a document
        Converts PDF to images and analyzes each page with cost-saving filtering
        """
        logger.info(f"👁️ Starting vision processing for: {Path(pdf_path).name}")

        try:
            # Convert PDF to images
            images = self.pdf_to_images(pdf_path)

            if not images:
                return [{
                    'error': 'Could not convert PDF to images for vision analysis',
                    'pdf_path': pdf_path
                }]

            # Analyze each page with cost-saving filter
            vision_results = []
            pages_analyzed = 0
            pages_skipped = 0

            for i, image in enumerate(images):
                page_number = i + 1
                logger.info(f"📄 Processing page {page_number} of {len(images)}")

                # Check if page has visual content worth analyzing
                if self.should_analyze_page(image):
                    page_analysis = self.analyze_medical_image(image, page_number)
                    pages_analyzed += 1
                else:
                    page_analysis = {
                        'page_number': page_number,
                        'skipped': True,
                        'reason': 'Low visual complexity - likely text-only page',
                        'timestamp': datetime.now().isoformat()
                    }
                    pages_skipped += 1

                vision_results.append(page_analysis)

            logger.info(f"✅ Vision processing complete: {pages_analyzed} pages analyzed, {pages_skipped} pages skipped")

            return vision_results

        except Exception as e:
            logger.error(f"❌ Document vision processing failed: {e}")
            return [{
                'error': str(e),
                'pdf_path': pdf_path
            }]
    
    
    def create_vision_description_for_vector_search(self, vision_results: List[Dict]) -> List[str]:
        """
        Create text descriptions from vision analysis for vector search integration
        """
        descriptions = []

        try:
            for result in vision_results:
                if 'error' in result:
                    continue

                page_num = result.get('page_number', 1)
                vision_response = result.get('vision_response', '')

                if vision_response:
                    description = f"[PAGE {page_num} VISUAL ANALYSIS]: {vision_response}"
                    descriptions.append(description)

            logger.info(f"✅ Created {len(descriptions)} vision descriptions for vector search")

        except Exception as e:
            logger.error(f"❌ Vision description creation failed: {e}")

        return descriptions

# def test_vision_processor():
#     """
#     Test function for simplified vision processor
#     """
#     print("🧪 Testing Medical Vision Processor...")

#     try:
#         # Initialize vision processor
#         vision_processor = MedicalVisionProcessor()

#         # Test image optimization
#         if PDF2IMAGE_AVAILABLE:
#             test_image = Image.new('RGB', (800, 600), color='white')
#             optimized = vision_processor._optimize_image_for_vision(test_image)
#             print(f"✅ Image optimization: {optimized.size}")

#         # Test base64 conversion
#         test_image = Image.new('RGB', (100, 100), color='red')
#         b64_string = vision_processor.image_to_base64(test_image)
#         print(f"✅ Base64 conversion: {len(b64_string)} characters")

#         # Test cost-saving filter
#         should_analyze = vision_processor.should_analyze_page(test_image)
#         print(f"✅ Cost-saving filter: {'Analyze' if should_analyze else 'Skip'}")

#         # Test vision prompt creation
#         prompt = vision_processor._create_vision_prompt(1)
#         print(f"✅ Vision prompt creation: {len(prompt)} characters")

#         print("🎉 Vision processor test PASSED!")
#         return True

#     except Exception as e:
#         print(f"❌ Vision processor test FAILED: {e}")
#         return False

# if __name__ == "__main__":
#     """
#     Run this file directly to test vision processing
#     """
    
#     print("🚀 Medical Vision Processor - Person 4")
#     print("=" * 50)
    
#     # Test the vision processor
#     if test_vision_processor():
#         print("\n✅ Vision processor ready!")
#         print("🔗 Integration points:")
#         print("   - Person 2: Use vision results in complete document processing")
#         print("   - Person 3: Use create_vision_description_for_vector_search() for vector storage")
#         print("   - Person 5: Display vision analysis results in Gradio interface")
#         print("\n📋 Key Functions available:")
#         print("   - process_document_vision(pdf_path)")
#         print("   - analyze_medical_image(image, page_number)")
#         print("   - create_vision_description_for_vector_search(vision_results)")
        
#         if not PDF2IMAGE_AVAILABLE:
#             print("\n⚠️ Warning: pdf2image not available")
#             print("   Install with: pip install pdf2image")
#             print("   System dependency needed (poppler)")
#     else:
#         print("\n❌ Vision processor needs troubleshooting")
#         print("🔧 Check:")
#         print("   1. pdf2image library: pip install pdf2image")
#         print("   2. Pillow library: pip install Pillow")
#         print("   3. System poppler installation")
#         print("   4. AWS configuration and Bedrock access")