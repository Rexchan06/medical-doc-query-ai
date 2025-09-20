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
    
    def detect_visual_content_type(self, image: Image.Image) -> Dict[str, Any]:
        """
        Quick analysis to determine what type of visual content the image contains
        Helps optimize the vision analysis prompts
        """
        try:
            # Convert to numpy array for basic analysis
            img_array = np.array(image)
            
            # Basic image statistics
            brightness = np.mean(img_array)
            contrast = np.std(img_array)
            
            # Color analysis
            is_grayscale = len(img_array.shape) == 2 or (
                len(img_array.shape) == 3 and 
                np.allclose(img_array[:,:,0], img_array[:,:,1]) and 
                np.allclose(img_array[:,:,1], img_array[:,:,2])
            )
            
            # Text density estimation (simplified)
            # High contrast usually indicates text or charts
            likely_text_heavy = contrast > 50
            
            # Image classification hints
            content_hints = {
                'brightness': float(brightness),
                'contrast': float(contrast),
                'is_grayscale': is_grayscale,
                'likely_contains_text': likely_text_heavy,
                'image_size': image.size,
                'suggested_analysis': self._suggest_analysis_type(brightness, contrast, likely_text_heavy)
            }
            
            return content_hints
            
        except Exception as e:
            logger.warning(f"⚠️ Visual content detection failed: {e}")
            return {'suggested_analysis': 'general_medical'}
    
    def _suggest_analysis_type(self, brightness: float, contrast: float, 
                             text_heavy: bool) -> str:
        """
        Suggest the type of analysis based on image characteristics
        """
        if text_heavy and contrast > 60:
            return 'text_and_charts'
        elif brightness < 100:  # Dark images might be medical scans
            return 'medical_imaging'
        elif contrast > 40:
            return 'charts_and_graphs'
        else:
            return 'general_medical'
    
    def analyze_medical_image(self, image: Image.Image, page_number: int = 1) -> Dict[str, Any]:
        """
        Comprehensive medical image analysis using Claude Vision
        Handles charts, graphs, medical images, and text extraction
        """
        logger.info(f"🔍 Analyzing medical image (page {page_number})...")
        
        try:
            # Detect content type for optimized analysis
            content_hints = self.detect_visual_content_type(image)
            analysis_type = content_hints['suggested_analysis']
            
            # Convert image to base64
            image_b64 = self.image_to_base64(image)
            if not image_b64:
                return {'error': 'Failed to convert image to base64'}
            
            # Create specialized prompt based on content type
            vision_prompt = self._create_vision_prompt(analysis_type, page_number)
            
            # Analyze with Claude Vision
            vision_response = self.aws_utils.safe_bedrock_vision_call(
                image_b64, 
                vision_prompt
            )
            
            # Process and structure the response
            analysis_result = {
                'page_number': page_number,
                'analysis_type': analysis_type,
                'content_hints': content_hints,
                'vision_response': vision_response,
                'medical_elements': self._extract_medical_elements(vision_response),
                'charts_detected': self._detect_charts_in_response(vision_response),
                'medical_values': self._extract_medical_values(vision_response),
                'clinical_significance': self._assess_clinical_significance(vision_response),
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
    
    def _create_vision_prompt(self, analysis_type: str, page_number: int) -> str:
        """
        Create specialized prompts based on detected content type
        """
        base_prompt = f"""You are analyzing page {page_number} of a medical document. """
        
        if analysis_type == 'charts_and_graphs':
            return base_prompt + """Focus on extracting data from charts, graphs, and visual data representations:

1. **Chart Type**: Identify the type of chart/graph (line graph, bar chart, table, etc.)
2. **Medical Data**: Extract specific values, trends, and measurements
3. **Time Series**: Note any temporal data or progression over time
4. **Reference Ranges**: Identify normal/abnormal ranges if shown
5. **Clinical Trends**: Describe any patterns or trends in the data
6. **Key Findings**: Highlight the most important medical insights

Provide specific numerical values and medical interpretations where possible."""

        elif analysis_type == 'medical_imaging':
            return base_prompt + """This appears to be a medical image or scan. Analyze:

1. **Image Type**: X-ray, MRI, CT, ultrasound, pathology slide, etc.
2. **Anatomical Structures**: Identify visible organs, bones, tissues
3. **Abnormal Findings**: Note any abnormalities, lesions, or pathological changes
4. **Medical Measurements**: Extract any measurements or quantitative data
5. **Technical Quality**: Assess image quality and diagnostic value
6. **Clinical Implications**: Suggest what this might indicate clinically

Be specific about medical findings and their potential significance."""

        elif analysis_type == 'text_and_charts':
            return base_prompt + """This page contains both text and visual elements. Analyze:

1. **Text Content**: Extract important medical information from text
2. **Visual Data**: Analyze any charts, tables, or graphs present
3. **Lab Results**: Extract numerical values and reference ranges
4. **Medical Relationships**: Connect text descriptions with visual data
5. **Critical Values**: Highlight any abnormal or concerning findings
6. **Treatment Plans**: Note any treatment recommendations or protocols

Integrate textual and visual information for comprehensive analysis."""

        else:  # general_medical
            return base_prompt + """Analyze this medical document page comprehensively:

1. **Content Overview**: Describe what type of medical content is present
2. **Key Medical Information**: Extract important patient data, diagnoses, treatments
3. **Visual Elements**: Identify and analyze any charts, graphs, or medical images
4. **Quantitative Data**: Extract all numerical values, measurements, lab results
5. **Clinical Significance**: Assess the medical importance of findings
6. **Action Items**: Note any recommendations, follow-ups, or critical alerts

Provide a thorough medical analysis focusing on clinically relevant information."""
    
    def _extract_medical_elements(self, vision_response: str) -> Dict[str, List[str]]:
        """
        Extract structured medical elements from vision analysis response
        """
        elements = {
            'medications': [],
            'procedures': [],
            'lab_values': [],
            'vital_signs': [],
            'diagnoses': [],
            'measurements': []
        }
        
        try:
            response_lower = vision_response.lower()
            
            # Extract medication mentions
            med_keywords = ['mg', 'mcg', 'units', 'dose', 'medication', 'drug', 'prescription']
            sentences = vision_response.split('.')
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in med_keywords):
                    elements['medications'].append(sentence.strip())
            
            # Extract lab values and measurements
            import re
            number_pattern = r'\d+\.?\d*\s*(?:mg/dl|mmol/l|mg|mcg|units|mmhg|bpm|%|\w+/\w+)'
            measurements = re.findall(number_pattern, vision_response, re.IGNORECASE)
            elements['measurements'].extend(measurements)
            
            # Extract vital signs patterns
            vital_patterns = [
                r'blood pressure:?\s*\d+/\d+',
                r'heart rate:?\s*\d+',
                r'temperature:?\s*\d+\.?\d*',
                r'respiratory rate:?\s*\d+'
            ]
            for pattern in vital_patterns:
                matches = re.findall(pattern, vision_response, re.IGNORECASE)
                elements['vital_signs'].extend(matches)
            
        except Exception as e:
            logger.warning(f"⚠️ Medical element extraction failed: {e}")
        
        return elements
    
    def _detect_charts_in_response(self, vision_response: str) -> Dict[str, Any]:
        """
        Detect if the response indicates presence of charts or graphs
        """
        chart_detection = {
            'has_charts': False,
            'chart_types': [],
            'data_trends': [],
            'time_series': False
        }
        
        try:
            response_lower = vision_response.lower()
            
            # Chart type detection
            chart_types = ['line graph', 'bar chart', 'pie chart', 'table', 'histogram', 'scatter plot']
            for chart_type in chart_types:
                if chart_type in response_lower:
                    chart_detection['chart_types'].append(chart_type)
                    chart_detection['has_charts'] = True
            
            # Trend detection
            trend_keywords = ['increasing', 'decreasing', 'stable', 'trending', 'pattern', 'over time']
            for keyword in trend_keywords:
                if keyword in response_lower:
                    chart_detection['data_trends'].append(keyword)
            
            # Time series detection
            time_keywords = ['over time', 'temporal', 'progression', 'months', 'days', 'weeks', 'years']
            chart_detection['time_series'] = any(keyword in response_lower for keyword in time_keywords)
            
        except Exception as e:
            logger.warning(f"⚠️ Chart detection failed: {e}")
        
        return chart_detection
    
    def _extract_medical_values(self, vision_response: str) -> List[Dict[str, Any]]:
        """
        Extract specific medical values and measurements
        """
        medical_values = []
        
        try:
            import re
            
            # Common medical value patterns
            patterns = {
                'blood_glucose': r'glucose:?\s*(\d+\.?\d*)\s*(mg/dl|mmol/l)?',
                'blood_pressure': r'(?:blood pressure|bp):?\s*(\d+)/(\d+)',
                'heart_rate': r'(?:heart rate|hr|pulse):?\s*(\d+)\s*(?:bpm)?',
                'temperature': r'temperature:?\s*(\d+\.?\d*)\s*(?:°f|°c|f|c)?',
                'hba1c': r'hba1c:?\s*(\d+\.?\d*)\s*%?',
                'cholesterol': r'cholesterol:?\s*(\d+)\s*(?:mg/dl)?'
            }
            
            for value_type, pattern in patterns.items():
                matches = re.finditer(pattern, vision_response, re.IGNORECASE)
                for match in matches:
                    medical_values.append({
                        'type': value_type,
                        'value': match.group(1),
                        'unit': match.group(2) if len(match.groups()) > 1 else None,
                        'full_match': match.group(0)
                    })
            
        except Exception as e:
            logger.warning(f"⚠️ Medical value extraction failed: {e}")
        
        return medical_values
    
    def _assess_clinical_significance(self, vision_response: str) -> Dict[str, Any]:
        """
        Assess clinical significance of findings
        """
        significance = {
            'urgency_level': 'routine',
            'abnormal_findings': [],
            'follow_up_needed': False,
            'critical_values': []
        }
        
        try:
            response_lower = vision_response.lower()
            
            # Urgency indicators
            urgent_keywords = ['critical', 'urgent', 'immediate', 'emergency', 'severe', 'acute']
            high_keywords = ['elevated', 'high', 'abnormal', 'concerning', 'significant']
            
            if any(keyword in response_lower for keyword in urgent_keywords):
                significance['urgency_level'] = 'urgent'
            elif any(keyword in response_lower for keyword in high_keywords):
                significance['urgency_level'] = 'elevated'
            
            # Abnormal findings
            abnormal_indicators = ['abnormal', 'elevated', 'low', 'high', 'outside range', 'concerning']
            for indicator in abnormal_indicators:
                if indicator in response_lower:
                    significance['abnormal_findings'].append(indicator)
            
            # Follow-up indicators
            followup_keywords = ['follow-up', 'monitor', 'repeat', 'recheck', 'reassess']
            significance['follow_up_needed'] = any(keyword in response_lower for keyword in followup_keywords)
            
        except Exception as e:
            logger.warning(f"⚠️ Clinical significance assessment failed: {e}")
        
        return significance
    
    def process_document_vision(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Main function: Complete vision processing pipeline for a document
        Converts PDF to images and analyzes each page
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
            
            # Analyze each page
            vision_results = []
            for i, image in enumerate(images):
                page_number = i + 1
                logger.info(f"📄 Processing page {page_number} of {len(images)}")
                
                page_analysis = self.analyze_medical_image(image, page_number)
                vision_results.append(page_analysis)
            
            # Generate summary of all pages
            summary = self._generate_document_vision_summary(vision_results)
            
            logger.info(f"✅ Vision processing complete: {len(vision_results)} pages analyzed")
            
            return vision_results
            
        except Exception as e:
            logger.error(f"❌ Document vision processing failed: {e}")
            return [{
                'error': str(e),
                'pdf_path': pdf_path
            }]
    
    def _generate_document_vision_summary(self, vision_results: List[Dict]) -> Dict[str, Any]:
        """
        Generate summary of vision analysis across all pages
        """
        summary = {
            'total_pages_analyzed': len(vision_results),
            'pages_with_charts': 0,
            'pages_with_medical_images': 0,
            'total_medical_values_found': 0,
            'urgent_findings': 0,
            'chart_types_detected': set(),
            'medical_elements_summary': {
                'medications': 0,
                'lab_values': 0,
                'vital_signs': 0
            }
        }
        
        try:
            for result in vision_results:
                if 'error' in result:
                    continue
                
                # Count charts
                if result.get('charts_detected', {}).get('has_charts', False):
                    summary['pages_with_charts'] += 1
                    summary['chart_types_detected'].update(
                        result['charts_detected'].get('chart_types', [])
                    )
                
                # Count medical values
                medical_values = result.get('medical_values', [])
                summary['total_medical_values_found'] += len(medical_values)
                
                # Check urgency
                clinical_sig = result.get('clinical_significance', {})
                if clinical_sig.get('urgency_level') == 'urgent':
                    summary['urgent_findings'] += 1
                
                # Count medical elements
                medical_elements = result.get('medical_elements', {})
                for element_type, elements in medical_elements.items():
                    if element_type in summary['medical_elements_summary']:
                        summary['medical_elements_summary'][element_type] += len(elements)
            
            # Convert set to list for JSON serialization
            summary['chart_types_detected'] = list(summary['chart_types_detected'])
            
        except Exception as e:
            logger.warning(f"⚠️ Vision summary generation failed: {e}")
        
        return summary
    
    def create_vision_description_for_vector_search(self, vision_results: List[Dict]) -> List[str]:
        """
        Create text descriptions from vision analysis for vector search integration
        This allows visual content to be searchable in the vector database
        """
        descriptions = []
        
        try:
            for result in vision_results:
                if 'error' in result:
                    continue
                
                page_num = result.get('page_number', 1)
                vision_response = result.get('vision_response', '')
                
                # Create structured description
                description_parts = [f"[PAGE {page_num} VISUAL ANALYSIS]:"]
                
                # Add main vision response
                if vision_response:
                    description_parts.append(vision_response)
                
                # Add extracted medical values
                medical_values = result.get('medical_values', [])
                if medical_values:
                    values_text = []
                    for value in medical_values:
                        value_str = f"{value['type']}: {value['value']}"
                        if value.get('unit'):
                            value_str += f" {value['unit']}"
                        values_text.append(value_str)
                    description_parts.append(f"Medical values detected: {', '.join(values_text)}")
                
                # Add chart information
                charts = result.get('charts_detected', {})
                if charts.get('has_charts'):
                    chart_info = f"Charts detected: {', '.join(charts.get('chart_types', []))}"
                    if charts.get('data_trends'):
                        chart_info += f". Trends: {', '.join(charts['data_trends'])}"
                    description_parts.append(chart_info)
                
                # Add clinical significance
                clinical_sig = result.get('clinical_significance', {})
                if clinical_sig.get('urgency_level') != 'routine':
                    description_parts.append(f"Clinical urgency: {clinical_sig['urgency_level']}")
                
                if clinical_sig.get('abnormal_findings'):
                    description_parts.append(f"Abnormal findings: {', '.join(clinical_sig['abnormal_findings'])}")
                
                # Combine into single description
                full_description = "\n".join(description_parts)
                descriptions.append(full_description)
            
            logger.info(f"✅ Created {len(descriptions)} vision descriptions for vector search")
            
        except Exception as e:
            logger.error(f"❌ Vision description creation failed: {e}")
        
        return descriptions

def test_vision_processor():
    """
    Test function for vision processor
    """
    print("🧪 Testing Medical Vision Processor...")
    
    try:
        # Initialize vision processor
        vision_processor = MedicalVisionProcessor()
        
        # Test image optimization
        if PDF2IMAGE_AVAILABLE:
            # Create test image
            test_image = Image.new('RGB', (800, 600), color='white')
            optimized = vision_processor._optimize_image_for_vision(test_image)
            print(f"✅ Image optimization: {optimized.size}")
        
        # Test base64 conversion
        test_image = Image.new('RGB', (100, 100), color='red')
        b64_string = vision_processor.image_to_base64(test_image)
        print(f"✅ Base64 conversion: {len(b64_string)} characters")
        
        # Test content type detection
        content_hints = vision_processor.detect_visual_content_type(test_image)
        print(f"✅ Content detection: {content_hints['suggested_analysis']}")
        
        # Test vision prompt creation
        prompt = vision_processor._create_vision_prompt('charts_and_graphs', 1)
        print(f"✅ Vision prompt creation: {len(prompt)} characters")
        
        # Test medical element extraction
        sample_response = """
        This chart shows blood glucose levels over 6 months.
        Initial reading: 180 mg/dl in January
        Current reading: 120 mg/dl in June
        Blood pressure: 130/80 mmHg
        Heart rate: 72 bpm
        Patient is taking metformin 500mg twice daily.
        """
        
        elements = vision_processor._extract_medical_elements(sample_response)
        print(f"✅ Medical element extraction: {sum(len(v) for v in elements.values())} elements found")
        
        values = vision_processor._extract_medical_values(sample_response)
        print(f"✅ Medical value extraction: {len(values)} values found")
        
        significance = vision_processor._assess_clinical_significance(sample_response)
        print(f"✅ Clinical significance: {significance['urgency_level']}")
        
        print("🎉 Vision processor test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Vision processor test FAILED: {e}")
        return False

if __name__ == "__main__":
    """
    Run this file directly to test vision processing
    """
    
    print("🚀 Medical Vision Processor - Person 4")
    print("=" * 50)
    
    # Test the vision processor
    if test_vision_processor():
        print("\n✅ Vision processor ready!")
        print("🔗 Integration points:")
        print("   - Person 2: Use vision results in complete document processing")
        print("   - Person 3: Use create_vision_description_for_vector_search() for vector storage")
        print("   - Person 5: Display vision analysis results in Gradio interface")
        print("\n📋 Key Functions available:")
        print("   - process_document_vision(pdf_path)")
        print("   - analyze_medical_image(image, page_number)")
        print("   - create_vision_description_for_vector_search(vision_results)")
        
        if not PDF2IMAGE_AVAILABLE:
            print("\n⚠️ Warning: pdf2image not available")
            print("   Install with: pip install pdf2image")
            print("   System dependency needed (poppler)")
    else:
        print("\n❌ Vision processor needs troubleshooting")
        print("🔧 Check:")
        print("   1. pdf2image library: pip install pdf2image")
        print("   2. Pillow library: pip install Pillow")
        print("   3. System poppler installation")
        print("   4. AWS configuration and Bedrock access")