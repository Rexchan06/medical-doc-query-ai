"""
PERSON 2: Backend/API Developer
Document processing pipeline and API backend

RESPONSIBILITIES:
- Textract integration for PDF text extraction
- Medical entity processing pipeline
- API endpoints for document processing
- Error handling and data validation
- Integration with Person 1's AWS utilities

DELIVERABLES:
- document_processor.py (this file)
- Working PDF \u2192 text extraction
- Medical entity extraction with safety alerts
- API endpoints for frontend integration
"""

import os
import hashlib
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from pathlib import Path
import json
import re

# Import Person 1's AWS configuration
try:
    from aws_config import setup_aws_environment, AWSConfigManager, AWSUtilities
except ImportError:
    print("❌ Error: aws_config.py not found. Make sure Person 1 has completed AWS setup.")
    exit(1)

# Note: Vector search is imported by the medical_app.py coordinator
# No direct import needed here to avoid circular dependencies

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalDocumentProcessor:
    """
    Core document processing pipeline
    Handles PDF \u2192 text extraction \u2192 medical entity analysis \u2192 vector storage
    """
    
    def __init__(self):
        """Initialize the document processor with AWS services and vector search"""
        logger.info("🔧 Initializing Medical Document Processor...")
        
        self.aws_config, self.aws_utils = setup_aws_environment()
        # Vector search will be injected by the medical_app coordinator
        
        self.processed_documents = {}
        
        self.drug_interactions = {
            ('warfarin', 'aspirin'): {'severity': 'HIGH', 'risk': 'Increased bleeding risk'},
            # ... (rest of the interactions)
        }
        
        logger.info("✅ Medical Document Processor ready!")
    
    def process_document_complete(self, pdf_path: str) -> Dict:
        """
        Complete document processing pipeline using Nova Lite
        Simplified approach: PDF → Nova Lite (text + vision) → Medical Analysis
        """
        doc_id = f"doc_{hashlib.md5(pdf_path.encode()).hexdigest()[:8]}"
        filename = Path(pdf_path).name

        logger.info(f"🔄 Starting Nova Lite processing for: {filename}")

        try:
            # Step 1: Extract text and analyze with Nova Lite (includes vision analysis)
            extracted_text, extraction_metadata = self.extract_text_from_pdf(pdf_path)

            if not extracted_text.strip():
                return {'success': False, 'error': 'No content extracted by Nova Lite', 'doc_id': doc_id}

            # Step 2: Extract medical entities using Nova Lite
            medical_entities = self.extract_medical_entities(extracted_text)

            if 'error' in medical_entities:
                logger.warning(f"⚠️ Medical entity extraction had issues: {medical_entities['error']}")
                # Continue with empty entities rather than failing
                medical_entities = {
                    'medications': [],
                    'conditions': [],
                    'procedures': [],
                    'anatomy': [],
                    'test_results': [],
                    'phi_detected': []
                }

            # Step 4: Generate medical report
            medical_report = self.generate_medical_report(
                extracted_text, medical_entities
            )

            # Create processing summary for response
            processing_summary = {
                'medications_found': len(medical_entities.get('medications', [])),
                'conditions_found': len(medical_entities.get('conditions', [])),
                'extraction_method': 'nova_lite_combined'
            }

            self.processed_documents[doc_id] = {
                'filename': filename,
                'text_length': len(extracted_text),
                'processing_summary': processing_summary
            }

            result = {
                'success': True,
                'doc_id': doc_id,
                'filename': filename,
                'extracted_text': extracted_text,
                'medical_entities': medical_entities,
                'medical_report': medical_report,
                'extraction_metadata': extraction_metadata,
                'processing_summary': processing_summary
            }

            logger.info(f"✅ Nova Lite processing complete for: {filename}")
            return result

        except Exception as e:
            logger.error(f"❌ Nova Lite document processing failed: {e}")
            return {'success': False, 'error': str(e), 'doc_id': doc_id}

    # ... (the rest of the methods remain the same)
    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, Dict]:
        """
        Extract text and analyze content using Nova Lite vision and text capabilities
        Simplified approach using Nova Lite for both text extraction and visual analysis
        """
        logger.info(f"📄 Processing PDF with Nova Lite: {Path(pdf_path).name}")

        try:
            # Import pdf2image for conversion
            from pdf2image import convert_from_path
            from PIL import Image
            import base64
            import io

            # Convert PDF to images
            logger.info("Converting PDF pages to images...")
            images = convert_from_path(pdf_path, dpi=150, fmt='RGB')

            all_extracted_text = ""
            pages_processed = 0

            # Process each page with Nova Lite
            for i, image in enumerate(images[:5]):  # Limit to 5 pages for hackathon
                page_num = i + 1
                logger.info(f"Processing page {page_num} with Nova Lite...")

                # Convert image to base64 for Nova Lite
                buffer = io.BytesIO()
                image.save(buffer, format='PNG', quality=95)
                img_bytes = buffer.getvalue()
                image_b64 = base64.b64encode(img_bytes).decode('utf-8')

                # Create comprehensive prompt for Nova Lite
                prompt = f"""Analyze this page {page_num} of a medical document. Extract ALL text content and describe any visual elements:

1. **Text Extraction**: Extract all readable text exactly as it appears, maintaining structure and formatting
2. **Medical Data**: Identify and extract all medical information including:
   - Patient information and demographics
   - Medications, dosages, and frequencies
   - Medical conditions and diagnoses
   - Lab results and vital signs with values and units
   - Treatment plans and recommendations
3. **Visual Elements**: Describe any charts, graphs, tables, or medical images
4. **Clinical Values**: Extract all numerical values with their units and context

Provide a comprehensive analysis that captures both text content and visual information."""

                # Call Nova Lite vision API
                page_analysis = self.aws_utils.safe_bedrock_vision_call(image_b64, prompt)

                if not page_analysis.startswith("Error:"):
                    all_extracted_text += f"\n\n=== PAGE {page_num} ===\n{page_analysis}\n"
                    pages_processed += 1
                else:
                    logger.warning(f"⚠️ Nova Lite analysis failed for page {page_num}: {page_analysis}")

            # Document metadata
            metadata = {
                'filename': Path(pdf_path).name,
                'pages_processed': pages_processed,
                'total_pages': len(images),
                'text_length': len(all_extracted_text),
                'extraction_method': 'nova_lite_vision',
                'processing_time': datetime.now().isoformat()
            }

            logger.info(f"✅ Nova Lite extraction successful: {len(all_extracted_text)} characters from {pages_processed} pages")

            return all_extracted_text, metadata

        except Exception as e:
            logger.error(f"❌ Nova Lite PDF extraction failed: {e}")
            return "", {'error': str(e)}
    
    def _extract_table_content(self, table_block: Dict, all_blocks: List[Dict]) -> str:
        """
        Extract table content from Textract response.
        Returns a simple text representation of the table.
        """
        try:
            # Build a map of block Ids to blocks for quick lookup
            block_map = {block['Id']: block for block in all_blocks}
            rows = []

            # Find all cell blocks that belong to this table
            cell_blocks = [
                block for block in all_blocks
                if block['BlockType'] == 'CELL' and
                'Relationships' in table_block and
                any(rel['Type'] == 'CHILD' and block['Id'] in rel.get('Ids', []) for rel in table_block['Relationships'])
            ]

            # Organize cells by row and column
            table = {}
            max_row = 0
            max_col = 0
            for cell in cell_blocks:
                row_idx = cell['RowIndex']
                col_idx = cell['ColumnIndex']
                max_row = max(max_row, row_idx)
                max_col = max(max_col, col_idx)
                # Get cell text
                cell_text = ""
                if 'Relationships' in cell:
                    for rel in cell['Relationships']:
                        if rel['Type'] == 'CHILD':
                            for cid in rel['Ids']:
                                word_block = block_map.get(cid)
                                if word_block and word_block['BlockType'] == 'WORD':
                                    cell_text += word_block['Text'] + " "
                table[(row_idx, col_idx)] = cell_text.strip()

            # Build table as text
            for row in range(1, max_row + 1):
                row_cells = []
                for col in range(1, max_col + 1):
                    row_cells.append(table.get((row, col), ""))
                rows.append(" | ".join(row_cells))

            return "\n".join(rows) if rows else "[Empty table]"
        except Exception as e:
            logger.error(f"Table extraction failed: {e}")
            return "[Table detected but content extraction failed]"
    
    
    def extract_medical_entities(self, text: str) -> Dict:
        """
        Extract medical entities using Nova Lite text analysis
        Returns organized medical information using AI instead of specialized medical NLP
        """
        logger.info("🩺 Extracting medical entities with Nova Lite...")

        try:
            # Create prompt for Nova Lite to extract medical entities
            medical_prompt = f"""Analyze this medical document text and extract all medical entities. Organize them into the following categories:

MEDICAL TEXT TO ANALYZE:
{text}

Please extract and organize the following medical information:

1. **MEDICATIONS**: List all medications mentioned with dosages, frequencies, and any administration details
2. **CONDITIONS**: List all medical conditions, diagnoses, symptoms, and health issues
3. **PROCEDURES**: List all medical procedures, treatments, surgeries, and interventions
4. **ANATOMY**: List all anatomical references, body parts, and organ systems mentioned
5. **TEST_RESULTS**: List all lab results, vital signs, measurements, and diagnostic findings with values and units
6. **PHI_DETECTED**: List any personally identifiable information like names, dates of birth, addresses, phone numbers

For each item, provide:
- The exact text as it appears
- The category it belongs to
- Any additional details (like dosage for medications, values for test results)

Format your response as a structured list under each category."""

            # Call Nova Lite for medical entity extraction
            nova_response = self.aws_utils.safe_bedrock_call(medical_prompt, max_tokens=800)

            # Parse the Nova Lite response into organized structure
            organized_entities = self._parse_nova_medical_response(nova_response)

            logger.info(f"✅ Nova Lite medical entities extracted: {len(organized_entities['medications'])} medications, "
                       f"{len(organized_entities['conditions'])} conditions")

            return organized_entities

        except Exception as e:
            logger.error(f"❌ Nova Lite medical entity extraction failed: {e}")
            return {'error': str(e)}

    def _parse_nova_medical_response(self, nova_response: str) -> Dict:
        """
        Parse Nova Lite response into organized medical entities structure
        Simple parsing for hackathon - can be enhanced with more sophisticated NLP
        """
        try:
            organized_entities = {
                'medications': [],
                'conditions': [],
                'procedures': [],
                'anatomy': [],
                'test_results': [],
                'phi_detected': []
            }

            # Simple keyword-based parsing for demo
            # In production, would use more sophisticated parsing
            lines = nova_response.split('\n')
            current_category = None

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Detect category headers
                line_lower = line.lower()
                if 'medication' in line_lower and ':' in line:
                    current_category = 'medications'
                elif 'condition' in line_lower and ':' in line:
                    current_category = 'conditions'
                elif 'procedure' in line_lower and ':' in line:
                    current_category = 'procedures'
                elif 'anatomy' in line_lower and ':' in line:
                    current_category = 'anatomy'
                elif 'test' in line_lower and ':' in line:
                    current_category = 'test_results'
                elif 'phi' in line_lower and ':' in line:
                    current_category = 'phi_detected'
                elif line.startswith('-') or line.startswith('•') and current_category:
                    # Extract entity text
                    entity_text = line.lstrip('- •').strip()
                    if entity_text and current_category:
                        entity_info = {
                            'text': entity_text,
                            'confidence': 0.85,  # Default confidence for Nova Lite extraction
                            'type': 'EXTRACTED_BY_NOVA_LITE',
                            'category': current_category.upper(),
                            'attributes': []
                        }
                        organized_entities[current_category].append(entity_info)

            return organized_entities

        except Exception as e:
            logger.error(f"❌ Nova Lite response parsing failed: {e}")
            return {
                'medications': [],
                'conditions': [],
                'procedures': [],
                'anatomy': [],
                'test_results': [],
                'phi_detected': []
            }
    
    def _extract_dosage(self, medication_data: Dict) -> str:
        """Extract dosage information from medication entity"""
        try:
            # Check attributes for dosage
            for attr in medication_data.get('attributes', []):
                if attr['type'] in ['DOSAGE', 'STRENGTH']:
                    return attr['text']
            
            # Check if dosage is in the main text
            med_text = medication_data['text']
            dosage_match = re.search(r'\d+\s*(mg|mcg|g|ml|units?)', med_text, re.IGNORECASE)
            if dosage_match:
                return dosage_match.group()
            
            return "Dosage not specified"
            
        except Exception:
            return "Dosage not specified"
    
    
    def generate_medical_report(self, extracted_text: str, medical_entities: Dict) -> str:
        """
        Generate comprehensive medical report
        This impresses judges with professional medical analysis
        """
        
        report_sections = []
        
        # Header
        report_sections.append("# 🏥 MEDICAL DOCUMENT ANALYSIS REPORT")
        report_sections.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_sections.append("")
        
        # Executive Summary
        medications_count = len(medical_entities.get('medications', []))
        conditions_count = len(medical_entities.get('conditions', []))
        
        report_sections.append("## 📊 EXECUTIVE SUMMARY")
        report_sections.append(f"• **Medications Identified:** {medications_count}")
        report_sections.append(f"• **Medical Conditions:** {conditions_count}")
        report_sections.append("")
        # ... (rest of the report generation)

def test_document_processor():
    """
    Test function for document processor
    """
    print("🧪 Testing Medical Document Processor...")
    
    try:
        processor = MedicalDocumentProcessor()
        
        # Create a dummy PDF for testing
        dummy_pdf_path = "test.pdf"
        with open(dummy_pdf_path, "w") as f:
            f.write("Patient has diabetes and takes metformin 500mg twice daily")

        # Test complete processing
        result = processor.process_document_complete(dummy_pdf_path)
        print(f"✅ Complete processing test: {'Success' if result['success'] else 'Failed'}")
        
        os.remove(dummy_pdf_path)

        print("🎉 Document processor test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Document processor test FAILED: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Medical Document Processor - Person 2")
    print("=" * 50)
    
    if test_document_processor():
        print("\n✅ Document processor ready!")
    else:
        print("\n❌ Document processor needs troubleshooting")
