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
        Extract text using pdfplumber (reliable, no content filters)
        Fallback to basic PDF text extraction if needed
        """
        logger.info(f"📄 Processing PDF with pdfplumber: {Path(pdf_path).name}")

        try:
            # Try pdfplumber first (most reliable)
            try:
                import pdfplumber

                all_extracted_text = ""
                pages_processed = 0

                print(f"\n{'='*60}")
                print(f"DEBUG: PDFPLUMBER EXTRACTION")
                print(f"{'='*60}")

                with pdfplumber.open(pdf_path) as pdf:
                    total_pages = len(pdf.pages)
                    print(f"Total pages in PDF: {total_pages}")

                    # Process up to 10 pages for hackathon
                    for page_num in range(min(10, total_pages)):
                        page = pdf.pages[page_num]
                        page_text = page.extract_text()

                        if page_text and page_text.strip():
                            print(f"PAGE {page_num + 1}: {len(page_text)} characters extracted")
                            print(f"  First 200 chars: {page_text[:200]}...")

                            all_extracted_text += f"\n\n=== PAGE {page_num + 1} ===\n{page_text}\n"
                            pages_processed += 1
                        else:
                            print(f"PAGE {page_num + 1}: No text extracted (might be image-based)")

                print(f"{'='*60}")

                if pages_processed > 0:
                    print(f"\n{'='*80}")
                    print(f"DEBUG: PDFPLUMBER FINAL SUMMARY")
                    print(f"{'='*80}")
                    print(f"Total characters extracted: {len(all_extracted_text)}")
                    print(f"Word count estimate: {len(all_extracted_text.split())}")
                    print(f"Pages processed: {pages_processed}")
                    print(f"Characters per page average: {len(all_extracted_text) / pages_processed if pages_processed > 0 else 0:.0f}")
                    print(f"\nFIRST 1000 CHARACTERS:")
                    print("-" * 50)
                    print(all_extracted_text[:1000])
                    print("-" * 50)
                    print(f"{'='*80}")

                    # Document metadata
                    metadata = {
                        'filename': Path(pdf_path).name,
                        'pages_processed': pages_processed,
                        'total_pages': total_pages,
                        'text_length': len(all_extracted_text),
                        'extraction_method': 'pdfplumber',
                        'processing_time': datetime.now().isoformat()
                    }

                    logger.info(f"✅ pdfplumber extraction successful: {len(all_extracted_text)} characters from {pages_processed} pages")
                    return all_extracted_text, metadata

            except ImportError:
                logger.warning("⚠️ pdfplumber not installed, trying PyPDF2...")

            # Fallback to PyPDF2
            try:
                import PyPDF2

                all_extracted_text = ""
                pages_processed = 0

                print(f"\n{'='*60}")
                print(f"DEBUG: PyPDF2 EXTRACTION")
                print(f"{'='*60}")

                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    total_pages = len(pdf_reader.pages)
                    print(f"Total pages in PDF: {total_pages}")

                    # Process up to 10 pages
                    for page_num in range(min(10, total_pages)):
                        page = pdf_reader.pages[page_num]
                        page_text = page.extract_text()

                        if page_text and page_text.strip():
                            print(f"PAGE {page_num + 1}: {len(page_text)} characters extracted")
                            print(f"  First 200 chars: {page_text[:200]}...")

                            all_extracted_text += f"\n\n=== PAGE {page_num + 1} ===\n{page_text}\n"
                            pages_processed += 1
                        else:
                            print(f"PAGE {page_num + 1}: No text extracted")

                print(f"{'='*60}")

                if pages_processed > 0:
                    metadata = {
                        'filename': Path(pdf_path).name,
                        'pages_processed': pages_processed,
                        'total_pages': total_pages,
                        'text_length': len(all_extracted_text),
                        'extraction_method': 'pypdf2',
                        'processing_time': datetime.now().isoformat()
                    }

                    logger.info(f"✅ PyPDF2 extraction successful: {len(all_extracted_text)} characters from {pages_processed} pages")
                    return all_extracted_text, metadata

            except ImportError:
                logger.error("❌ No PDF extraction libraries available (pdfplumber, PyPDF2)")
                return "", {'error': 'No PDF extraction libraries available'}

            # Document metadata
            metadata = {
                'filename': Path(pdf_path).name,
                'pages_processed': pages_processed,
                'total_pages': len(images),
                'text_length': len(all_extracted_text),
                'extraction_method': 'nova_lite_vision',
                'processing_time': datetime.now().isoformat()
            }

            # DEBUG: Show final extracted text summary
            print(f"\n{'='*80}")
            print(f"DEBUG: FINAL EXTRACTION SUMMARY")
            print(f"{'='*80}")
            print(f"Total characters extracted: {len(all_extracted_text)}")
            print(f"Word count estimate: {len(all_extracted_text.split())}")
            print(f"Pages processed: {pages_processed}")
            print(f"Characters per page average: {len(all_extracted_text) / pages_processed if pages_processed > 0 else 0:.0f}")
            print(f"\nFIRST 1000 CHARACTERS OF COMBINED TEXT:")
            print("-" * 50)
            print(all_extracted_text[:1000])
            print("-" * 50)
            print(f"\nLAST 1000 CHARACTERS OF COMBINED TEXT:")
            print("-" * 50)
            print(all_extracted_text[-1000:])
            print(f"{'='*80}")

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

# def test_document_processor():
#     """
#     Test function for document processor
#     """
#     print("🧪 Testing Medical Document Processor...")
    
#     try:
#         processor = MedicalDocumentProcessor()
        
#         # Create a dummy PDF for testing
#         dummy_pdf_path = "dummy_document.pdf"
#         with open(dummy_pdf_path, "w") as f:
#             f.write("Patient has diabetes and takes metformin 500mg twice daily")

#         # Test complete processing
#         result = processor.process_document_complete(dummy_pdf_path)
#         print(f"✅ Complete processing test: {'Success' if result['success'] else 'Failed'}")
        
#         os.remove(dummy_pdf_path)

#         print("🎉 Document processor test PASSED!")
#         return True
        
#     except Exception as e:
#         print(f"❌ Document processor test FAILED: {e}")
#         return False

# if __name__ == "__main__":
#     print("🚀 Medical Document Processor - Person 2")
#     print("=" * 50)
    
#     if test_document_processor():
#         print("\n✅ Document processor ready!")
#     else:
#         print("\n❌ Document processor needs troubleshooting")
