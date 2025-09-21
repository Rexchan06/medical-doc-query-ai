#!/usr/bin/env python3
"""
PERSON 5: Frontend + Demo Lead
Gradio interface, integration, and demo preparation

RESPONSIBILITIES:
- Complete Gradio web interface
- Integration of all team components (Persons 1-4)
- Demo preparation and user experience
- Error handling and user feedback
- Presentation and demo materials

DELIVERABLES:
- medical_app.py (this file)
- Complete working web application
- Demo scenarios and test cases
- Integration of all backend components
"""

import gradio as gr
import logging
import traceback
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
import json
import os
import tempfile

# Import all team components
try:
    # Person 1: AWS Infrastructure
    from aws_config import setup_aws_environment, AWSConfigManager, AWSUtilities
    
    # Person 2: Document Processing
    from document_processor import MedicalDocumentProcessor
    
    # Person 3: Vector Search
    from vector_search import MedicalVectorSearch
    
    # Person 4: Vision Processing
    from vision_processor import MedicalVisionProcessor
    
    ALL_COMPONENTS_AVAILABLE = True
    
except ImportError as e:
    print(f"❌ Error importing team components: {e}")
    print("🔧 Make sure all team members have completed their components:")
    print("   - Person 1: aws_config.py")
    print("   - Person 2: document_processor.py") 
    print("   - Person 3: vector_search.py")
    print("   - Person 4: vision_processor.py")
    ALL_COMPONENTS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalDocumentApp:
    """
    Complete Medical Document Analysis Application
    Integrates all team components into a unified system
    """
    
    def __init__(self):
        """Initialize the complete medical document application"""
        logger.info("🚀 Initializing Complete Medical Document Application...")
        
        if not ALL_COMPONENTS_AVAILABLE:
            raise ImportError("Not all team components are available. Check imports above.")
        
        # Initialize all system components
        self._initialize_components()
        
        # Application state
        self.processed_documents = {}
        self.session_stats = {
            'documents_processed': 0,
            'queries_answered': 0,
            'session_start': datetime.now().isoformat()
        }
        
        # Demo mode settings
        self.demo_mode = False
        self.demo_documents = []
        
        logger.info("✅ Medical Document Application ready!")
    
    def _initialize_components(self):
        """Initialize all team components"""
        try:
            logger.info("🔧 Initializing system components...")
            
            # Person 1: AWS Infrastructure
            self.aws_config, self.aws_utils = setup_aws_environment()
            
            # Person 2: Document Processing
            self.document_processor = MedicalDocumentProcessor()
            
            # Person 3: Vector Search
            self.vector_search = MedicalVectorSearch()
            
            # Person 4: Vision Processing
            self.vision_processor = MedicalVisionProcessor()
            
            logger.info("✅ All components initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Component initialization failed: {e}")
            raise
    
    def process_document_complete(self, pdf_file, progress_callback=None) -> Dict[str, Any]:
        """
        Simplified document processing pipeline using Nova Lite
        PDF → Nova Lite (text + vision) → Kendra storage → Results
        """
        if pdf_file is None:
            return {
                'success': False,
                'error': 'No file uploaded',
                'user_message': 'Please upload a PDF file to analyze.'
            }

        # Handle both Gradio file objects and file paths
        if hasattr(pdf_file, 'name'):
            # Gradio file object
            pdf_path = pdf_file.name
        else:
            # Already a file path
            pdf_path = str(pdf_file)

        return self.process_document_complete_by_path(pdf_path, progress_callback)

    def process_document_complete_by_path(self, pdf_path: str, progress_callback=None) -> Dict[str, Any]:
        """
        Process document using file path
        """
        try:
            filename = Path(pdf_path).name
            logger.info(f"🔄 Starting Nova Lite processing for: {filename}")

            if progress_callback:
                progress_callback(0.1, "Starting Nova Lite document analysis...")

            # Step 1: Nova Lite document processing (text + vision combined)
            logger.info("🤖 Step 1: Processing with Nova Lite (text + vision analysis)...")
            doc_result = self.document_processor.process_document_complete(pdf_path)

            if not doc_result['success']:
                return {
                    'success': False,
                    'error': doc_result['error'],
                    'user_message': f"Nova Lite processing failed: {doc_result['error']}"
                }

            if progress_callback:
                progress_callback(0.6, "Nova Lite analysis complete. Storing in Kendra...")

            # Step 2: Store in Kendra vector database
            logger.info("🗄️ Step 2: Storing in Kendra vector database...")

            # The extracted_text from Nova Lite already includes vision analysis
            vector_success = self.vector_search.store_document_vectors(
                doc_result['doc_id'],
                doc_result['extracted_text'],  # This includes both text and vision analysis
                doc_result['medical_entities'],
                [],  # No separate vision descriptions needed - already in extracted_text
                doc_result['extraction_metadata'],
                pdf_path  # Pass PDF file path for direct processing fallback
            )

            if not vector_success:
                logger.warning("⚠️ Kendra storage failed, but continuing with basic functionality")

            if progress_callback:
                progress_callback(0.9, "Finalizing results...")

            # Create comprehensive summary
            comprehensive_summary = self._create_nova_lite_summary(doc_result)

            # Compile results
            comprehensive_result = {
                'success': True,
                'doc_id': doc_result['doc_id'],
                'filename': filename,
                'document_processing': doc_result,
                'vector_search_enabled': vector_success,
                'processing_timestamp': datetime.now().isoformat(),
                'comprehensive_summary': comprehensive_summary
            }

            # Update session stats
            self.session_stats['documents_processed'] += 1

            # Store in app state
            self.processed_documents[doc_result['doc_id']] = comprehensive_result

            if progress_callback:
                progress_callback(1.0, "Nova Lite processing complete!")

            logger.info(f"✅ Nova Lite processing finished for: {filename}")
            return comprehensive_result

        except Exception as e:
            logger.error(f"❌ Nova Lite processing failed: {e}")
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e),
                'user_message': f"An error occurred during Nova Lite processing: {str(e)}"
            }

    def process_multiple_documents(self, pdf_files, progress_callback=None) -> Dict[str, Any]:
        """
        Process multiple PDF documents with Nova Lite and create unified analysis
        """
        if pdf_files is None or len(pdf_files) == 0:
            return {
                'success': False,
                'error': 'No files uploaded',
                'user_message': 'Please upload at least one PDF file to analyze.'
            }

        try:
            total_files = len(pdf_files)
            all_results = []
            failed_files = []

            logger.info(f"🔄 Starting multi-document Nova Lite processing for {total_files} files")

            # Process each document individually
            for i, pdf_file in enumerate(pdf_files, 1):
                filename = Path(pdf_file.name).name

                if progress_callback:
                    progress_callback(i / (total_files + 1), f"Processing file {i}/{total_files}: {filename}")

                logger.info(f"📄 Processing document {i}/{total_files}: {filename}")

                # Process individual document
                # Extract file path from Gradio file object
                file_path = pdf_file.name if hasattr(pdf_file, 'name') else str(pdf_file)
                doc_result = self.process_document_complete_by_path(file_path)

                if doc_result['success']:
                    all_results.append(doc_result)
                    logger.info(f"✅ Successfully processed: {filename}")
                else:
                    failed_files.append({'filename': filename, 'error': doc_result['error']})
                    logger.warning(f"❌ Failed to process: {filename} - {doc_result['error']}")

            if progress_callback:
                progress_callback(0.95, "Generating cross-document analysis...")

            # Create unified multi-document summary
            if all_results:
                unified_summary = self._create_multi_document_summary(all_results, failed_files)

                # Update session stats
                self.session_stats['documents_processed'] += len(all_results)

                if progress_callback:
                    progress_callback(1.0, f"Multi-document processing complete! Processed {len(all_results)}/{total_files} files")

                return {
                    'success': True,
                    'total_files': total_files,
                    'successful_files': len(all_results),
                    'failed_files': len(failed_files),
                    'individual_results': all_results,
                    'failed_results': failed_files,
                    'unified_summary': unified_summary,
                    'processing_timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'error': 'All documents failed to process',
                    'user_message': f'Failed to process any of the {total_files} uploaded documents.',
                    'failed_results': failed_files
                }

        except Exception as e:
            logger.error(f"❌ Multi-document processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'user_message': f"An error occurred during multi-document processing: {str(e)}"
            }
    
    def _create_multi_document_summary(self, all_results: List[Dict], failed_files: List[Dict]) -> str:
        """
        Create comprehensive cross-document analysis summary
        """
        summary_parts = []

        # Header
        summary_parts.append("# 🏥 MULTI-DOCUMENT MEDICAL ANALYSIS")
        summary_parts.append(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary_parts.append(f"**Documents Processed:** {len(all_results)}")
        if failed_files:
            summary_parts.append(f"**Failed Documents:** {len(failed_files)}")
        summary_parts.append(f"**Analysis Method:** Nova Lite AI (Combined Text + Vision)")
        summary_parts.append("")

        # Executive Summary
        total_medications = sum(result.get('document_processing', {}).get('processing_summary', {}).get('medications_found', 0) for result in all_results)
        total_conditions = sum(result.get('document_processing', {}).get('processing_summary', {}).get('conditions_found', 0) for result in all_results)
        total_safety_alerts = sum(len(result.get('document_processing', {}).get('safety_alerts', [])) for result in all_results)
        total_pages = sum(result.get('document_processing', {}).get('extraction_metadata', {}).get('pages_processed', 0) for result in all_results)

        summary_parts.append("## 📊 CROSS-DOCUMENT EXECUTIVE SUMMARY")
        summary_parts.append(f"• **Total Medications Identified:** {total_medications}")
        summary_parts.append(f"• **Total Medical Conditions:** {total_conditions}")
        summary_parts.append(f"• **Total Pages Analyzed:** {total_pages}")
        summary_parts.append(f"• **Safety Alerts Generated:** {total_safety_alerts}")
        summary_parts.append(f"• **Analysis Coverage:** {len(all_results)} patients/documents")
        summary_parts.append("")

        # Cross-Document Safety Analysis
        all_safety_alerts = []
        for result in all_results:
            alerts = result.get('document_processing', {}).get('safety_alerts', [])
            for alert in alerts:
                alert['source_doc'] = result.get('filename', 'Unknown')
                all_safety_alerts.append(alert)

        if all_safety_alerts:
            summary_parts.append("## 🚨 CRITICAL CROSS-DOCUMENT SAFETY ALERTS")
            for i, alert in enumerate(all_safety_alerts, 1):
                severity_emoji = "🚨" if alert['severity'] == 'HIGH' else "⚠️"
                summary_parts.append(f"### {severity_emoji} Alert {i} - {alert['severity']} RISK (from {alert['source_doc']})")
                summary_parts.append(f"**Interaction:** {alert['drug1']['name']} + {alert['drug2']['name']}")
                summary_parts.append(f"**Risk:** {alert['risk_description']}")
                summary_parts.append(f"**Action Required:** {alert['recommended_action']}")
                summary_parts.append("")
        else:
            summary_parts.append("## ✅ CROSS-DOCUMENT SAFETY STATUS")
            summary_parts.append("No drug interactions detected across all documents.")
            summary_parts.append("")

        # Individual Document Summaries
        summary_parts.append("## 📋 INDIVIDUAL DOCUMENT SUMMARIES")
        for i, result in enumerate(all_results, 1):
            doc_processing = result.get('document_processing', {})
            processing_summary = doc_processing.get('processing_summary', {})

            summary_parts.append(f"### Document {i}: {result.get('filename', 'Unknown')}")
            summary_parts.append(f"• **Medications:** {processing_summary.get('medications_found', 0)}")
            summary_parts.append(f"• **Conditions:** {processing_summary.get('conditions_found', 0)}")
            summary_parts.append(f"• **Safety Alerts:** {len(doc_processing.get('safety_alerts', []))}")

            # Brief medical report excerpt
            medical_report = doc_processing.get('medical_report', '')
            if medical_report:
                report_excerpt = medical_report[:300] + "..." if len(medical_report) > 300 else medical_report
                summary_parts.append(f"• **Key Findings:** {report_excerpt}")
            summary_parts.append("")

        # Failed Files Summary
        if failed_files:
            summary_parts.append("## ❌ PROCESSING FAILURES")
            for failed in failed_files:
                summary_parts.append(f"• **{failed['filename']}:** {failed['error']}")
            summary_parts.append("")

        # Population Health Insights
        if len(all_results) > 1:
            summary_parts.append("## 🏥 POPULATION HEALTH INSIGHTS")
            summary_parts.append("**Cross-Patient Analysis:**")
            summary_parts.append(f"• Analyzed {len(all_results)} patient records")
            summary_parts.append(f"• Average medications per patient: {total_medications / len(all_results):.1f}")
            summary_parts.append(f"• Average conditions per patient: {total_conditions / len(all_results):.1f}")
            if total_safety_alerts > 0:
                summary_parts.append(f"• {(total_safety_alerts / len(all_results) * 100):.1f}% of patients have safety alerts")
            summary_parts.append("")
            summary_parts.append("**Recommended Actions:**")
            summary_parts.append("• Review all safety alerts before prescribing")
            summary_parts.append("• Consider cross-patient medication patterns")
            summary_parts.append("• Monitor for population-level adverse events")

        return "\n".join(summary_parts)

    def _create_nova_lite_summary(self, doc_result: Dict) -> str:
        """
        Create a comprehensive summary from Nova Lite analysis results
        Simplified version that works with combined text + vision analysis
        """
        summary_parts = []

        # Header
        summary_parts.append("# 🏥 COMPREHENSIVE MEDICAL DOCUMENT ANALYSIS")
        summary_parts.append(f"**Document:** {doc_result.get('filename', 'Unknown')}")
        summary_parts.append(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary_parts.append(f"**Analysis Method:** Nova Lite AI (Combined Text + Vision)")
        summary_parts.append("")

        # Executive Summary
        processing_summary = doc_result.get('processing_summary', {})
        medications_count = processing_summary.get('medications_found', 0)
        conditions_count = processing_summary.get('conditions_found', 0)
        has_high_risk = processing_summary.get('has_high_risk_interactions', False)
        pages_processed = doc_result.get('extraction_metadata', {}).get('pages_processed', 0)

        summary_parts.append("## 📊 EXECUTIVE SUMMARY")
        summary_parts.append(f"• **Medications Identified:** {medications_count}")
        summary_parts.append(f"• **Medical Conditions:** {conditions_count}")
        summary_parts.append(f"• **Pages Analyzed:** {pages_processed}")
        summary_parts.append(f"• **Analysis Method:** AI-powered text + vision extraction")
        summary_parts.append("")

        # Safety Alerts Section
        safety_alerts = doc_result.get('safety_alerts', [])
        if safety_alerts:
            summary_parts.append("## 🚨 CRITICAL SAFETY ALERTS")
            for alert in safety_alerts:
                severity_emoji = "🚨" if alert['severity'] == 'HIGH' else "⚠️"
                summary_parts.append(f"### {severity_emoji} {alert['severity']} RISK")
                summary_parts.append(f"**Interaction:** {alert['drug1']['name']} + {alert['drug2']['name']}")
                summary_parts.append(f"**Risk:** {alert['risk_description']}")
                summary_parts.append(f"**Action Required:** {alert['recommended_action']}")
                summary_parts.append("")
        else:
            summary_parts.append("## ✅ SAFETY STATUS")
            summary_parts.append("No drug interactions detected.")
            summary_parts.append("")

        # Medical Report Section
        if doc_result.get('medical_report'):
            summary_parts.append("## 📋 DETAILED MEDICAL ANALYSIS")
            summary_parts.append(doc_result['medical_report'])

        return "\n".join(summary_parts)

    def _create_comprehensive_summary(self, doc_result: Dict, vision_results: List[Dict]) -> str:
        """
        Create a comprehensive summary combining all analysis results
        """
        summary_parts = []
        
        # Header
        summary_parts.append("# 🏥 COMPREHENSIVE MEDICAL DOCUMENT ANALYSIS")
        summary_parts.append(f"**Document:** {doc_result.get('filename', 'Unknown')}")
        summary_parts.append(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary_parts.append("")
        
        # Executive Summary
        processing_summary = doc_result.get('processing_summary', {})
        medications_count = processing_summary.get('medications_found', 0)
        has_high_risk = processing_summary.get('has_high_risk_interactions', False)
        
        # Count vision analysis results
        pages_analyzed = len([r for r in vision_results if 'error' not in r])
        charts_found = sum(1 for r in vision_results if r.get('charts_detected', {}).get('has_charts', False))
        
        summary_parts.append("## 📊 EXECUTIVE SUMMARY")
        summary_parts.append(f"• **Medications Identified:** {medications_count}")
        summary_parts.append(f"• **Pages with Visual Content:** {pages_analyzed}")
        summary_parts.append(f"• **Charts/Graphs Analyzed:** {charts_found}")
        summary_parts.append("")
        
        # Safety Alerts Section
        safety_alerts = doc_result.get('safety_alerts', [])
        if safety_alerts:
            summary_parts.append("## 🚨 CRITICAL SAFETY ALERTS")
            for alert in safety_alerts:
                severity_emoji = "🚨" if alert['severity'] == 'HIGH' else "⚠️"
                summary_parts.append(f"### {severity_emoji} {alert['severity']} RISK")
                summary_parts.append(f"**Interaction:** {alert['drug1']['name']} + {alert['drug2']['name']}")
                summary_parts.append(f"**Risk:** {alert['risk_description']}")
                summary_parts.append(f"**Action Required:** {alert['recommended_action']}")
                summary_parts.append("")
        else:
            summary_parts.append("## ✅ SAFETY STATUS")
            summary_parts.append("No drug interactions detected.")
            summary_parts.append("")
        
        # Visual Analysis Summary
        if charts_found > 0:
            summary_parts.append("## 📊 VISUAL CONTENT ANALYSIS")
            for result in vision_results:
                if 'error' in result:
                    continue
                
                page_num = result.get('page_number', 1)
                if result.get('charts_detected', {}).get('has_charts', False):
                    summary_parts.append(f"### Page {page_num}")
                    
                    chart_types = result['charts_detected'].get('chart_types', [])
                    if chart_types:
                        summary_parts.append(f"**Chart Types:** {', '.join(chart_types)}")
                    
                    medical_values = result.get('medical_values', [])
                    if medical_values:
                        values_text = []
                        for value in medical_values[:3]:  # Top 3 values
                            value_str = f"{value['type']}: {value['value']}"
                            if value.get('unit'):
                                value_str += f" {value['unit']}"
                            values_text.append(value_str)
                        summary_parts.append(f"**Key Values:** {', '.join(values_text)}")
                    
                    summary_parts.append("")
        
        # Medical Report Section
        if doc_result.get('medical_report'):
            summary_parts.append("## 📋 DETAILED MEDICAL ANALYSIS")
            summary_parts.append(doc_result['medical_report'])
        
        return "\n".join(summary_parts)
    
    def query_documents(self, question: str, k_value: int = 7,
                       use_pattern_analysis: bool = True) -> str:
        """
        Simplified query processing using Nova Lite
        Query → Kendra search → Nova Lite response generation
        """
        if not question.strip():
            return "❓ Please enter a question about your medical documents."

        try:
            logger.info(f"🔍 Processing query with Nova Lite: '{question}'")

            # Update session stats
            self.session_stats['queries_answered'] += 1

            # Step 1: Multi-document vector search using Kendra
            print(f"\n{'='*80}")
            print(f"DEBUG: SEARCH QUERY")
            print(f"{'='*80}")
            print(f"Question: '{question}'")
            print(f"K value: {k_value}")
            print(f"Searching across documents...")
            print(f"{'='*80}")

            search_results = self.vector_search.search_across_documents(
                question, k=k_value
            )

            # DEBUG: Show search results
            print(f"\n{'='*80}")
            print(f"DEBUG: SEARCH RESULTS")
            print(f"{'='*80}")
            print(f"Number of results found: {len(search_results)}")
            for i, result in enumerate(search_results[:3]):
                print(f"\nRESULT {i+1}:")
                print(f"  Source: {result.get('source', 'unknown')}")
                print(f"  Filename: {result.get('filename', 'unknown')}")
                print(f"  Doc ID: {result.get('doc_id', 'unknown')}")
                print(f"  Similarity: {result.get('similarity_score', 'unknown')}")
                content_preview = result.get('content', '')[:300]
                print(f"  Content preview: {content_preview}...")
            print(f"{'='*80}")

            if not search_results:
                total_docs = len(self.processed_documents)
                return f"""❌ **No relevant information found**

I couldn't find information related to your question in the {total_docs} document(s) currently processed.

**Suggestions:**
• Try rephrasing with different medical terms
• Upload more relevant medical documents
• Check if your question relates to the uploaded content

**Example queries:**
• "What medications are prescribed?"
• "What symptoms are mentioned across all patients?"
• "Compare treatment outcomes between documents"
"""

            # Step 2: Generate intelligent response using Nova Lite
            logger.info("🤖 Generating response with Nova Lite...")

            # Compile context from search results
            context_parts = []
            has_direct_pdf_result = any(result.get('source') == 'direct_pdf' for result in search_results)

            for i, result in enumerate(search_results[:5]):  # Use top 5 results
                context_parts.append(f"**Source {i+1}** (from {result['filename']}):")

                # For direct PDF results, use FULL content. For others, truncate to avoid overwhelming context
                if result.get('source') == 'direct_pdf':
                    logger.info(f"🔥 Using FULL content from direct PDF processing ({len(result['content'])} characters)")
                    context_parts.append(result['content'])  # Use full content for direct PDF
                else:
                    # Truncate other sources to 500 chars
                    context_parts.append(result['content'][:500] + "..." if len(result['content']) > 500 else result['content'])

                context_parts.append("")

            context_text = "\n".join(context_parts)

            # Create Nova Lite prompt for intelligent response
            if has_direct_pdf_result:
                # Enhanced prompt for direct PDF processing with full document context
                nova_prompt = f"""You are analyzing a complete medical document to answer a question. You have access to the FULL document content.

USER QUESTION: {question}

COMPLETE MEDICAL DOCUMENT CONTENT:
{context_text}

INSTRUCTIONS:
1. Answer ONLY the specific question asked - do not provide extra information
2. Focus on the exact information requested in the question
3. Quote specific details, names, values, dates that directly answer the question
4. Be concise and direct - avoid unnecessary background information
5. If the question asks for one thing, provide only that thing
6. IMPORTANT: Use plain text formatting only - no markdown symbols like *, #, or other special characters
7. Use simple formatting like dashes, colons, and line breaks for clarity

Provide a focused answer that directly addresses only what was asked, using the document content."""
            else:
                # Standard prompt for other search results
                nova_prompt = f"""You are analyzing specific medical document content to answer a question. Focus ONLY on what is actually written in the provided document sections.

USER QUESTION: {question}

ACTUAL DOCUMENT CONTENT:
{context_text}

INSTRUCTIONS:
1. **Answer based ONLY on the actual document content provided above**
2. **Quote specific information, names, values, and details from the documents**
3. **If the answer isn't in the document content, say so clearly**
4. **Be specific - mention actual patient names, test results, medications, dates if present**
5. **Do NOT provide general medical advice - only analyze what's in these documents**

Provide a direct answer that references the specific information found in the document content."""

            # Get Nova Lite response (use more tokens for comprehensive direct PDF analysis)
            max_tokens = 1500 if has_direct_pdf_result else 600
            logger.info(f"🤖 Using {max_tokens} max tokens for {'comprehensive direct PDF' if has_direct_pdf_result else 'standard'} response")
            ai_response = self.aws_utils.safe_bedrock_call(nova_prompt, max_tokens=max_tokens)

            # Compile final response
            response_parts = []
            response_parts.append("## 📋 ANSWER")
            response_parts.append(ai_response)

            # Add source information (ensure doc_ids are strings for set compatibility)
            unique_docs = set(str(result['doc_id']) for result in search_results if result.get('doc_id'))
            response_parts.append(f"\n## 📄 SOURCES")
            response_parts.append(f"• **Documents searched:** {len(unique_docs)}")
            response_parts.append(f"• **Relevant sections found:** {len(search_results)}")

            # Document list (ensure filenames are strings)
            doc_files = set(str(result['filename']) for result in search_results if result.get('filename'))
            response_parts.append(f"• **Source files:** {', '.join(doc_files)}")

            final_response = "\n".join(response_parts)

            logger.info(f"✅ Nova Lite query processed successfully: {len(search_results)} results")
            return final_response

        except Exception as e:
            logger.error(f"❌ Nova Lite query processing failed: {e}")
            return f"❌ **Error processing query:** {str(e)}\n\nPlease try again or contact support if the issue persists."
    
    def _create_basic_response(self, question: str, search_results: List[Dict]) -> str:
        """
        Create basic response when advanced AI generation is not available
        """
        response_parts = []
        
        # Group results by document
        doc_groups = {}
        for result in search_results:
            doc_id = result['doc_id']
            if doc_id not in doc_groups:
                doc_groups[doc_id] = []
            doc_groups[doc_id].append(result)
        
        # Create response from top results
        response_parts.append(f"Based on analysis of {len(doc_groups)} document(s), here are the most relevant findings:")
        response_parts.append("")
        
        for i, result in enumerate(search_results[:3]):  # Top 3 results
            # Handle both numeric and string similarity scores
            sim_score = result['similarity_score']
            if isinstance(sim_score, str):
                relevance = sim_score
            else:
                relevance = f"{sim_score:.1%}"
            filename = result['filename']
            content_preview = result['content'][:200]
            
            response_parts.append(f"**Result {i+1}** (Relevance: {relevance}) from {filename}:")
            response_parts.append(f"{content_preview}...")
            response_parts.append("")
        
        return "\n".join(response_parts)
    
    def get_system_status(self) -> str:
        """
        Get comprehensive system status for monitoring
        """
        try:
            # Get component statuses
            aws_cost_summary = self.aws_config.get_cost_summary()
            vector_stats = self.vector_search.get_system_statistics()
            
            status_parts = []
            
            # Header
            status_parts.append("# 🔧 MEDICAL DOCUMENT SYSTEM STATUS")
            status_parts.append(f"**Status Check:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            status_parts.append("")
            
            # Session Statistics
            status_parts.append("## 📊 SESSION STATISTICS")
            status_parts.append(f"• **Documents Processed:** {self.session_stats['documents_processed']}")
            status_parts.append(f"• **Queries Answered:** {self.session_stats['queries_answered']}")
            status_parts.append(f"• **Session Duration:** {self._calculate_session_duration()}")
            status_parts.append("")
            
            # AWS Status
            status_parts.append("## ☁️ AWS SERVICES STATUS")
            status_parts.append(f"• **Region:** {aws_cost_summary.get('aws_region', 'Unknown')}")
            status_parts.append(f"• **Estimated Cost:** ${aws_cost_summary.get('total_estimated_cost', 0)}")
            status_parts.append(f"• **Remaining Budget:** ${aws_cost_summary.get('remaining_budget', 0)}")
            status_parts.append("")
            
            # Vector Database Status
            if 'vector_database' in vector_stats:
                vdb_stats = vector_stats['vector_database']
                status_parts.append("## 🗄️ VECTOR DATABASE STATUS")
                status_parts.append(f"• **Total Documents:** {vdb_stats.get('total_documents', 0)}")
                status_parts.append(f"• **Total Chunks:** {vdb_stats.get('total_chunks', 0)}")
                status_parts.append(f"• **Average Chunks/Doc:** {vdb_stats.get('average_chunks_per_doc', 0)}")
                status_parts.append("")
            
            # Component Status
            status_parts.append("## ⚙️ COMPONENT STATUS")
            components = [
                ("AWS Infrastructure", "✅ Operational"),
                ("Document Processor", "✅ Operational"), 
                ("Vector Search", "✅ Operational"),
                ("Vision Processor", "✅ Operational"),
                ("Web Interface", "✅ Operational")
            ]
            
            for component, status in components:
                status_parts.append(f"• **{component}:** {status}")
            
            status_parts.append("")
            status_parts.append("🎉 **All systems operational and ready for medical document analysis!**")
            
            return "\n".join(status_parts)
            
        except Exception as e:
            logger.error(f"❌ Status check failed: {e}")
            return f"❌ **System Status Error:** {str(e)}"
    
    def _calculate_session_duration(self) -> str:
        """Calculate session duration in human-readable format"""
        try:
            start_time = datetime.fromisoformat(self.session_stats['session_start'])
            duration = datetime.now() - start_time
            
            hours = duration.seconds // 3600
            minutes = (duration.seconds % 3600) // 60
            
            if hours > 0:
                return f"{hours}h {minutes}m"
            else:
                return f"{minutes}m"
                
        except Exception:
            return "Unknown"
    
    def create_demo_scenarios(self) -> List[Tuple[str, str]]:
        """
        Create demo scenarios for presentations
        Returns list of (question, expected_response_type) tuples
        """
        scenarios = [
            # Basic medical queries
            ("What medications are mentioned in this document?", "medication_list"),
            ("What are the patient's primary symptoms?", "symptom_analysis"),
            ("What treatment recommendations are provided?", "treatment_plan"),
            
            # Multi-document analysis
            ("Compare medication effectiveness across all patients", "cross_document_comparison"),
            ("Which patients show the best treatment outcomes?", "outcome_analysis"),
            ("What patterns emerge in diabetes management across cases?", "pattern_analysis"),
            
            # Safety and alerts
            ("Are there any drug interactions or safety concerns?", "safety_analysis"),
            ("What follow-up care is recommended across all documents?", "followup_recommendations"),
            
            # Visual content analysis
            ("What trends are shown in any lab result charts?", "chart_analysis"),
            ("Describe any changes in vital signs over time", "temporal_analysis"),
            
            # Advanced insights
            ("What medical insights can be drawn from this patient population?", "population_insights"),
            ("Identify risk factors that appear across multiple patients", "risk_assessment")
        ]
        
        return scenarios

def create_gradio_interface(app: MedicalDocumentApp):
    """
    Create the complete Gradio interface
    Professional medical document analysis system with external CSS
    """

    # Load external CSS
    try:
        with open("style.css", "r") as f:
            custom_css = f.read()
    except FileNotFoundError:
        # Fallback CSS if style.css is not found
        custom_css = """
        .gradio-container {
            max-width: 1400px !important;
            margin: auto;
        }
        """

    with gr.Blocks(
        title="🏥 Medical Document Query System",
        theme=gr.themes.Base(),
        css=custom_css
    ) as demo:
        
        # Header with clean design
        gr.HTML("""
        <div class="header">
            <div class="hero">
                <h1>👨🏻‍⚕️ Doct.Bot 🧪</h1>
                <div class="features">
                    <div class="feature">📋 Multi-PDF Intelligence</div>
                    <div class="feature">📊 Chart Analysis</div>
                    <div class="feature">📁 Cross-Document Patterns</div>
                    <div class="feature">🚨 Safety Alerts</div>
                </div>
            </div>
        </div>
        """)
        
        with gr.Tab("📄 Document Analysis & Query") as main_tab:
            # Results display first (like myDemo.py)
            results_output = gr.Markdown(
                value="Ready to analyze your medical documents! Upload PDFs and start asking questions.",
                label="Multi-Document Analysis Results",
                elem_id="results_output"
            )
            processing_status = gr.Textbox(
                label="Processing Status",
                lines=2,
                visible=False
            )

            gr.HTML('<div class="section-header"><h3>🎯 Upload & Process Medical Documents</h3></div>')
            
            with gr.Row():
                with gr.Column(scale=1):
                    pdf_input = gr.File(
                        file_types=[".pdf"],
                        file_count="multiple",
                        label="📁 Upload Medical PDF Documents",
                        height=140
                    )
                    process_btn = gr.Button(
                        "📂 Process Documents",
                        variant="primary",
                        size="lg"
                    )

                with gr.Column(scale=1):
                    question_input = gr.Textbox(
                        label="Enter your medical question",
                        placeholder="e.g., 'What medications show the best outcomes?' or 'Compare patient symptoms across all documents'",
                        lines=4
                    )
                    query_btn = gr.Button(
                        "🚀 Search & Analyze",
                        variant="primary",
                        size="lg"
                    )

            # Advanced options
            with gr.Row():
                with gr.Column():
                    k_slider = gr.Slider(
                        minimum=3,
                        maximum=15,
                        value=7,
                        step=1,
                        label="🎯 Search depth (sections to analyze)"
                    )
                with gr.Column():
                    pattern_analysis_checkbox = gr.Checkbox(
                        label="📊 Enable cross-document pattern analysis",
                        value=True
                    )
        
        with gr.Tab("📊 System Dashboard") as dashboard_tab:
            gr.Markdown("### 🔧 System Health & Performance Monitoring")
            
            with gr.Row():
                status_btn = gr.Button("🔍 Check System Status", variant="secondary")
                refresh_btn = gr.Button("🔄 Refresh Statistics", variant="secondary")
            
            system_status_output = gr.Textbox(
                label="System Status Report",
                lines=20,
                show_copy_button=True
            )
            
            # Cost monitoring
            gr.Markdown("### 💰 Budget & Cost Tracking")
            cost_display = gr.Textbox(
                label="AWS Cost Summary",
                lines=5,
                show_copy_button=True
            )
        
        with gr.Tab("🎪 Demo Scenarios") as demo_tab:
            gr.Markdown("### 💡 Demo Questions & Use Cases")
            
            demo_scenarios = app.create_demo_scenarios()
            
            gr.Markdown("""
            **Single Document Analysis:**
            • "What medications are prescribed in this document?"
            • "What are the patient's symptoms and vital signs?"
            • "What lab results and test findings are mentioned?"
            • "What treatment plan is recommended?"
            
            **Multi-Document Intelligence:**
            • "Compare treatment outcomes between different patients"
            • "Which medications appear most frequently across all cases?"
            • "What patterns emerge in diabetes management strategies?"
            • "Find patients with similar symptom presentations"
            
            **Safety & Risk Analysis:**
            • "Are there any drug interactions detected?"
            • "What safety alerts should healthcare providers know about?"
            • "Identify patients at highest risk for complications"
            
            **Chart & Visual Analysis:**
            • "What trends are shown in the blood glucose monitoring charts?"
            • "Describe changes in vital signs over time"
            • "What do the lab result graphs reveal about treatment effectiveness?"
            
            **Population Health Insights:**
            • "What medical insights emerge from this patient population?"
            • "Compare medication adherence patterns across patients"
            • "Identify factors associated with successful treatment outcomes"
            """)
            
            # Quick demo buttons
            gr.Markdown("### 🚀 Quick Demo Queries")
            
            demo_buttons_row1 = gr.Row()
            with demo_buttons_row1:
                demo_btn1 = gr.Button("💊 Medications Analysis", size="sm")
                demo_btn2 = gr.Button("📊 Cross-Document Patterns", size="sm") 
                demo_btn3 = gr.Button("🚨 Safety Alerts", size="sm")
                demo_btn4 = gr.Button("📈 Chart Analysis", size="sm")
        
        with gr.Tab("ℹ️ About & Help") as help_tab:
            gr.Markdown("""
            ## 🏥 Medical Document Query System
            
            ### **What This System Does**
            
            This AI-powered platform transforms static medical PDFs into an intelligent, searchable knowledge base using advanced AWS AI services.
            
            ### **Key Features**
            
            **🔍 Intelligent Document Processing**
            • Advanced OCR with Amazon Textract for both digital and scanned documents
            • Medical entity extraction with Amazon Comprehend Medical
            • AI vision analysis of charts, graphs, and medical images
            
            **🧠 Smart Search & Analysis**
            • Semantic vector search across multiple documents simultaneously
            • Cross-document pattern analysis and medical insights
            • Drug interaction detection and safety alert system
            
            **👩‍⚕️ Healthcare-Optimized**
            • Built specifically for medical terminology and healthcare workflows
            • HIPAA-compliant processing with privacy protection
            • Professional medical reporting with source attribution
            
            ### **AWS AI Services Used**
            • **Amazon Bedrock (Nova Lite)** - Advanced multimodal AI for text extraction, vision analysis, and medical reasoning
            • **Amazon Kendra** - Enterprise search service for semantic document indexing and retrieval
            • **Integrated AI Pipeline** - Streamlined processing with fewer dependencies for faster, more reliable results
            
            ### **How to Use**
            
            1. **Upload Documents**: Add one or more medical PDF documents
            2. **Process**: Click "Process Document" to extract and analyze content
            3. **Query**: Ask questions in natural language about your documents
            4. **Analyze**: Get comprehensive answers with source attribution
            5. **Insights**: Discover patterns across multiple documents
            
            ### **Example Workflows**
            
            **Individual Patient Analysis**
            Upload patient records → Ask about medications, symptoms, treatment plans
            
            **Population Health Study**
            Upload multiple patient files → Identify patterns, compare outcomes, assess risks
            
            **Clinical Decision Support**
            Upload guidelines and patient data → Get evidence-based recommendations
            
            ### **System Requirements**
            • Internet connection for AWS AI services
            • PDF documents in readable format
            • Modern web browser with JavaScript enabled
            
            ### **Privacy & Security**
            • Documents processed in secure AWS environment
            • No persistent storage of patient data
            • PHI detection and protection
            • Audit logging for compliance
            
            ---
            
            **Built by:** Team Medical AI for Healthcare Innovation Hackathon
            **Powered by:** AWS AI Services & Advanced Machine Learning
            """)
        
        # Event handlers
        def process_documents_with_progress(pdf_files):
            if pdf_files is None or len(pdf_files) == 0:
                return "❌ Please upload at least one PDF file first.", ""

            try:
                # Show processing status
                processing_status.visible = True

                def progress_callback(progress, message):
                    return f"⚙️ Processing... {int(progress*100)}% - {message}"

                # Handle single or multiple files
                if len(pdf_files) == 1:
                    # Single file processing (backward compatibility)
                    result = app.process_document_complete(pdf_files[0], progress_callback)

                    if result['success']:
                        summary = result['comprehensive_summary']
                        status_msg = f"✅ Successfully processed: {result['filename']}"
                    else:
                        summary = f"❌ Processing failed: {result.get('user_message', result.get('error', 'Unknown error'))}"
                        status_msg = "❌ Processing failed"
                else:
                    # Multiple files processing
                    result = app.process_multiple_documents(pdf_files, progress_callback)

                    if result['success']:
                        summary = result['unified_summary']
                        status_msg = f"✅ Successfully processed {result['successful_files']}/{result['total_files']} documents"
                        if result['failed_files'] > 0:
                            status_msg += f" ({result['failed_files']} failed)"
                    else:
                        summary = f"❌ Multi-document processing failed: {result.get('user_message', result.get('error', 'Unknown error'))}"
                        status_msg = "❌ Multi-document processing failed"

                return summary, status_msg

            except Exception as e:
                error_msg = f"❌ An error occurred: {str(e)}"
                return error_msg, error_msg
        
        def query_with_settings(question, k_value, pattern_analysis):
            if not question.strip():
                return "❓ Please enter a question about your medical documents."
            
            return app.query_documents(
                question=question,
                k_value=k_value,
                use_pattern_analysis=pattern_analysis
            )
        
        def get_system_status():
            return app.get_system_status()
        
        def get_cost_summary():
            try:
                cost_summary = app.aws_config.get_cost_summary()
                return f"""💰 **AWS Cost Summary**

**Total Estimated Cost:** ${cost_summary.get('total_estimated_cost', 0)}
**Remaining Budget:** ${cost_summary.get('remaining_budget', 100)}
**Services Used:** {', '.join(cost_summary.get('services_used', []))}
**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

💡 **Cost Breakdown:**
• Document processing and OCR
• AI reasoning and analysis  
• Medical entity extraction
• Vector search operations
"""
            except Exception as e:
                return f"❌ Cost summary error: {str(e)}"
        
        # Demo button handlers
        def set_demo_query(query_text):
            return query_text
        
        # Wire up event handlers
        process_btn.click(
            fn=process_documents_with_progress,
            inputs=[pdf_input],
            outputs=[results_output, processing_status]
        )
        
        query_btn.click(
            fn=query_with_settings,
            inputs=[question_input, k_slider, pattern_analysis_checkbox],
            outputs=[results_output]
        )
        
        status_btn.click(
            fn=get_system_status,
            outputs=[system_status_output]
        )
        
        refresh_btn.click(
            fn=get_cost_summary,
            outputs=[cost_display]
        )
        
        # Demo button events
        demo_btn1.click(
            fn=lambda: "What medications are mentioned across all uploaded documents?",
            outputs=[question_input]
        )
        
        demo_btn2.click(
            fn=lambda: "Compare treatment patterns and outcomes between different patients",
            outputs=[question_input]
        )
        
        demo_btn3.click(
            fn=lambda: "Are there any drug interactions or safety alerts I should know about?",
            outputs=[question_input]
        )
        
        demo_btn4.click(
            fn=lambda: "What trends and patterns are shown in any charts or graphs?",
            outputs=[question_input]
        )
        
        # Example interactions for multi-document analysis
        gr.Examples(
            examples=[
                [None, "What medications are prescribed across all documents?", 5, True],
                [None, "Compare symptoms and outcomes between patients", 8, True],
                [None, "What safety alerts or drug interactions are present?", 6, False],
                [None, "Analyze trends shown in any medical charts or graphs", 7, True],
                [None, "What follow-up care recommendations are mentioned?", 5, False],
                [None, "Which patients have the highest medication compliance?", 6, True],
                [None, "Identify common conditions across multiple patients", 7, True]
            ],
            inputs=[pdf_input, question_input, k_slider, pattern_analysis_checkbox]
        )
    
    return demo

def main():
    """
    Main function to launch the medical document application
    """
    print("Starting Medical Document Query System...")
    print("=" * 60)

    try:
        # Initialize the application
        app = MedicalDocumentApp()

        # Create Gradio interface
        demo = create_gradio_interface(app)

        print("Application initialized successfully!")
        print("Launching web interface...")
        print("Access the application at the URL shown below")
        
        # Launch the application
        demo.launch(
            share=True,              # Create shareable public URL for demo
            server_name="0.0.0.0",   # Allow external access
            server_port=7864,        # Alternative port (changed from 7863)
            show_error=True,         # Show detailed errors
            favicon_path=None,       # Custom favicon (optional)
            ssl_verify=False         # For development
        )
        
    except Exception as e:
        print(f"Application startup failed: {e}")
        print("\nTroubleshooting checklist:")
        print("   1. All team components available (Persons 1-4)")
        print("   2. AWS credentials configured: aws configure")
        print("   3. Required Python packages installed")
        print("   4. Internet connection for AWS services")
        print("   5. Sufficient system memory for ML models")
        
        logger.error(f"Application startup failed: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()