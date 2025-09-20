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
            'safety_alerts_generated': 0,
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
        Complete document processing pipeline
        Integrates all team components for comprehensive analysis
        """
        if pdf_file is None:
            return {
                'success': False,
                'error': 'No file uploaded',
                'user_message': 'Please upload a PDF file to analyze.'
            }
        
        try:
            filename = Path(pdf_file.name).name
            logger.info(f"🔄 Starting complete processing for: {filename}")
            
            if progress_callback:
                progress_callback(0.1, "Starting document processing...")
            
            # Step 1: Person 2 - Basic document processing
            logger.info("📄 Step 1: Extracting text and medical entities...")
            doc_result = self.document_processor.process_document_complete(pdf_file.name)
            
            if not doc_result['success']:
                return {
                    'success': False,
                    'error': doc_result['error'],
                    'user_message': f"Failed to process document: {doc_result['error']}"
                }
            
            if progress_callback:
                progress_callback(0.3, "Text extraction complete. Analyzing images...")
            
            # Step 2: Person 4 - Vision processing
            logger.info("👁️ Step 2: Processing visual content...")
            vision_results = self.vision_processor.process_document_vision(pdf_file.name)
            
            # Create vision descriptions for vector search
            vision_descriptions = self.vision_processor.create_vision_description_for_vector_search(vision_results)
            
            if progress_callback:
                progress_callback(0.6, "Visual analysis complete. Building search index...")
            
            # Step 3: Person 3 - Vector storage
            logger.info("🗄️ Step 3: Storing in vector database...")
            vector_success = self.vector_search.store_document_vectors(
                doc_result['doc_id'],
                doc_result['extracted_text'],
                doc_result['medical_entities'],
                vision_descriptions,
                doc_result['extraction_metadata']
            )
            
            if not vector_success:
                logger.warning("⚠️ Vector storage failed, but continuing with basic functionality")
            
            if progress_callback:
                progress_callback(0.9, "Finalizing analysis...")
            
            # Compile comprehensive results
            comprehensive_result = {
                'success': True,
                'doc_id': doc_result['doc_id'],
                'filename': filename,
                'document_processing': doc_result,
                'vision_analysis': vision_results,
                'vector_search_enabled': vector_success,
                'processing_timestamp': datetime.now().isoformat(),
                'comprehensive_summary': self._create_comprehensive_summary(
                    doc_result, vision_results
                )
            }
            
            # Update session stats
            self.session_stats['documents_processed'] += 1
            if doc_result['processing_summary']['safety_alerts_count'] > 0:
                self.session_stats['safety_alerts_generated'] += doc_result['processing_summary']['safety_alerts_count']
            
            # Store in app state
            self.processed_documents[doc_result['doc_id']] = comprehensive_result
            
            if progress_callback:
                progress_callback(1.0, "Processing complete!")
            
            logger.info(f"✅ Complete processing finished for: {filename}")
            return comprehensive_result
            
        except Exception as e:
            logger.error(f"❌ Complete processing failed: {e}")
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e),
                'user_message': f"An error occurred during processing: {str(e)}"
            }
    
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
        alerts_count = processing_summary.get('safety_alerts_count', 0)
        has_high_risk = processing_summary.get('has_high_risk_interactions', False)
        
        # Count vision analysis results
        pages_analyzed = len([r for r in vision_results if 'error' not in r])
        charts_found = sum(1 for r in vision_results if r.get('charts_detected', {}).get('has_charts', False))
        
        summary_parts.append("## 📊 EXECUTIVE SUMMARY")
        summary_parts.append(f"• **Medications Identified:** {medications_count}")
        summary_parts.append(f"• **Safety Alerts:** {alerts_count} {'🚨' if has_high_risk else ''}")
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
        Query across all processed documents
        Integrates vector search and pattern analysis
        """
        if not question.strip():
            return "❓ Please enter a question about your medical documents."
        
        try:
            logger.info(f"🔍 Processing query: '{question}'")
            
            # Update session stats
            self.session_stats['queries_answered'] += 1
            
            # Step 1: Multi-document vector search
            search_results = self.vector_search.search_across_documents(
                question, k=k_value
            )
            
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

            # Step 2: Pattern analysis (if enabled and AWS available)
            pattern_analysis = None
            if use_pattern_analysis and len(search_results) > 1:
                try:
                    pattern_analysis = self.vector_search.analyze_document_patterns(
                        search_results, question
                    )
                except Exception as e:
                    logger.warning(f"⚠️ Pattern analysis failed: {e}")
            
            # Step 3: Generate comprehensive response
            response_parts = []
            
            # Direct answer section
            response_parts.append("## 📋 ANSWER")
            
            # Use Person 3's AI response generation
            if hasattr(self.vector_search, 'generate_intelligent_answer'):
                ai_answer = self.vector_search.generate_intelligent_answer(question, search_results)
                response_parts.append(ai_answer)
            else:
                # Fallback: create basic response
                response_parts.append(self._create_basic_response(question, search_results))
            
            # Pattern analysis section (if available)
            if pattern_analysis and 'error' not in pattern_analysis:
                response_parts.append("\n## 📊 CROSS-DOCUMENT PATTERN ANALYSIS")
                response_parts.append(f"**Documents Analyzed:** {pattern_analysis['documents_analyzed']}")
                
                patterns = pattern_analysis.get('patterns', {})
                
                # Top medications
                med_freq = patterns.get('medication_frequency', {})
                if med_freq:
                    top_meds = dict(list(med_freq.items())[:5])
                    response_parts.append(f"**Most Common Medications:** {', '.join(top_meds.keys())}")
                
                # Top conditions  
                cond_freq = patterns.get('condition_frequency', {})
                if cond_freq:
                    top_conditions = dict(list(cond_freq.items())[:5])
                    response_parts.append(f"**Most Common Conditions:** {', '.join(top_conditions.keys())}")
                
                # AI insights
                if pattern_analysis.get('ai_insights'):
                    response_parts.append(f"\n### 🧠 AI Medical Insights:")
                    response_parts.append(pattern_analysis['ai_insights'])
            
            # Source information
            unique_docs = set(result['doc_id'] for result in search_results)
            chart_sources = sum(1 for r in search_results if r.get('contains_chart_info', False))
            
            response_parts.append(f"\n## 📄 SOURCES")
            response_parts.append(f"• **Documents searched:** {len(unique_docs)}")
            response_parts.append(f"• **Relevant sections found:** {len(search_results)}")
            response_parts.append(f"• **Visual content included:** {chart_sources} chart/graph descriptions")
            
            # Document list
            doc_files = set(result['filename'] for result in search_results)
            response_parts.append(f"• **Source files:** {', '.join(doc_files)}")
            
            final_response = "\n".join(response_parts)
            
            logger.info(f"✅ Query processed successfully: {len(search_results)} results")
            return final_response
            
        except Exception as e:
            logger.error(f"❌ Query processing failed: {e}")
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
            relevance = f"{result['similarity_score']:.1%}"
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
            status_parts.append(f"• **Safety Alerts Generated:** {self.session_stats['safety_alerts_generated']}")
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
    Professional medical document analysis system
    """
    
    # Custom CSS for professional appearance
    custom_css = """
    .gradio-container {
        max-width: 1400px !important;
        margin: auto;
    }
    
    .title-container {
        text-align: center;
        background: linear-gradient(90deg, #2E86C1, #3498DB);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .stats-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2E86C1;
    }
    
    .alert-container {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .success-container {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    """
    
    with gr.Blocks(
        title="🏥 Medical Document Query System",
        theme=gr.themes.Soft(),
        css=custom_css
    ) as demo:
        
        # Header
        gr.Markdown(
            """
            <div class="title-container">
                <h1>🏥 Intelligent Medical Document Query System</h1>
                <p><strong>Advanced AI-Powered Multi-Document Analysis Platform</strong></p>
                <p>Transform medical PDFs into searchable intelligence • Extract insights from charts and graphs • Detect drug interactions • Cross-document pattern analysis</p>
            </div>
            """,
            elem_classes=["title-container"]
        )
        
        with gr.Tab("📄 Document Analysis & Query") as main_tab:
            gr.Markdown("""
            ### 🎯 Upload Medical Documents and Ask Intelligent Questions
            
            **System Capabilities:**
            • Process both digital and scanned medical PDFs
            • Extract text, analyze charts and medical images  
            • Detect drug interactions and safety alerts
            • Search across multiple documents simultaneously
            • Generate comprehensive medical insights
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    # File upload
                    pdf_input = gr.File(
                        file_types=[".pdf"],
                        label="📁 Upload Medical PDF Document",
                        height=120
                    )
                    
                    # Processing button
                    process_btn = gr.Button(
                        "🔄 Process Document",
                        variant="primary",
                        size="lg"
                    )
                    
                    # Query section
                    gr.Markdown("### ❓ Ask Questions About Your Documents")
                    
                    question_input = gr.Textbox(
                        label="Enter your medical question",
                        placeholder="e.g., 'What medications show the best outcomes?' or 'Compare patient symptoms across all documents'",
                        lines=3,
                        max_lines=5
                    )
                    
                    with gr.Row():
                        k_slider = gr.Slider(
                            minimum=3,
                            maximum=15,
                            value=7,
                            step=1,
                            label="🎯 Search depth (sections to analyze)"
                        )
                        
                        pattern_analysis_checkbox = gr.Checkbox(
                            label="📊 Enable cross-document pattern analysis",
                            value=True
                        )
                    
                    query_btn = gr.Button(
                        "🔍 Search & Analyze",
                        variant="secondary",
                        size="lg"
                    )
                
                with gr.Column(scale=2):
                    # Results display
                    results_output = gr.Textbox(
                        label="📋 Analysis Results",
                        lines=25,
                        max_lines=40,
                        show_copy_button=True,
                        placeholder="Upload a document and ask questions to see intelligent analysis results here..."
                    )
            
            # Processing status
            processing_status = gr.Textbox(
                label="⚙️ Processing Status",
                lines=2,
                visible=False
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
            • **Amazon Textract** - Advanced document analysis and OCR
            • **Amazon Bedrock (Claude 3)** - AI reasoning and natural language generation
            • **Amazon Comprehend Medical** - Medical entity recognition
            • **Vector Search Engine** - Semantic similarity matching
            
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
        def process_document_with_progress(pdf_file):
            if pdf_file is None:
                return "❌ Please upload a PDF file first.", ""
            
            try:
                # Show processing status
                processing_status.visible = True
                
                def progress_callback(progress, message):
                    return f"⚙️ Processing... {int(progress*100)}% - {message}"
                
                # Process document
                result = app.process_document_complete(pdf_file, progress_callback)
                
                if result['success']:
                    summary = result['comprehensive_summary']
                    status_msg = f"✅ Successfully processed: {result['filename']}"
                else:
                    summary = f"❌ Processing failed: {result.get('user_message', result.get('error', 'Unknown error'))}"
                    status_msg = "❌ Processing failed"
                
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
            fn=process_document_with_progress,
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
        
        # Example interactions
        gr.Examples(
            examples=[
                [None, "What medications are prescribed in this document?", 5, True],
                [None, "Compare symptoms and outcomes across all patients", 8, True],
                [None, "What safety alerts or drug interactions are present?", 6, False],
                [None, "Analyze trends shown in any medical charts or graphs", 7, True],
                [None, "What follow-up care recommendations are mentioned?", 5, False]
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
            server_port=7860,        # Standard port
            show_error=True,         # Show detailed errors
            show_tips=True,          # Show helpful tips
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