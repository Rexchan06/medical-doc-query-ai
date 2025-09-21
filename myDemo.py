#!/usr/bin/env python3
"""
Gradio UI Demo with External CSS (No Inline Styles)
"""

import gradio as gr
from datetime import datetime


def mock_process_multiple_documents(pdf_files):
    """Mock document processing for multiple files - processes one by one"""
    if pdf_files is None or len(pdf_files) == 0:
        return "Please upload at least one PDF file first.", ""

    all_results = []
    processing_status = []

    for i, pdf_file in enumerate(pdf_files, 1):
        processing_status.append(f"Processing file {i}/{len(pdf_files)}: {pdf_file.name}")
        result = f"""
# DOCUMENT {i} ANALYSIS: {pdf_file.name}
**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## EXECUTIVE SUMMARY
• **Document Type:** Medical Report
• **Medications Identified:** {2 + i}
• **Safety Alerts:** {1 if i % 2 == 0 else 0}
• **Pages with Visual Content:** {i}
• **Charts/Graphs Analyzed:** {i % 3}

## KEY FINDINGS
### Medications Found:
• Primary: Medication-{i}A (Dosage: {10 * i}mg)  
• Secondary: Medication-{i}B (Dosage: {5 * i}mg)

### Clinical Notes:
Patient shows response to treatment protocol {i}.  
{f"⚠️ SAFETY ALERT: Drug interaction detected in document {i}" if i % 2 == 0 else "✅ No safety concerns identified"}

---
"""
        all_results.append(result)

    final_result = f"""# MULTI-DOCUMENT ANALYSIS COMPLETE
**Total Documents Processed:** {len(pdf_files)}  
**Processing Time:** {len(pdf_files) * 2} seconds (simulated)

## CROSS-DOCUMENT SUMMARY
• **Total Medications Found:** {sum(2 + i for i in range(1, len(pdf_files) + 1))}  
• **Documents with Safety Alerts:** {len([i for i in range(1, len(pdf_files) + 1) if i % 2 == 0])}  
• **Total Charts Analyzed:** {sum(i % 3 for i in range(1, len(pdf_files) + 1))}  

{'=' * 60}

{"".join(all_results)}

## PROCESSING SUMMARY
Files processed in sequence:  
{chr(10).join(f"• {pdf_file.name} - ✅ Completed" for pdf_file in pdf_files)}
"""
    status_message = f"✅ Successfully processed {len(pdf_files)} documents sequentially"
    return final_result, status_message


def mock_query(question, k_value, pattern_analysis):
    if not question.strip():
        return "Please enter a question about your medical documents."
    return f"""## ANSWER
Based on your question: "{question}"

This is a demo response showing how the AI would analyze your medical documents.

**Key findings:**  
• Found relevant information across 2 documents  
• Identified 3 related medications  
• Cross-referenced with safety protocols  

## CROSS-DOCUMENT PATTERN ANALYSIS
**Documents Analyzed:** 2  
**Most Common Medications:** Metformin, Lisinopril, Aspirin  
**Most Common Conditions:** Diabetes, Hypertension  

### AI Medical Insights:
Pattern analysis suggests consistent medication adherence across patients with similar conditions.

## SOURCES
• **Documents searched:** 2  
• **Relevant sections found:** {k_value}  
• **Visual content included:** 1 chart/graph descriptions  
• **Source files:** medical_report_1.pdf, patient_history_2.pdf
"""


def mock_system_status():
    return f"""# MEDICAL DOCUMENT SYSTEM STATUS
**Status Check:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## SESSION STATISTICS
• **Documents Processed:** 2  
• **Queries Answered:** 5  
• **Safety Alerts Generated:** 1  
• **Session Duration:** 15m  

## AWS SERVICES STATUS (DEMO MODE)
• **Region:** us-east-1  
• **Estimated Cost:** $0.25  
• **Remaining Budget:** $99.75  

## VECTOR DATABASE STATUS
• **Total Documents:** 2  
• **Total Chunks:** 47  
• **Average Chunks/Doc:** 23.5  

## COMPONENT STATUS
• AWS Infrastructure: ✅ Operational  
• Document Processor: ✅ Operational  
• Vector Search: ✅ Operational  
• Vision Processor: ✅ Operational  
• Web Interface: ✅ Operational
"""


def create_demo_interface():

    with open("style.css", "r") as f:
        custom_css = f.read()

    with gr.Blocks(
        title="Medical Document Query System (Demo)",
        theme=gr.themes.Base(),
        css=custom_css,   # external stylesheet
    ) as demo:

        # Header (no inline styles, just class)
        gr.HTML("""
        <div class="header">
            <div class="hero">
                <h1>👨🏻‍⚕️ Doct.Bot 🧪</h1>
                <div class="features">
                    <div class="feature">📋PDF Intelligence</div>
                    <div class="feature">📊Chart Analysis</div>
                    <div class="feature">📑Cross-Document Patterns</div>
                </div>
            </div>
        </div>
        """)

        with gr.Tab("Document Analysis & Query"):

            results_output = gr.Markdown(
                value="Document Analizer? I'm Here!",
                label="Multi-Document Analysis Results",
                elem_id="results_output"
            )
            processing_status = gr.Textbox(label="Processing Status", lines=2)

            with gr.Row():   
                with gr.Column(scale=1):
                    pdf_input = gr.File(file_types=[".pdf"], file_count="multiple", label="", height=140)
                    process_btn = gr.Button("📂 Upload", variant="primary", size="lg")

                with gr.Column(scale=1):
                    question_input = gr.Textbox(
                        label="Enter your medical question",
                        placeholder="e.g., 'What medications show the best outcomes?'",
                        lines=4
                    )
                    query_btn = gr.Button("🚀 Search & Analyze", variant="primary", size="lg")

        with gr.Tab("System Dashboard"):
            gr.HTML("<h3 class='tab-title'>System Health & Performance Monitoring (Demo)</h3>")
            status_btn = gr.Button("Check System Status (Demo)", variant="secondary", size="lg")
            system_status_output = gr.Textbox(label="System Status Report", lines=25, show_copy_button=True)

        with gr.Tab("Demo Scenarios"):
            gr.HTML("<h3 class='tab-title'>Demo Questions & Use Cases</h3>")
            gr.HTML("<h4 class='sub-title'>Quick Demo Queries</h4>")

            with gr.Row():
                demo_btn1 = gr.Button("Medications Analysis", variant="secondary")
                demo_btn2 = gr.Button("Cross-Document Patterns", variant="secondary")
                demo_btn3 = gr.Button("Safety Alerts", variant="secondary")
                demo_btn4 = gr.Button("Chart Analysis", variant="secondary")

        # Event handlers
        process_btn.click(fn=mock_process_multiple_documents, inputs=[pdf_input], outputs=[results_output, processing_status])
        query_btn.click(fn=mock_query, inputs=[question_input], outputs=[results_output])
        status_btn.click(fn=mock_system_status, outputs=[system_status_output])

        demo_btn1.click(fn=lambda: "What medications are mentioned across all uploaded documents?", outputs=[question_input])
        demo_btn2.click(fn=lambda: "Compare treatment patterns and outcomes between different patients", outputs=[question_input])
        demo_btn3.click(fn=lambda: "Are there any drug interactions or safety alerts I should know about?", outputs=[question_input])
        demo_btn4.click(fn=lambda: "What trends and patterns are shown in any charts or graphs?", outputs=[question_input])

    return demo


if __name__ == "__main__":
    print("Launching Gradio App with External CSS...")
    demo = create_demo_interface()
    demo.launch(share=True, server_name="0.0.0.0", inbrowser=True)
