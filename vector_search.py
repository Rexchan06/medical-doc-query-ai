"""
PERSON 3: AI/ML Developer
Vector search, embeddings, and pattern analysis

RESPONSIBILITIES:
- Kendra index setup and management
- Text embedding generation and optimization
- Multi-document vector search implementation
- Cross-document pattern analysis
- Integration with Person 2's document processing

DELIVERABLES:
- vector_search.py (this file)
- Working vector search across multiple documents
- Pattern analysis across document collections
- Optimized embedding and retrieval system
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from collections import defaultdict, Counter
import hashlib
import json
import os
import base64

# Import Person 1's AWS configuration
try:
    from aws_config import AWSConfigManager, AWSUtilities, setup_aws_environment
except ImportError:
    print("❌ Error: aws_config.py not found. Make sure Person 1 has completed AWS setup.")
    exit(1)

# Note: Document processor is imported by the medical_app.py coordinator
# No direct import needed here to avoid circular dependencies

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalVectorSearch:
    """
    Advanced vector search system for medical documents
    Enables semantic search across multiple documents using AWS Kendra
    """
    
    def __init__(self):
        """Initialize the vector search system"""
        logger.info("🧠 Initializing Medical Vector Search System with AWS Kendra...")
        
        # Initialize AWS utilities
        try:
            self.aws_config, self.aws_utils = setup_aws_environment()
        except Exception as e:
            logger.error(f"❌ AWS utilities not available: {e}")
            raise
        
        # Initialize Kendra Index
        self._initialize_kendra_index()
        
        # Configuration
        self.chunk_size = 500
        self.chunk_overlap = 50
        self.max_chunks_per_doc = 50  # Prevent memory issues
        
        # Document tracking
        self.document_registry = {}
        self.search_history = []

        # In-memory cache for immediate search
        self.local_document_cache = {}  # {doc_id: {content, chunks, metadata}}
        self.local_search_enabled = True

        # Track original PDF files for direct processing fallback
        self.pdf_file_registry = {}  # {doc_id: pdf_file_path}

        logger.info("✅ Medical Vector Search System ready!")
    
    def _initialize_kendra_index(self):
        """Initialize the Kendra index for medical documents"""
        try:
            logger.info("🗄️ Setting up AWS Kendra index...")
            
            self.kendra_index_id = os.environ.get("KENDRA_INDEX_ID")
            if not self.kendra_index_id:
                # As a fallback for the hackathon, we can try to list indices
                kendra_client = self.aws_config.get_service_client('kendra')
                indices = kendra_client.list_indices().get('IndexConfigurationSummaryItems', [])
                if indices:
                    self.kendra_index_id = indices[0]['Id']
                    logger.warning(f"⚠️ KENDRA_INDEX_ID not set, using first available index: {self.kendra_index_id}")
                else:
                    # This is a critical error, as we can't proceed without an index
                    logger.error("❌ No Kendra index found. Please create one and set KENDRA_INDEX_ID.")
                    raise ValueError("Kendra index not configured")
            
            logger.info(f"✅ Kendra index configured: {self.kendra_index_id}")
            
        except Exception as e:
            logger.error(f"❌ Kendra initialization failed: {e}")
            raise
    
    def chunk_text(self, text: str, doc_id: str) -> List[Dict]:
        """
        Intelligent text chunking for medical documents
        Creates overlapping chunks while preserving medical context
        """
        if not text or not text.strip():
            return []
        
        sentences = text.split('.')
        chunks = []
        current_chunk = ""
        chunk_id = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            potential_chunk = current_chunk + ". " + sentence if current_chunk else sentence
            
            if len(potential_chunk.split()) <= self.chunk_size:
                current_chunk = potential_chunk
            else:
                if len(current_chunk.split()) > 20:
                    chunks.append({
                        'content': current_chunk,
                        'chunk_id': f"{doc_id}_chunk_{chunk_id}",
                        'doc_id': doc_id,
                    })
                    chunk_id += 1
                
                words = current_chunk.split()
                if len(words) > self.chunk_overlap:
                    overlap_text = " ".join(words[-self.chunk_overlap:])
                    current_chunk = overlap_text + ". " + sentence
                else:
                    current_chunk = sentence
        
        if current_chunk and len(current_chunk.split()) > 20:
            chunks.append({
                'content': current_chunk,
                'chunk_id': f"{doc_id}_chunk_{chunk_id}",
                'doc_id': doc_id,
            })
        
        logger.info(f"📑 Created {len(chunks)} chunks for document {doc_id}")
        return chunks[:self.max_chunks_per_doc]

    def store_document_vectors(self, doc_id: str, text_content: str,
                             medical_entities: Dict, chart_descriptions: List[str],
                             metadata: Dict, pdf_file_path: Optional[str] = None) -> bool:
        """
        Store document in Kendra index for multi-document search
        """
        try:
            logger.info(f"💾 Storing document {doc_id} in Kendra index...")

            full_content = text_content
            if chart_descriptions:
                full_content += "\n\n" + "\n".join(chart_descriptions)

            # DEBUG: Show what's being sent to chunking
            print(f"\n{'='*80}")
            print(f"DEBUG: KENDRA STORAGE INPUT")
            print(f"{'='*80}")
            print(f"Full content length: {len(full_content)} characters")
            print(f"Word count: {len(full_content.split())}")
            print(f"Chart descriptions: {len(chart_descriptions)} items")
            print(f"\nFIRST 1000 CHARS GOING TO KENDRA:")
            print("-" * 50)
            print(full_content[:1000])
            print("-" * 50)
            print(f"{'='*80}")

            text_chunks = self.chunk_text(full_content, doc_id)

            # DEBUG: Show chunking results
            print(f"\n{'='*80}")
            print(f"DEBUG: CHUNKING RESULTS")
            print(f"{'='*80}")
            print(f"Number of chunks created: {len(text_chunks)}")
            for i, chunk in enumerate(text_chunks[:3]):  # Show first 3 chunks
                print(f"\nCHUNK {i+1}:")
                print(f"  Length: {len(chunk['content'])} chars")
                print(f"  Preview: {chunk['content'][:200]}...")
            if len(text_chunks) > 3:
                print(f"\n... and {len(text_chunks) - 3} more chunks")
            print(f"{'='*80}")

            if not text_chunks:
                logger.warning(f"⚠️ No chunks created for document {doc_id}")
                return False
            
            documents_to_upload = []
            for i, chunk in enumerate(text_chunks):
                documents_to_upload.append({
                    'Id': chunk['chunk_id'],
                    'Title': f"{metadata.get('filename', 'unknown')} - Chunk {i+1}",
                    'Blob': chunk['content'].encode('utf-8'),
                    'ContentType': 'PLAIN_TEXT',
                    'Attributes': [
                        {'Key': 'doc_id', 'Value': {'StringValue': doc_id}},
                        {'Key': 'filename', 'Value': {'StringValue': metadata.get('filename', 'unknown')}},
                        {'Key': 'contains_chart_info', 'Value': {'StringValue': str(any(desc in chunk['content'] for desc in chart_descriptions))}},
                    ]
                })
            
            # DEBUG: Show what's being uploaded to Kendra
            print(f"\n{'='*80}")
            print(f"DEBUG: KENDRA UPLOAD")
            print(f"{'='*80}")
            print(f"Documents to upload: {len(documents_to_upload)}")
            for i, doc in enumerate(documents_to_upload[:2]):  # Show first 2 documents
                print(f"\nDOCUMENT {i+1} TO KENDRA:")
                print(f"  ID: {doc['Id']}")
                print(f"  Title: {doc['Title']}")
                print(f"  Content length: {len(doc['Blob'])} bytes")
                content_str = doc['Blob'].decode('utf-8')
                print(f"  Content preview: {content_str[:300]}...")
            print(f"{'='*80}")

            response = self.aws_utils.safe_kendra_batch_put_document(
                index_id=self.kendra_index_id,
                documents=documents_to_upload
            )
            
            # Store in local cache for immediate search (regardless of Kendra success)
            self.local_document_cache[doc_id] = {
                'full_content': full_content,
                'chunks': text_chunks,
                'metadata': metadata,
                'medical_entities': medical_entities,
                'timestamp': datetime.now().isoformat(),
                'filename': metadata.get('filename', 'unknown')
            }
            logger.info(f"✅ Stored document {doc_id} in local cache for immediate search")

            # Store PDF file path for direct processing fallback
            if pdf_file_path and os.path.exists(pdf_file_path):
                self.pdf_file_registry[doc_id] = pdf_file_path
                logger.info(f"📄 Registered PDF file for direct processing: {pdf_file_path}")

            if not response['success']:
                logger.warning(f"⚠️ Kendra storage failed for document {doc_id}: {response['error']}")
                logger.info("📋 Document available for immediate search via local cache")
                return True  # Still return True since local cache worked

            self.document_registry[doc_id] = {
                'filename': metadata.get('filename', 'unknown'),
                'chunks_stored': len(documents_to_upload),
                'timestamp': datetime.now().isoformat(),
                'medical_entities': medical_entities,
                'has_chart_data': len(chart_descriptions) > 0
            }

            logger.info(f"✅ Stored {len(documents_to_upload)} chunks for document {doc_id} in Kendra + local cache")
            return True
            
        except Exception as e:
            logger.error(f"❌ Kendra storage failed for document {doc_id}: {e}")
            return False

    def _search_local_cache(self, query: str, k: int = 10) -> List[Dict]:
        """
        Search through local document cache using Nova Lite for immediate results
        """
        if not self.local_document_cache:
            return []

        try:
            results = []

            for doc_id, doc_data in self.local_document_cache.items():
                # Use Nova Lite to analyze if the query matches this document's content
                content = doc_data['full_content']

                # Create a search prompt for Nova Lite
                search_prompt = f"""
Analyze this medical document content and determine if it contains information relevant to the query: "{query}"

Document content:
{content[:3000]}...

Query: {query}

If this document contains relevant information, provide:
1. A relevance score (0-100)
2. Key relevant excerpts (max 300 characters each)
3. Brief explanation of relevance

If not relevant, respond with: "NOT_RELEVANT"
"""

                # Call Nova Lite for semantic analysis
                response = self.aws_utils.safe_bedrock_call(search_prompt, max_tokens=300)

                if response and not response.startswith("Error:") and "NOT_RELEVANT" not in response:
                    # Extract relevance info and add to results
                    print(f"\nLOCAL CACHE MATCH FOUND:")
                    print(f"  Doc ID: {doc_id}")
                    print(f"  Filename: {doc_data['filename']}")
                    print(f"  Content length: {len(content)} chars")
                    print(f"  Content preview: {content[:300]}...")
                    print(f"  AI Analysis: {response[:200]}...")

                    results.append({
                        'doc_id': doc_id,
                        'filename': doc_data['filename'],
                        'content': content[:2000],  # Increase content size for better context
                        'relevance_analysis': response,
                        'timestamp': doc_data['timestamp'],
                        'source': 'local_cache',
                        'similarity_score': 'High'  # Add similarity_score for consistency
                    })

            # Sort by relevance (could be improved with actual scoring)
            results = results[:k]
            return results

        except Exception as e:
            logger.error(f"❌ Local cache search failed: {e}")
            return []

    def _direct_pdf_search(self, query: str) -> List[Dict]:
        """
        EMERGENCY FALLBACK: Direct PDF processing with Nova Lite
        When both Kendra and local cache fail
        """
        if not self.pdf_file_registry:
            logger.info("📄 No PDF files registered for direct processing")
            return []

        try:
            logger.info(f"🚨 EMERGENCY FALLBACK: Direct PDF processing for query: '{query}'")
            results = []

            for doc_id, pdf_path in self.pdf_file_registry.items():
                if not os.path.exists(pdf_path):
                    continue

                # Check PDF size (Nova Lite limit: 4.5MB)
                pdf_size = os.path.getsize(pdf_path)
                if pdf_size > 4.5 * 1024 * 1024:
                    logger.warning(f"⚠️ PDF too large for direct processing: {pdf_size/1024/1024:.1f}MB")
                    continue

                logger.info(f"📄 Processing PDF directly: {pdf_path}")

                # Read PDF as bytes
                with open(pdf_path, 'rb') as pdf_file:
                    pdf_bytes = pdf_file.read()

                # Create prompt for Nova Lite
                prompt = f"""
You are a medical AI assistant analyzing a medical document.

Question: {query}

Please analyze the attached medical document and provide a comprehensive answer to the question.

Focus on:
1. Direct answers from the document content
2. Relevant medical details
3. Specific patient information if available
4. Any diagnoses, treatments, or assessments mentioned

Be thorough and accurate in your response. Return the information in a clear, structured format.
"""

                try:
                    # Use existing aws_utils for bedrock call with document
                    logger.info("🤖 Calling Nova Lite via existing AWS utilities...")

                    # For now, let's extract text from PDF and use regular bedrock call
                    # This is a simpler approach that works with your existing setup

                    # Try to extract text content from PDF using existing document processor
                    try:
                        from document_processor import MedicalDocumentProcessor
                        temp_processor = MedicalDocumentProcessor()
                        temp_result = temp_processor.process_document_complete(pdf_path)

                        if temp_result['success']:
                            pdf_text_content = temp_result['extracted_text']
                            logger.info(f"📄 Extracted {len(pdf_text_content)} characters from PDF")

                            # Create enhanced prompt with extracted text
                            enhanced_prompt = f"""
{prompt}

MEDICAL DOCUMENT CONTENT:
{pdf_text_content}

Please provide a comprehensive answer based on the medical document content above.
"""

                            # Use existing safe_bedrock_call
                            response_text = self.aws_utils.safe_bedrock_call(enhanced_prompt, max_tokens=2000)

                            if response_text and not response_text.startswith("Error:"):
                                logger.info("✅ Successfully processed PDF via document processor + Nova Lite")
                            else:
                                raise Exception(f"Bedrock call failed: {response_text}")
                        else:
                            raise Exception(f"Document processing failed: {temp_result['error']}")

                    except Exception as fallback_error:
                        logger.warning(f"⚠️ Document processor approach failed: {fallback_error}")
                        # Fallback to simple text extraction if available
                        raise Exception("Direct PDF processing not available with current AWS setup")

                    results.append({
                        'doc_id': doc_id,
                        'filename': os.path.basename(pdf_path),
                        'content': response_text,
                        'similarity_score': 'Direct PDF Processing',
                        'source': 'direct_pdf',
                        'pdf_path': pdf_path
                    })

                    logger.info(f"✅ Direct PDF processing successful for {pdf_path}")

                except Exception as pdf_error:
                    logger.error(f"❌ Direct PDF processing failed for {pdf_path}: {pdf_error}")
                    continue

            return results

        except Exception as e:
            logger.error(f"❌ Direct PDF search failed: {e}")
            return []

    def search_across_documents(self, query: str, k: int = 10, 
                              filter_params: Optional[Dict] = None) -> List[Dict]:
        """
        CORE FEATURE: Multi-document semantic search with Kendra
        """
        try:
            logger.info(f"🔍 Multi-document search: '{query}' (k={k})")
            logger.info(f"📊 Search Strategy: Kendra first → Direct PDF fallback (no local cache)")

            # First, try Kendra for best search quality
            logger.info("🔍 Step 1: Searching Kendra index...")
            response = self.aws_utils.safe_kendra_query(
                index_id=self.kendra_index_id,
                query_text=query,
                k=k,
                attribute_filter=filter_params
            )

            if response['success']:
                search_results = response['results'].get('ResultItems', [])
                logger.info(f"🔍 Kendra response: {len(search_results)} results found")

                # DEBUG: Show raw Kendra response
                print(f"\n{'='*80}")
                print(f"DEBUG: RAW KENDRA RESPONSE")
                print(f"{'='*80}")
                print(f"Query: '{query}'")
                print(f"Success: {response['success']}")
                print(f"Results count: {len(search_results)}")
                if search_results:
                    print("First result details:")
                    print(f"  Raw result keys: {list(search_results[0].keys())}")
                    print(f"  Document Title: {search_results[0].get('DocumentTitle', 'None')}")
                    print(f"  Document Excerpt: {search_results[0].get('DocumentExcerpt', {})}")
                    print(f"  Score: {search_results[0].get('ScoreAttributes', {})}")
                print(f"{'='*80}")

                if search_results:
                    logger.info(f"✅ Kendra found {len(search_results)} results")
                    # Process Kendra results
                    formatted_results = []

                    for result in search_results:
                        doc_id_attr = next((attr for attr in result.get('DocumentAttributes', []) if attr['Key'] == 'doc_id'), None)

                        # Safe extraction of doc_id
                        if doc_id_attr and isinstance(doc_id_attr.get('Value'), dict):
                            doc_id = doc_id_attr['Value'].get('StringValue', 'unknown')
                        elif doc_id_attr and doc_id_attr.get('Value'):
                            doc_id = str(doc_id_attr['Value'])  # Convert to string if not a dict
                        else:
                            doc_id = 'unknown'

                        formatted_results.append({
                            'content': result.get('DocumentExcerpt', {}).get('Text', ''),
                            'similarity_score': result.get('ScoreAttributes', {}).get('ScoreConfidence', 'Low'),
                            'doc_id': doc_id,
                            'filename': result.get('DocumentTitle', 'unknown'),
                            'chunk_id': result.get('DocumentId', 'unknown'),
                            'metadata': result.get('DocumentAttributes', [])
                        })

                    unique_docs = set(str(result['doc_id']) for result in formatted_results if result.get('doc_id'))
                    logger.info(f"🎯 Kendra search results: {len(formatted_results)} chunks from {len(unique_docs)} documents")

                    self.search_history.append({
                        'query': query,
                        'results_count': len(formatted_results),
                        'unique_documents': len(unique_docs),
                        'timestamp': datetime.now().isoformat()
                    })

                    return formatted_results
                else:
                    # Check if we have documents in the registry (uploaded to Kendra)
                    total_docs_uploaded = len(self.document_registry)
                    logger.info(f"⚪ Kendra returned 0 results. Documents uploaded to Kendra: {total_docs_uploaded}")

                    if total_docs_uploaded > 0:
                        logger.info("📋 Documents exist in Kendra but no matches found (likely still indexing).")
                        logger.info("🔍 Step 2: Skipping local cache - going to Direct PDF processing...")
                    else:
                        logger.info("⚠️ No documents uploaded to Kendra yet.")
                        logger.info("🔍 Step 2: Trying Direct PDF processing...")

                    # Skip local cache entirely - go straight to direct PDF processing
                    logger.info("🚨 ACTIVATING EMERGENCY FALLBACK: Direct PDF processing")
                    direct_results = self._direct_pdf_search(query)
                    if direct_results:
                        logger.info(f"🎉 Direct PDF processing found {len(direct_results)} results!")
                        return direct_results
                    else:
                        logger.error("❌ Both Kendra and Direct PDF processing failed")
                        return []
            else:
                logger.warning(f"⚠️ Kendra search failed: {response['error']}")
                logger.info("🔍 Skipping local cache - trying Direct PDF processing...")

                # Skip local cache entirely - go straight to direct PDF processing
                direct_results = self._direct_pdf_search(query)
                if direct_results:
                    logger.info(f"🎉 Direct PDF processing found {len(direct_results)} results!")
                    return direct_results
                else:
                    logger.error("❌ Both Kendra and Direct PDF processing failed")
                    return []
            
        except Exception as e:
            logger.error(f"❌ Kendra search failed: {e}")
            return []

    def check_kendra_status(self) -> Dict:
        """
        Check Kendra index status and document count
        """
        try:
            kendra_client = self.aws_config.get_service_client('kendra')

            # Get index description
            index_info = kendra_client.describe_index(Id=self.kendra_index_id)

            # Get index statistics
            try:
                stats_response = kendra_client.describe_index(Id=self.kendra_index_id)
                index_stats = stats_response.get('IndexStatistics', {})
            except:
                index_stats = {}

            status = {
                'index_id': self.kendra_index_id,
                'index_status': index_info.get('Status', 'Unknown'),
                'documents_uploaded_by_app': len(self.document_registry),
                'local_cache_docs': len(self.local_document_cache),
                'index_created': index_info.get('CreatedAt', 'Unknown'),
                'index_updated': index_info.get('UpdatedAt', 'Unknown'),
                'indexed_text_documents': index_stats.get('TextDocumentStatistics', {}).get('IndexedTextDocumentsCount', 'Unknown'),
                'indexing_errors': index_stats.get('TextDocumentStatistics', {}).get('IndexedTextBytesCount', 'Unknown')
            }

            logger.info(f"📊 Kendra Status: {status}")
            print(f"\n{'='*50}")
            print(f"KENDRA INDEX STATUS")
            print(f"{'='*50}")
            print(f"Index Status: {status['index_status']}")
            print(f"Documents uploaded by app: {status['documents_uploaded_by_app']}")
            print(f"Actually indexed documents: {status['indexed_text_documents']}")
            print(f"Local cache documents: {status['local_cache_docs']}")
            print(f"{'='*50}")

            return status

        except Exception as e:
            logger.error(f"❌ Failed to check Kendra status: {e}")
            return {'error': str(e)}

    def get_system_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive system statistics for Kendra
        """
        try:
            total_docs = len(self.document_registry)
            total_chunks = sum(doc['chunks_stored'] for doc in self.document_registry.values())
            
            return {
                'vector_database': {
                    'total_documents': total_docs,
                    'total_chunks': total_chunks,
                },
                'search_analytics': {
                    'total_searches': len(self.search_history),
                    'unique_queries': len(set(search['query'] for search in self.search_history)),
                },
                'model_info': {
                    'vector_database': 'AWS Kendra',
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Statistics generation failed: {e}")
            return {'error': str(e)}

# def test_vector_search():
#     """
#     Test function for vector search system with Kendra
#     """
#     print("🧪 Testing Medical Vector Search System with Kendra...")
    
#     try:
#         vector_search = MedicalVectorSearch()
        
#         sample_text = """
#         Patient is a 65-year-old male with type 2 diabetes mellitus and hypertension.
#         Current medications include metformin 500mg twice daily, lisinopril 10mg daily,
#         and aspirin 81mg for cardioprotection. Recent HbA1c is 7.2%, showing good
#         glycemic control. Blood pressure is well-controlled at 130/80 mmHg.
#         """
        
#         chunks = vector_search.chunk_text(sample_text, "test_doc")
#         print(f"✅ Text chunking: {len(chunks)} chunks created")
        
#         mock_medical_entities = {
#             'medications': [{'text': 'metformin 500mg'}, {'text': 'lisinopril 10mg'}],
#             'conditions': [{'text': 'diabetes mellitus'}, {'text': 'hypertension'}]
#         }
        
#         success = vector_search.store_document_vectors(
#             "test_doc_001",
#             sample_text,
#             mock_medical_entities,
#             ["Chart shows blood glucose trending downward"],
#             {"filename": "test_patient.pdf"}
#         )
#         print(f"✅ Kendra vector storage: {'Success' if success else 'Failed'}")
        
#         search_results = vector_search.search_across_documents(
#             "diabetes medications and blood pressure",
#             k=5
#         )
#         print(f"✅ Kendra multi-document search: {len(search_results)} results found")
        
#         stats = vector_search.get_system_statistics()
#         print(f"✅ System statistics: {stats.get('vector_database', {}).get('total_chunks', 0)} chunks in database")
        
#         print("🎉 Kendra vector search system test PASSED!")
#         return True
        
#     except Exception as e:
#         print(f"❌ Kendra vector search system test FAILED: {e}")
#         return False

# if __name__ == "__main__":
#     print("🚀 Medical Vector Search System - Person 3")
#     print("=" * 50)
    
#     if test_vector_search():
#         print("\n✅ Kendra vector search system ready!")
#     else:
#         print("\n❌ Kendra vector search system needs troubleshooting")
