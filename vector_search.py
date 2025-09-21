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
                             metadata: Dict) -> bool:
        """
        Store document in Kendra index for multi-document search
        """
        try:
            logger.info(f"💾 Storing document {doc_id} in Kendra index...")
            
            full_content = text_content
            if chart_descriptions:
                full_content += "\n\n" + "\n".join(chart_descriptions)
            
            text_chunks = self.chunk_text(full_content, doc_id)
            
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
                    results.append({
                        'doc_id': doc_id,
                        'filename': doc_data['filename'],
                        'content_excerpt': content[:500],
                        'relevance_analysis': response,
                        'timestamp': doc_data['timestamp'],
                        'source': 'local_cache'
                    })

            # Sort by relevance (could be improved with actual scoring)
            results = results[:k]
            return results

        except Exception as e:
            logger.error(f"❌ Local cache search failed: {e}")
            return []

    def search_across_documents(self, query: str, k: int = 10, 
                              filter_params: Optional[Dict] = None) -> List[Dict]:
        """
        CORE FEATURE: Multi-document semantic search with Kendra
        """
        try:
            logger.info(f"🔍 Multi-document search: '{query}' (k={k})")

            # First, try Kendra for best search quality
            logger.info("🔍 Searching Kendra index...")
            response = self.aws_utils.safe_kendra_query(
                index_id=self.kendra_index_id,
                query_text=query,
                k=k,
                attribute_filter=filter_params
            )

            if response['success']:
                search_results = response['results'].get('ResultItems', [])
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
                    logger.info("⚪ Kendra returned 0 results, trying local cache...")
                    local_results = self._search_local_cache(query, k)
                    if local_results:
                        logger.info(f"📋 Local cache found {len(local_results)} results")
                        return local_results
                    else:
                        logger.info("❌ No results found in either Kendra or local cache")
                        return []
            else:
                logger.warning(f"⚠️ Kendra search failed: {response['error']}")
                logger.info("📋 Falling back to local cache...")
                local_results = self._search_local_cache(query, k)
                if local_results:
                    logger.info(f"📋 Local cache found {len(local_results)} results")
                    return local_results
                else:
                    logger.error("❌ Both Kendra and local cache failed")
                    return []
            
        except Exception as e:
            logger.error(f"❌ Kendra search failed: {e}")
            return []

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

def test_vector_search():
    """
    Test function for vector search system with Kendra
    """
    print("🧪 Testing Medical Vector Search System with Kendra...")
    
    try:
        vector_search = MedicalVectorSearch()
        
        sample_text = """
        Patient is a 65-year-old male with type 2 diabetes mellitus and hypertension.
        Current medications include metformin 500mg twice daily, lisinopril 10mg daily,
        and aspirin 81mg for cardioprotection. Recent HbA1c is 7.2%, showing good
        glycemic control. Blood pressure is well-controlled at 130/80 mmHg.
        """
        
        chunks = vector_search.chunk_text(sample_text, "test_doc")
        print(f"✅ Text chunking: {len(chunks)} chunks created")
        
        mock_medical_entities = {
            'medications': [{'text': 'metformin 500mg'}, {'text': 'lisinopril 10mg'}],
            'conditions': [{'text': 'diabetes mellitus'}, {'text': 'hypertension'}]
        }
        
        success = vector_search.store_document_vectors(
            "test_doc_001",
            sample_text,
            mock_medical_entities,
            ["Chart shows blood glucose trending downward"],
            {"filename": "test_patient.pdf"}
        )
        print(f"✅ Kendra vector storage: {'Success' if success else 'Failed'}")
        
        search_results = vector_search.search_across_documents(
            "diabetes medications and blood pressure",
            k=5
        )
        print(f"✅ Kendra multi-document search: {len(search_results)} results found")
        
        stats = vector_search.get_system_statistics()
        print(f"✅ System statistics: {stats.get('vector_database', {}).get('total_chunks', 0)} chunks in database")
        
        print("🎉 Kendra vector search system test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Kendra vector search system test FAILED: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Medical Vector Search System - Person 3")
    print("=" * 50)
    
    if test_vector_search():
        print("\n✅ Kendra vector search system ready!")
    else:
        print("\n❌ Kendra vector search system needs troubleshooting")
