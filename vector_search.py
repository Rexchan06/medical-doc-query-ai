#!/usr/bin/env python3
"""
PERSON 3: AI/ML Developer
Vector search, embeddings, and pattern analysis

RESPONSIBILITIES:
- ChromaDB vector database setup and management
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

# Vector database and ML libraries
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Import Person 1's AWS configuration
try:
    from aws_config import AWSConfigManager, AWSUtilities
except ImportError:
    print("❌ Error: aws_config.py not found. Make sure Person 1 has completed AWS setup.")
    exit(1)

# Import Person 2's document processor
try:
    from document_processor import MedicalDocumentProcessor
except ImportError:
    print("❌ Error: document_processor.py not found. Make sure Person 2 has completed backend setup.")
    exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalVectorSearch:
    """
    Advanced vector search system for medical documents
    Enables semantic search across multiple documents
    """
    
    def __init__(self):
        """Initialize the vector search system"""
        logger.info("🧠 Initializing Medical Vector Search System...")
        
        # Initialize embedding model
        self._initialize_embedding_model()
        
        # Initialize vector database
        self._initialize_vector_database()
        
        # Initialize AWS utilities (for pattern analysis)
        try:
            from aws_config import setup_aws_environment
            self.aws_config, self.aws_utils = setup_aws_environment()
        except Exception as e:
            logger.warning(f"⚠️ AWS utilities not available: {e}")
            self.aws_utils = None
        
        # Configuration
        self.chunk_size = 500
        self.chunk_overlap = 50
        self.max_chunks_per_doc = 50  # Prevent memory issues
        
        # Document tracking
        self.document_registry = {}
        self.search_history = []
        
        logger.info("✅ Medical Vector Search System ready!")
    
    def _initialize_embedding_model(self):
        """Initialize the sentence transformer model for embeddings"""
        try:
            logger.info("📊 Loading embedding model...")
            
            # Use efficient model for hackathon
            model_name = 'all-MiniLM-L6-v2'  # Fast, good quality, 384 dimensions
            self.embedding_model = SentenceTransformer(model_name)
            
            # Test embedding generation
            test_embedding = self.embedding_model.encode(["test medical text"])
            self.embedding_dimension = test_embedding.shape[1]
            
            logger.info(f"✅ Embedding model loaded: {model_name} ({self.embedding_dimension}D)")
            
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            raise
    
    def _initialize_vector_database(self):
        """Initialize ChromaDB vector database"""
        try:
            logger.info("🗄️ Setting up ChromaDB vector database...")
            
            # Initialize ChromaDB client
            self.chroma_client = chromadb.Client(
                Settings(anonymized_telemetry=False)
            )
            
            # Create or get collection for medical documents
            collection_name = "medical_documents"
            try:
                self.collection = self.chroma_client.get_collection(collection_name)
                existing_count = self.collection.count()
                logger.info(f"📚 Connected to existing collection with {existing_count} chunks")
            except:
                self.collection = self.chroma_client.create_collection(
                    name=collection_name,
                    metadata={"description": "Medical document embeddings for semantic search"}
                )
                logger.info("📚 Created new medical documents collection")
            
        except Exception as e:
            logger.error(f"❌ Vector database initialization failed: {e}")
            raise
    
    def chunk_text(self, text: str, doc_id: str) -> List[Dict]:
        """
        Intelligent text chunking for medical documents
        Creates overlapping chunks while preserving medical context
        """
        if not text or not text.strip():
            return []
        
        # Split into sentences first (better for medical context)
        sentences = text.split('.')
        chunks = []
        current_chunk = ""
        chunk_id = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if adding sentence would exceed chunk size
            potential_chunk = current_chunk + ". " + sentence if current_chunk else sentence
            
            if len(potential_chunk.split()) <= self.chunk_size:
                current_chunk = potential_chunk
            else:
                # Save current chunk if it's substantial
                if len(current_chunk.split()) > 20:  # Minimum chunk size
                    chunks.append({
                        'content': current_chunk,
                        'chunk_id': f"{doc_id}_chunk_{chunk_id}",
                        'doc_id': doc_id,
                        'chunk_index': chunk_id,
                        'word_count': len(current_chunk.split())
                    })
                    chunk_id += 1
                
                # Start new chunk with overlap
                words = current_chunk.split()
                if len(words) > self.chunk_overlap:
                    overlap_text = " ".join(words[-self.chunk_overlap:])
                    current_chunk = overlap_text + ". " + sentence
                else:
                    current_chunk = sentence
        
        # Add final chunk
        if current_chunk and len(current_chunk.split()) > 20:
            chunks.append({
                'content': current_chunk,
                'chunk_id': f"{doc_id}_chunk_{chunk_id}",
                'doc_id': doc_id,
                'chunk_index': chunk_id,
                'word_count': len(current_chunk.split())
            })
        
        logger.info(f"📑 Created {len(chunks)} chunks for document {doc_id}")
        return chunks[:self.max_chunks_per_doc]  # Limit chunks to prevent memory issues
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate optimized embeddings for text chunks
        """
        if not texts:
            return np.array([])
        
        try:
            logger.info(f"🧮 Generating embeddings for {len(texts)} text chunks...")
            
            # Generate embeddings in batches for efficiency
            batch_size = 32
            embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = self.embedding_model.encode(
                    batch, 
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                embeddings.append(batch_embeddings)
            
            # Combine all embeddings
            all_embeddings = np.vstack(embeddings)
            
            logger.info(f"✅ Generated embeddings: {all_embeddings.shape}")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"❌ Embedding generation failed: {e}")
            return np.array([])
    
    def store_document_vectors(self, doc_id: str, text_content: str, 
                             medical_entities: Dict, chart_descriptions: List[str],
                             metadata: Dict) -> bool:
        """
        Store document in vector database for multi-document search
        This is the core function that enables cross-document search
        """
        try:
            logger.info(f"💾 Storing document {doc_id} in vector database...")
            
            # Combine text content with chart descriptions
            full_content = text_content
            if chart_descriptions:
                full_content += "\n\n" + "\n".join(chart_descriptions)
            
            # Create text chunks
            text_chunks = self.chunk_text(full_content, doc_id)
            
            if not text_chunks:
                logger.warning(f"⚠️ No chunks created for document {doc_id}")
                return False
            
            # Generate embeddings
            chunk_texts = [chunk['content'] for chunk in text_chunks]
            embeddings = self.generate_embeddings(chunk_texts)
            
            if embeddings.size == 0:
                logger.error(f"❌ No embeddings generated for document {doc_id}")
                return False
            
            # Prepare data for ChromaDB storage
            documents = []
            metadatas = []
            ids = []
            embedding_list = []
            
            for i, chunk in enumerate(text_chunks):
                # Determine if chunk contains chart information
                is_chart_chunk = any(desc in chunk['content'] for desc in chart_descriptions)
                
                documents.append(chunk['content'])
                
                # Rich metadata for filtering and analysis
                chunk_metadata = {
                    'doc_id': doc_id,
                    'chunk_id': chunk['chunk_id'],
                    'chunk_index': chunk['chunk_index'],
                    'filename': metadata.get('filename', 'unknown'),
                    'timestamp': datetime.now().isoformat(),
                    'word_count': chunk['word_count'],
                    'contains_chart_info': is_chart_chunk,
                    'extraction_method': metadata.get('extraction_method', 'unknown')
                }
                
                # Add medical entity counts to metadata
                if medical_entities:
                    chunk_metadata.update({
                        'medications_in_doc': len(medical_entities.get('medications', [])),
                        'conditions_in_doc': len(medical_entities.get('conditions', [])),
                        'procedures_in_doc': len(medical_entities.get('procedures', []))
                    })
                
                metadatas.append(chunk_metadata)
                ids.append(chunk['chunk_id'])
                embedding_list.append(embeddings[i].tolist())
            
            # Store in ChromaDB
            self.collection.add(
                documents=documents,
                embeddings=embedding_list,
                metadatas=metadatas,
                ids=ids
            )
            
            # Update document registry
            self.document_registry[doc_id] = {
                'filename': metadata.get('filename', 'unknown'),
                'chunks_stored': len(documents),
                'timestamp': datetime.now().isoformat(),
                'medical_entities': medical_entities,
                'has_chart_data': len(chart_descriptions) > 0
            }
            
            logger.info(f"✅ Stored {len(documents)} chunks for document {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Vector storage failed for document {doc_id}: {e}")
            return False
    
    def search_across_documents(self, query: str, k: int = 10, 
                              filter_params: Optional[Dict] = None) -> List[Dict]:
        """
        CORE FEATURE: Multi-document semantic search
        This is what makes the system powerful - searches across ALL documents
        """
        try:
            logger.info(f"🔍 Multi-document search: '{query}' (k={k})")
            
            # Check if we have any documents
            total_chunks = self.collection.count()
            if total_chunks == 0:
                logger.warning("⚠️ No documents in vector database")
                return []
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            
            # Prepare search parameters
            search_params = {
                'query_embeddings': query_embedding.tolist(),
                'n_results': min(k, total_chunks)
            }
            
            # Add filters if specified
            if filter_params:
                search_params['where'] = filter_params
            
            # Execute vector search
            search_results = self.collection.query(**search_params)
            
            # Process and format results
            formatted_results = []
            
            if search_results['documents'] and len(search_results['documents'][0]) > 0:
                for i in range(len(search_results['documents'][0])):
                    # Calculate similarity score
                    distance = search_results['distances'][0][i]
                    similarity_score = 1 - distance  # Convert distance to similarity
                    
                    # Extract metadata
                    metadata = search_results['metadatas'][0][i]
                    
                    formatted_result = {
                        'content': search_results['documents'][0][i],
                        'similarity_score': similarity_score,
                        'doc_id': metadata['doc_id'],
                        'filename': metadata['filename'],
                        'chunk_id': metadata['chunk_id'],
                        'contains_chart_info': metadata.get('contains_chart_info', False),
                        'word_count': metadata.get('word_count', 0),
                        'metadata': metadata
                    }
                    
                    formatted_results.append(formatted_result)
            
            # Get document diversity statistics
            unique_docs = set(result['doc_id'] for result in formatted_results)
            
            # Log search statistics
            chart_chunks = sum(1 for r in formatted_results if r['contains_chart_info'])
            
            logger.info(f"🎯 Search results: {len(formatted_results)} chunks from "
                       f"{len(unique_docs)} documents ({chart_chunks} with chart data)")
            
            # Track search for analytics
            self.search_history.append({
                'query': query,
                'results_count': len(formatted_results),
                'unique_documents': len(unique_docs),
                'timestamp': datetime.now().isoformat()
            })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"❌ Multi-document search failed: {e}")
            return []
    
    def analyze_document_patterns(self, search_results: List[Dict], 
                                query: str) -> Dict[str, Any]:
        """
        ADVANCED FEATURE: Cross-document pattern analysis
        Finds patterns across multiple medical documents
        """
        try:
            logger.info(f"📊 Analyzing patterns across {len(search_results)} results...")
            
            if not search_results:
                return {'error': 'No search results to analyze'}
            
            # Group results by document
            doc_groups = defaultdict(list)
            for result in search_results:
                doc_groups[result['doc_id']].append(result)
            
            # Extract medical patterns
            patterns = self._extract_medical_patterns(doc_groups)
            
            # Generate AI insights if AWS is available
            ai_insights = ""
            if self.aws_utils:
                ai_insights = self._generate_pattern_insights(patterns, query, doc_groups)
            
            # Compile pattern analysis
            pattern_analysis = {
                'query': query,
                'documents_analyzed': len(doc_groups),
                'total_chunks_analyzed': len(search_results),
                'patterns': patterns,
                'ai_insights': ai_insights,
                'document_coverage': {
                    doc_id: len(chunks) for doc_id, chunks in doc_groups.items()
                },
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"✅ Pattern analysis complete: {len(doc_groups)} documents analyzed")
            return pattern_analysis
            
        except Exception as e:
            logger.error(f"❌ Pattern analysis failed: {e}")
            return {'error': str(e)}
    
    def _extract_medical_patterns(self, doc_groups: Dict) -> Dict[str, Any]:
        """
        Extract medical patterns from grouped document results
        """
        patterns = {
            'medication_frequency': Counter(),
            'condition_frequency': Counter(),
            'common_phrases': Counter(),
            'document_similarity': {},
            'chart_data_distribution': {}
        }
        
        # Medical terms to look for (expandable)
        medical_terms = {
            'medications': ['metformin', 'insulin', 'aspirin', 'warfarin', 'atorvastatin', 
                          'lisinopril', 'amlodipine', 'furosemide', 'digoxin'],
            'conditions': ['diabetes', 'hypertension', 'heart disease', 'obesity',
                         'atrial fibrillation', 'coronary artery disease', 'heart failure'],
            'procedures': ['echocardiogram', 'stress test', 'catheterization', 'angioplasty',
                         'bypass surgery', 'pacemaker', 'defibrillator']
        }
        
        for doc_id, chunks in doc_groups.items():
            # Combine all chunks for this document
            doc_content = " ".join([chunk['content'].lower() for chunk in chunks])
            
            # Count medical terms
            for term in medical_terms['medications']:
                if term in doc_content:
                    patterns['medication_frequency'][term] += 1
            
            for term in medical_terms['conditions']:
                if term in doc_content:
                    patterns['condition_frequency'][term] += 1
            
            # Chart data analysis
            chart_chunks = [chunk for chunk in chunks if chunk['contains_chart_info']]
            patterns['chart_data_distribution'][doc_id] = len(chart_chunks)
            
            # Extract common medical phrases (3-4 word combinations)
            words = doc_content.split()
            for i in range(len(words) - 2):
                phrase = " ".join(words[i:i+3])
                if any(med_term in phrase for med_term in 
                      medical_terms['medications'] + medical_terms['conditions']):
                    patterns['common_phrases'][phrase] += 1
        
        return patterns
    
    def _generate_pattern_insights(self, patterns: Dict, query: str, 
                                 doc_groups: Dict) -> str:
        """
        Generate AI-powered insights from patterns using Claude
        """
        if not self.aws_utils:
            return "AI insights not available (AWS not configured)"
        
        try:
            # Prepare pattern summary for Claude
            pattern_summary = f"""
MEDICAL PATTERN ANALYSIS - Query: "{query}"

DOCUMENTS ANALYZED: {len(doc_groups)}

TOP MEDICATIONS ACROSS DOCUMENTS:
{dict(patterns['medication_frequency'].most_common(10))}

TOP CONDITIONS ACROSS DOCUMENTS:
{dict(patterns['condition_frequency'].most_common(10))}

CHART DATA DISTRIBUTION:
{patterns['chart_data_distribution']}

COMMON MEDICAL PHRASES:
{dict(patterns['common_phrases'].most_common(5))}
"""

            # Generate insights with Claude
            insight_prompt = f"""As a medical data analyst, analyze these patterns from multiple patient documents:

{pattern_summary}

Provide insights on:
1. **Key Medical Patterns**: Most significant patterns across patients
2. **Treatment Trends**: What treatments/medications appear most effective
3. **Risk Patterns**: Any concerning patterns or combinations
4. **Clinical Recommendations**: Actionable insights for healthcare providers
5. **Population Health**: What this data suggests about patient population

Focus on clinically relevant insights that improve patient care.

ANALYSIS:"""

            # Generate AI insights
            ai_response = self.aws_utils.safe_bedrock_call(
                insight_prompt, 
                max_tokens=600
            )
            
            return ai_response
            
        except Exception as e:
            logger.error(f"❌ AI insight generation failed: {e}")
            return f"AI insights generation failed: {str(e)}"
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive system statistics
        """
        try:
            total_chunks = self.collection.count()
            total_docs = len(self.document_registry)
            
            # Calculate average chunks per document
            avg_chunks = total_chunks / total_docs if total_docs > 0 else 0
            
            # Chart data statistics
            docs_with_charts = sum(1 for doc in self.document_registry.values() 
                                 if doc.get('has_chart_data', False))
            
            # Recent search statistics
            recent_searches = len(self.search_history)
            unique_queries = len(set(search['query'] for search in self.search_history))
            
            return {
                'vector_database': {
                    'total_chunks': total_chunks,
                    'total_documents': total_docs,
                    'average_chunks_per_doc': round(avg_chunks, 2),
                    'embedding_dimension': self.embedding_dimension
                },
                'document_analysis': {
                    'documents_with_charts': docs_with_charts,
                    'chart_coverage_percentage': round((docs_with_charts / total_docs * 100), 1) if total_docs > 0 else 0
                },
                'search_analytics': {
                    'total_searches': recent_searches,
                    'unique_queries': unique_queries,
                    'search_diversity': round((unique_queries / recent_searches), 2) if recent_searches > 0 else 0
                },
                'model_info': {
                    'embedding_model': 'all-MiniLM-L6-v2',
                    'vector_database': 'ChromaDB',
                    'chunk_size': self.chunk_size,
                    'chunk_overlap': self.chunk_overlap
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Statistics generation failed: {e}")
            return {'error': str(e)}

def test_vector_search():
    """
    Test function for vector search system
    """
    print("🧪 Testing Medical Vector Search System...")
    
    try:
        # Initialize vector search
        vector_search = MedicalVectorSearch()
        
        # Test embedding generation
        test_texts = [
            "Patient has diabetes and takes metformin 500mg twice daily",
            "Blood pressure medications include lisinopril and amlodipine",
            "Echocardiogram shows normal left ventricular function"
        ]
        
        embeddings = vector_search.generate_embeddings(test_texts)
        print(f"✅ Embedding generation: {embeddings.shape}")
        
        # Test text chunking
        sample_text = """
        Patient is a 65-year-old male with type 2 diabetes mellitus and hypertension.
        Current medications include metformin 500mg twice daily, lisinopril 10mg daily,
        and aspirin 81mg for cardioprotection. Recent HbA1c is 7.2%, showing good
        glycemic control. Blood pressure is well-controlled at 130/80 mmHg.
        """
        
        chunks = vector_search.chunk_text(sample_text, "test_doc")
        print(f"✅ Text chunking: {len(chunks)} chunks created")
        
        # Test vector storage (with mock data)
        mock_medical_entities = {
            'medications': [
                {'text': 'metformin 500mg', 'confidence': 0.95},
                {'text': 'lisinopril 10mg', 'confidence': 0.92}
            ],
            'conditions': [
                {'text': 'diabetes mellitus', 'confidence': 0.98},
                {'text': 'hypertension', 'confidence': 0.94}
            ]
        }
        
        success = vector_search.store_document_vectors(
            "test_doc_001",
            sample_text,
            mock_medical_entities,
            ["Chart shows blood glucose trending downward over 6 months"],
            {"filename": "test_patient.pdf", "extraction_method": "test"}
        )
        print(f"✅ Vector storage: {'Success' if success else 'Failed'}")
        
        # Test multi-document search
        search_results = vector_search.search_across_documents(
            "diabetes medications and blood pressure",
            k=5
        )
        print(f"✅ Multi-document search: {len(search_results)} results found")
        
        # Test pattern analysis
        if search_results:
            pattern_analysis = vector_search.analyze_document_patterns(
                search_results,
                "diabetes medications and blood pressure"
            )
            print(f"✅ Pattern analysis: {pattern_analysis.get('documents_analyzed', 0)} documents analyzed")
        
        # Get system statistics
        stats = vector_search.get_system_statistics()
        print(f"✅ System statistics: {stats.get('vector_database', {}).get('total_chunks', 0)} chunks in database")
        
        print("🎉 Vector search system test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Vector search system test FAILED: {e}")
        return False

if __name__ == "__main__":
    """
    Run this file directly to test vector search system
    """
    
    print("🚀 Medical Vector Search System - Person 3")
    print("=" * 50)
    
    # Test the vector search system
    if test_vector_search():
        print("\n✅ Vector search system ready!")
        print("🔗 Integration points:")
        print("   - Person 2: Use store_document_vectors() after document processing")
        print("   - Person 4: Integrate chart descriptions into vector storage")  
        print("   - Person 5: Connect search_across_documents() to Gradio interface")
        print("\n📋 Key Functions available:")
        print("   - store_document_vectors(doc_id, text, entities, charts, metadata)")
        print("   - search_across_documents(query, k, filters)")
        print("   - analyze_document_patterns(search_results, query)")
        print("   - get_system_statistics()")
    else:
        print("\n❌ Vector search system needs troubleshooting")
        print("🔧 Check:")
        print("   1. sentence-transformers library installed")
        print("   2. chromadb library installed")
        print("   3. AWS configuration (for pattern insights)")
        print("   4. Available memory for embedding model")