#!/usr/bin/env python3
"""
PERSON 1: AWS Infrastructure Lead
Complete AWS setup, configuration, and utilities

RESPONSIBILITIES:
- AWS services setup and configuration
- Credential management and security
- AWS utilities and error handling
- Cost monitoring and optimization
- Integration support for all team members

DELIVERABLES:
- aws_config.py (this file)
- Working AWS connections for all services
- Shared utilities for team use
- Cost tracking and monitoring
"""

import boto3
import json
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
import os
from botocore.exceptions import ClientError, NoCredentialsError

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ .env file loaded successfully")
except ImportError:
    print("⚠️ python-dotenv not installed. Install with: pip install python-dotenv")
    print("📝 Alternatively, set environment variables manually")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AWSConfigManager:
    """
    Central AWS configuration and management
    All team members will use this for AWS operations
    """
    
    def __init__(self, region_name: str = 'us-east-1'):
        """Initialize AWS configuration manager"""
        self.region_name = region_name
        self.services = {}
        self.account_id = 154744860097
        self.cost_tracker = {
            'bedrock_tokens': 0,
            'kendra_queries': 0,
            'kendra_documents': 0,
            'estimated_cost': 0.0
        }
        
        # Initialize all AWS service clients
        self._initialize_aws_services()
        
        # Test connectivity
        self._test_all_services()
    
    def _initialize_aws_services(self):
        """Initialize all required AWS service clients"""
        try:
            logger.info("🔧 Initializing AWS services for Nova Lite + Kendra...")

            # Core AI services (simplified)
            self.services['bedrock'] = boto3.client('bedrock-runtime', region_name=self.region_name)
            self.services['kendra'] = boto3.client('kendra', region_name=self.region_name)

            # Storage (optional)
            self.services['s3'] = boto3.client('s3', region_name=self.region_name)

            logger.info("✅ Nova Lite + Kendra services initialized successfully")
            
        except NoCredentialsError:
            logger.error("❌ AWS credentials not configured. Run 'aws configure'")
            raise
        except Exception as e:
            logger.error(f"❌ AWS initialization failed: {e}")
            raise
    
    def _test_all_services(self) -> Dict[str, bool]:
        """Test connectivity to Nova Lite and Kendra services"""
        logger.info("🧪 Testing Nova Lite + Kendra connectivity...")

        service_status = {}

        # Test Bedrock Nova Lite
        try:
            response = self.services['bedrock'].converse(
                modelId='us.amazon.nova-lite-v1:0',
                messages=[{'role': 'user', 'content': [{'text': 'test'}]}],
                inferenceConfig={'temperature': 0.1, 'maxTokens': 10}
            )
            service_status['bedrock'] = True
            logger.info("✅ Bedrock Nova Lite connectivity confirmed")
        except Exception as e:
            logger.error(f"❌ Bedrock Nova Lite test failed: {e}")
            service_status['bedrock'] = False

        # Test Kendra
        try:
            self.services['kendra'].list_indices()
            service_status['kendra'] = True
            logger.info("✅ Kendra connectivity confirmed")
        except Exception as e:
            service_status['kendra'] = False
            logger.warning(f"⚠️ Kendra test failed: {e}")

        # Log overall status
        working_services = sum(service_status.values())
        total_services = len(service_status)
        logger.info(f"📊 AWS Services Status: {working_services}/{total_services} working")

        return service_status
    
    def get_service_client(self, service_name: str):
        """Get AWS service client safely"""
        if service_name not in self.services:
            raise ValueError(f"Service {service_name} not available")
        return self.services[service_name]
    
    def track_cost(self, service: str, operation: str, quantity: int = 1):
        """Track AWS costs for budget monitoring"""
        
        # Cost estimates (approximate)
        cost_per_unit = {
            'bedrock_1k_tokens': 0.00025,
            'kendra_query': 0.0003,
            'kendra_document': 0.01
        }
        
        cost_key = f"{service}_{operation}"
        if cost_key in cost_per_unit:
            cost = cost_per_unit[cost_key] * quantity
            self.cost_tracker['estimated_cost'] += cost
            
            logger.info(f"💰 Cost tracking: {service} {operation} x{quantity} = ${cost:.4f}")
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get current cost summary"""
        return {
            'total_estimated_cost': round(self.cost_tracker['estimated_cost'], 4),
            'remaining_budget': round(100 - self.cost_tracker['estimated_cost'], 2),
            'services_used': ['bedrock-nova-lite', 'kendra', 's3'],
            'timestamp': datetime.now().isoformat()
        }
    
    def create_s3_bucket_if_needed(self, bucket_name: str) -> bool:
        """Create S3 bucket for document storage if needed"""
        try:
            s3_client = self.get_service_client('s3')
            
            # Check if bucket exists
            try:
                s3_client.head_bucket(Bucket=bucket_name)
                logger.info(f"✅ S3 bucket {bucket_name} already exists")
                return True
            except ClientError:
                pass
            
            # Create bucket
            if self.region_name == 'us-east-1':
                s3_client.create_bucket(Bucket=bucket_name)
            else:
                s3_client.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': self.region_name}
                )
            
            logger.info(f"✅ Created S3 bucket: {bucket_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create S3 bucket {bucket_name}: {e}")
            return False

class AWSUtilities:
    """
    Shared AWS utility functions for all team members
    """
    
    def __init__(self, aws_config: AWSConfigManager):
        self.aws_config = aws_config
        self.logger = logging.getLogger(__name__)
    
    def safe_bedrock_call(self, prompt: str, max_tokens: int = 500, model_id: str = None) -> str:
        """
        Safe Bedrock API call with error handling and cost tracking
        For use by Person 3 (AI/ML) and Person 4 (Document Processing)
        """
        try:
            bedrock_client = self.aws_config.get_service_client('bedrock')

            if model_id is None:
                model_id = 'us.amazon.nova-lite-v1:0'

            # Use the correct converse API for Nova Lite
            response = bedrock_client.converse(
                modelId=model_id,
                messages=[{'role': 'user', 'content': [{'text': prompt}]}],
                inferenceConfig={'temperature': 0.1, 'maxTokens': max_tokens}
            )

            # Parse response from converse API
            answer = response['output']['message']['content'][0]['text']

            # Track costs (approximate)
            estimated_tokens = len(prompt.split()) + len(answer.split())
            self.aws_config.track_cost('bedrock', '1k_tokens', max(1, estimated_tokens // 1000))

            self.logger.info(f"✅ Bedrock Nova Lite call successful: {len(answer)} characters")
            return answer

        except Exception as e:
            self.logger.error(f"❌ Bedrock Nova Lite API call failed: {e}")
            return f"Error: {str(e)}"
    
    def safe_bedrock_vision_call(self, image_base64: str, prompt: str) -> str:
        """
        Safe Bedrock vision API call for image analysis using Nova Lite
        For use by Person 4 (Document Processing)
        """
        try:
            import base64
            bedrock_client = self.aws_config.get_service_client('bedrock')

            # Decode base64 string to bytes
            image_bytes = base64.b64decode(image_base64)

            # Use converse API for Nova Lite vision
            response = bedrock_client.converse(
                modelId='us.amazon.nova-lite-v1:0',
                messages=[{
                    'role': 'user',
                    'content': [
                        {'text': prompt},
                        {
                            'image': {
                                'format': 'png',
                                'source': {'bytes': image_bytes}
                            }
                        }
                    ]
                }],
                inferenceConfig={'temperature': 0.1, 'maxTokens': 400}
            )

            # Parse response from converse API
            answer = response['output']['message']['content'][0]['text']

            # Track vision costs
            self.aws_config.track_cost('bedrock', 'vision_image', 1)

            self.logger.info("✅ Bedrock Nova Lite vision call successful")
            return answer

        except Exception as e:
            self.logger.error(f"❌ Bedrock Nova Lite vision call failed: {e}")
            return f"Vision analysis failed: {str(e)}"
    
    def safe_kendra_create_index(self, index_name: str, description: str = "Medical Document Index") -> Dict:
        """
        Create a Kendra index for medical documents
        """
        try:
            kendra_client = self.aws_config.get_service_client('kendra')
            
            # Create index
            response = kendra_client.create_index(
                Name=index_name,
                Description=description,
                RoleArn=f"arn:aws:iam::{self.aws_config.account_id}:role/KendraServiceRole",
                Edition='DEVELOPER_EDITION'
            )
            
            index_id = response['Id']
            self.logger.info(f"✅ Kendra index created: {index_id}")
            
            return {'success': True, 'index_id': index_id}
            
        except Exception as e:
            self.logger.error(f"❌ Kendra index creation failed: {e}")
            return {'error': str(e), 'success': False}
    
    def safe_kendra_batch_put_document(self, index_id: str, documents: List[Dict]) -> Dict:
        """
        Upload documents to Kendra index
        For use by Person 3 (Vector Search replacement)
        """
        try:
            kendra_client = self.aws_config.get_service_client('kendra')
            
            response = kendra_client.batch_put_document(
                IndexId=index_id,
                Documents=documents
            )
            
            # Track costs
            self.aws_config.track_cost('kendra', 'document', len(documents))
            
            self.logger.info(f"✅ Kendra document upload successful: {len(documents)} documents")
            return {'success': True, 'response': response}
            
        except Exception as e:
            self.logger.error(f"❌ Kendra document upload failed: {e}")
            return {'error': str(e), 'success': False}
    
    def safe_kendra_query(self, index_id: str, query_text: str, k: int = 10, 
                         attribute_filter: Dict = None) -> Dict:
        """
        Query Kendra index for medical documents
        For use by Person 3 (Vector Search replacement)
        """
        try:
            kendra_client = self.aws_config.get_service_client('kendra')
            
            query_params = {
                'IndexId': index_id,
                'QueryText': query_text,
                'PageSize': k
            }
            
            if attribute_filter:
                query_params['AttributeFilter'] = attribute_filter
            
            response = kendra_client.query(**query_params)
            
            # Track costs
            self.aws_config.track_cost('kendra', 'query', 1)
            
            self.logger.info(f"✅ Kendra query successful: {len(response.get('ResultItems', []))} results")
            return {'success': True, 'results': response}
            
        except Exception as e:
            self.logger.error(f"❌ Kendra query failed: {e}")
            return {'error': str(e), 'success': False}

def setup_aws_environment() -> tuple[AWSConfigManager, AWSUtilities]:
    """
    Initialize AWS environment for the entire team
    Call this once at the start of your application
    """
    try:
        # Initialize AWS configuration
        aws_config = AWSConfigManager()
        
        # Create utilities
        aws_utils = AWSUtilities(aws_config)
        
        # Create S3 bucket for team use (optional)
        bucket_name = f"medical-hackathon-{int(time.time())}"
        aws_config.create_s3_bucket_if_needed(bucket_name)
        
        logger.info("🎉 AWS environment setup complete!")
        logger.info(f"💰 Current budget status: {aws_config.get_cost_summary()}")
        
        return aws_config, aws_utils
        
    except Exception as e:
        logger.error(f"❌ AWS environment setup failed: {e}")
        raise

def test_aws_setup():
    """
    Test function - run this to verify AWS setup is working
    """
    print("Testing AWS setup...")

    try:
        aws_config, aws_utils = setup_aws_environment()

        # Test basic text analysis
        test_response = aws_utils.safe_bedrock_call("Hello, this is a test medical query about diabetes.")
        print(f"Bedrock test response: {test_response[:100]}...")

        # Test cost tracking
        cost_summary = aws_config.get_cost_summary()
        print(f"Cost tracking working: {cost_summary}")

        print("AWS setup test PASSED!")
        return True

    except Exception as e:
        print(f"AWS setup test FAILED: {e}")
        return False

if __name__ == "__main__":
    """
    Run this file directly to test AWS setup
    """
    
    print("AWS Infrastructure Setup - Person 1")
    print("=" * 50)

    # Test AWS setup
    if test_aws_setup():
        print("\nAWS environment ready for team!")
        print("Share aws_config.py with all team members")
        print("Next: Person 2 can start Textract integration")
        print("Next: Person 3 can start vector search setup")
        print("Next: Person 4 can start document processing")
        print("Next: Person 5 can start Gradio interface")
    else:
        print("\nAWS setup needs troubleshooting")
        print("Check:")
        print("   1. AWS credentials: aws configure")
        print("   2. Internet connection")
        print("   3. AWS service permissions")
        print("   4. Bedrock Nova Lite model access enabled")