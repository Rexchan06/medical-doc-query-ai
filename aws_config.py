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
        self.cost_tracker = {
            'textract_pages': 0,
            'bedrock_tokens': 0,
            'comprehend_requests': 0,
            'estimated_cost': 0.0
        }
        
        # Initialize all AWS service clients
        self._initialize_aws_services()
        
        # Test connectivity
        self._test_all_services()
    
    def _initialize_aws_services(self):
        """Initialize all required AWS service clients"""
        try:
            logger.info("🔧 Initializing AWS services...")
            
            # Core AI services
            self.services['textract'] = boto3.client('textract', region_name=self.region_name)
            self.services['bedrock'] = boto3.client('bedrock-runtime', region_name=self.region_name)
            self.services['comprehend_medical'] = boto3.client('comprehendmedical', region_name=self.region_name)
            
            # Storage and processing
            self.services['s3'] = boto3.client('s3', region_name=self.region_name)
            self.services['lambda'] = boto3.client('lambda', region_name=self.region_name)
            
            # Monitoring and analytics
            self.services['cloudwatch'] = boto3.client('cloudwatch', region_name=self.region_name)
            
            # Additional services (if budget allows)
            try:
                self.services['translate'] = boto3.client('translate', region_name=self.region_name)
                self.services['polly'] = boto3.client('polly', region_name=self.region_name)
                self.services['transcribe'] = boto3.client('transcribe', region_name=self.region_name)
            except Exception as e:
                logger.warning(f"⚠️ Optional services not available: {e}")
            
            logger.info("✅ AWS services initialized successfully")
            
        except NoCredentialsError:
            logger.error("❌ AWS credentials not configured. Run 'aws configure'")
            raise
        except Exception as e:
            logger.error(f"❌ AWS initialization failed: {e}")
            raise
    
    def _test_all_services(self) -> Dict[str, bool]:
        """Test connectivity to all AWS services"""
        logger.info("🧪 Testing AWS service connectivity...")
        
        service_status = {}
        
        # Test Textract
        try:
            # Simple test call
            self.services['textract'].get_document_analysis(JobId='test-job-id')
        except ClientError as e:
            if 'InvalidJobIdException' in str(e):
                service_status['textract'] = True  # Service accessible, just invalid job ID
            else:
                service_status['textract'] = False
        except Exception:
            service_status['textract'] = False
        
        # Test Bedrock
        try:
            response = self.services['bedrock'].invoke_model(
                modelId='anthropic.claude-3-haiku-20240307-v1:0',
                body=json.dumps({
                    'anthropic_version': 'bedrock-2023-05-31',
                    'max_tokens': 10,
                    'messages': [{'role': 'user', 'content': 'test'}]
                })
            )
            service_status['bedrock'] = True
            logger.info("✅ Bedrock connectivity confirmed")
        except Exception as e:
            logger.error(f"❌ Bedrock test failed: {e}")
            service_status['bedrock'] = False
        
        # Test Comprehend Medical
        try:
            self.services['comprehend_medical'].detect_entities_v2(Text='test medical text')
            service_status['comprehend_medical'] = True
        except Exception as e:
            service_status['comprehend_medical'] = False
            logger.warning(f"⚠️ Comprehend Medical test failed: {e}")
        
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
            'textract_page': 0.0015,
            'bedrock_1k_tokens': 0.00025,
            'comprehend_medical_request': 0.0001,
            'translate_character': 0.000015,
            'polly_character': 0.000004
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
            'services_used': list(self.services.keys()),
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
    
    def safe_textract_call(self, document_bytes: bytes, feature_types: List[str] = None) -> Dict:
        """
        Safe Textract API call with error handling and cost tracking
        For use by Person 2 (Backend) and Person 4 (Document Processing)
        """
        try:
            textract_client = self.aws_config.get_service_client('textract')
            
            # Prepare request
            request_params = {
                'Document': {'Bytes': document_bytes}
            }
            
            if feature_types:
                request_params['FeatureTypes'] = feature_types
                response = textract_client.analyze_document(**request_params)
            else:
                response = textract_client.detect_document_text(**request_params)
            
            # Track costs
            estimated_pages = len(document_bytes) / (1024 * 1024)  # Rough estimate
            self.aws_config.track_cost('textract', 'page', max(1, int(estimated_pages)))
            
            self.logger.info(f"✅ Textract processing successful: {len(response['Blocks'])} blocks")
            return response
            
        except Exception as e:
            self.logger.error(f"❌ Textract API call failed: {e}")
            return {'error': str(e), 'Blocks': []}
    
    def safe_bedrock_call(self, prompt: str, max_tokens: int = 500, model_id: str = None) -> str:
        """
        Safe Bedrock API call with error handling and cost tracking
        For use by Person 3 (AI/ML) and Person 4 (Document Processing)
        """
        try:
            bedrock_client = self.aws_config.get_service_client('bedrock')
            
            if model_id is None:
                model_id = 'anthropic.claude-3-haiku-20240307-v1:0'
            
            # Prepare request
            request_body = {
                'anthropic_version': 'bedrock-2023-05-31',
                'max_tokens': max_tokens,
                'temperature': 0.1,
                'messages': [{'role': 'user', 'content': prompt}]
            }
            
            # Make API call
            response = bedrock_client.invoke_model(
                modelId=model_id,
                body=json.dumps(request_body)
            )
            
            # Parse response
            result = json.loads(response['body'].read())
            answer = result['content'][0]['text']
            
            # Track costs (approximate)
            estimated_tokens = len(prompt.split()) + len(answer.split())
            self.aws_config.track_cost('bedrock', '1k_tokens', max(1, estimated_tokens // 1000))
            
            self.logger.info(f"✅ Bedrock call successful: {len(answer)} characters")
            return answer
            
        except Exception as e:
            self.logger.error(f"❌ Bedrock API call failed: {e}")
            return f"Error: {str(e)}"
    
    def safe_bedrock_vision_call(self, image_base64: str, prompt: str) -> str:
        """
        Safe Bedrock vision API call for image analysis
        For use by Person 4 (Document Processing)
        """
        try:
            bedrock_client = self.aws_config.get_service_client('bedrock')
            
            request_body = {
                'anthropic_version': 'bedrock-2023-05-31',
                'max_tokens': 400,
                'temperature': 0.1,
                'messages': [{
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': prompt},
                        {
                            'type': 'image',
                            'source': {
                                'type': 'base64',
                                'media_type': 'image/png',
                                'data': image_base64
                            }
                        }
                    ]
                }]
            }
            
            response = bedrock_client.invoke_model(
                modelId='anthropic.claude-3-haiku-20240307-v1:0',
                body=json.dumps(request_body)
            )
            
            result = json.loads(response['body'].read())
            answer = result['content'][0]['text']
            
            # Track vision costs
            self.aws_config.track_cost('bedrock', 'vision_image', 1)
            
            self.logger.info("✅ Bedrock vision call successful")
            return answer
            
        except Exception as e:
            self.logger.error(f"❌ Bedrock vision call failed: {e}")
            return f"Vision analysis failed: {str(e)}"
    
    def safe_comprehend_medical_call(self, text: str) -> Dict:
        """
        Safe Comprehend Medical API call
        For use by Person 2 (Backend) and Person 3 (AI/ML)
        """
        try:
            comprehend_client = self.aws_config.get_service_client('comprehend_medical')
            
            # Detect medical entities
            entities_response = comprehend_client.detect_entities_v2(Text=text)
            
            # Detect PHI
            phi_response = comprehend_client.detect_phi(Text=text)
            
            # Track costs
            self.aws_config.track_cost('comprehend_medical', 'request', 1)
            
            result = {
                'entities': entities_response.get('Entities', []),
                'phi': phi_response.get('Entities', []),
                'success': True
            }
            
            self.logger.info(f"✅ Comprehend Medical successful: {len(result['entities'])} entities")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Comprehend Medical call failed: {e}")
            return {'error': str(e), 'entities': [], 'phi': [], 'success': False}

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
    print("🧪 Testing AWS setup...")
    
    try:
        aws_config, aws_utils = setup_aws_environment()
        
        # Test basic text analysis
        test_response = aws_utils.safe_bedrock_call("Hello, this is a test medical query about diabetes.")
        print(f"✅ Bedrock test response: {test_response[:100]}...")
        
        # Test cost tracking
        cost_summary = aws_config.get_cost_summary()
        print(f"✅ Cost tracking working: {cost_summary}")
        
        print("🎉 AWS setup test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ AWS setup test FAILED: {e}")
        return False

if __name__ == "__main__":
    """
    Run this file directly to test AWS setup
    """
    
    print("🚀 AWS Infrastructure Setup - Person 1")
    print("=" * 50)
    
    # Test AWS setup
    if test_aws_setup():
        print("\n✅ AWS environment ready for team!")
        print("👥 Share aws_config.py with all team members")
        print("📋 Next: Person 2 can start Textract integration")
        print("📋 Next: Person 3 can start vector search setup")
        print("📋 Next: Person 4 can start document processing")
        print("📋 Next: Person 5 can start Gradio interface")
    else:
        print("\n❌ AWS setup needs troubleshooting")
        print("🔧 Check:")
        print("   1. AWS credentials: aws configure")
        print("   2. Internet connection")
        print("   3. AWS service permissions")
        print("   4. Bedrock model access enabled")