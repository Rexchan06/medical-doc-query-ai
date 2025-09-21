#!/usr/bin/env python3
"""
Quick test script for Bedrock Nova Lite connectivity
"""

import boto3
import json
from aws_config import setup_aws_environment

def test_nova_lite():
    """Test Nova Lite text and vision capabilities"""
    print("Testing Bedrock Nova Lite...")

    try:
        # Initialize AWS
        aws_config, aws_utils = setup_aws_environment()

        # Test 1: Simple text query
        print("\n1. Testing text query...")
        response = aws_utils.safe_bedrock_call("What is diabetes? Give a brief medical definition.")
        print(f"Response: {response}")

        # Test 2: Check if we can make a direct API call
        print("\n2. Testing direct API call...")
        bedrock_client = aws_config.get_service_client('bedrock')

        # Use the correct converse API for Nova Lite
        response = bedrock_client.converse(
            modelId='us.amazon.nova-lite-v1:0',
            messages=[{'role': 'user', 'content': [{'text': 'Hello, are you Nova Lite?'}]}],
            inferenceConfig={'temperature': 0.1, 'maxTokens': 100}
        )

        print(f"Direct API response structure: {response.keys()}")

        if 'output' in response:
            answer = response['output']['message']['content'][0]['text']
            print(f"Direct API answer: {answer}")
        else:
            print(f"Unexpected response format: {response}")

        print("\nNova Lite test completed successfully!")
        return True

    except Exception as e:
        print(f"Nova Lite test failed: {e}")
        return False

if __name__ == "__main__":
    test_nova_lite()