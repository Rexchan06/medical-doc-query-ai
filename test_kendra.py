#!/usr/bin/env python3
"""
Simple Kendra test script
"""

import os
import sys

def test_kendra_basic():
    """Test basic Kendra connectivity"""
    print("Testing Kendra setup...")

    try:
        # Import AWS configuration
        from aws_config import setup_aws_environment
        print("AWS config imported successfully")

        # Initialize AWS
        aws_config, aws_utils = setup_aws_environment()
        print("AWS environment setup complete")

        # Test Kendra connectivity
        kendra_client = aws_config.get_service_client('kendra')
        print("Kendra client obtained")

        # List indices
        response = kendra_client.list_indices()
        indices = response.get('IndexConfigurationSummaryItems', [])

        print(f"Found {len(indices)} Kendra indices:")
        for idx in indices:
            print(f"  - ID: {idx['Id']}")
            print(f"    Name: {idx['Name']}")
            print(f"    Status: {idx['Status']}")

        if len(indices) == 0:
            print("No Kendra indices found. You need to create one first.")
            return False

        # Test with first available index
        test_index_id = indices[0]['Id']
        print(f"Testing query with index: {test_index_id}")

        # Simple query test
        query_response = kendra_client.query(
            IndexId=test_index_id,
            QueryText="test query",
            PageSize=5
        )

        results = query_response.get('ResultItems', [])
        print(f"Query returned {len(results)} results")

        print("Kendra test PASSED!")
        return True

    except Exception as e:
        print(f"Kendra test FAILED: {e}")
        return False

if __name__ == "__main__":
    success = test_kendra_basic()
    sys.exit(0 if success else 1)