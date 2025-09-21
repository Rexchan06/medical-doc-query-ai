#!/usr/bin/env python3
"""
Test Kendra index creation step by step
"""

import boto3
import json
import time
import os
from dotenv import load_dotenv
load_dotenv()

def test_kendra_creation():
    """Test each step of Kendra index creation"""

    try:
        # Step 1: Check AWS credentials and account
        print("Step 1: Checking AWS credentials...")
        sts_client = boto3.client('sts', region_name='us-east-1')
        account_id = sts_client.get_caller_identity()['Account']
        print(f"SUCCESS: AWS Account ID: {account_id}")

        # Step 2: Check IAM role
        print("\nStep 2: Checking IAM role...")
        iam_client = boto3.client('iam', region_name='us-east-1')
        role_name = "KendraServiceRole"

        try:
            role_response = iam_client.get_role(RoleName=role_name)
            print(f"SUCCESS: Role {role_name} exists")
            print(f"Role ARN: {role_response['Role']['Arn']}")

            # Check trust policy
            trust_policy = role_response['Role']['AssumeRolePolicyDocument']
            print(f"Trust policy: {json.dumps(trust_policy, indent=2)}")

        except iam_client.exceptions.NoSuchEntityException:
            print(f"ERROR: Role {role_name} does not exist")
            return False

        # Step 3: Test Kendra service availability
        print("\nStep 3: Testing Kendra service...")
        kendra_client = boto3.client('kendra', region_name='us-east-1')

        # Try to list indices (this tests if Kendra is available)
        list_response = kendra_client.list_indices()
        print(f"SUCCESS: Kendra service is available")
        print(f"Current indices: {len(list_response.get('IndexConfigurationSummaryItems', []))}")

        # Step 4: Try to create index
        print(f"\nStep 4: Attempting to create Kendra index...")
        role_arn = f"arn:aws:iam::{account_id}:role/{role_name}"

        try:
            response = kendra_client.create_index(
                Name="medical-documents-index-test",
                Description="Test index for medical document analysis",
                RoleArn=role_arn,
                Edition='DEVELOPER_EDITION'
            )

            index_id = response['Id']
            print(f"SUCCESS: Created Kendra index with ID: {index_id}")
            return index_id

        except Exception as create_error:
            print(f"ERROR creating index: {create_error}")

            # If role is not ready, try waiting and retry once
            if "role" in str(create_error).lower():
                print("Waiting 30 seconds for role to propagate...")
                time.sleep(30)

                try:
                    response = kendra_client.create_index(
                        Name="medical-documents-index-retry",
                        Description="Retry test index for medical document analysis",
                        RoleArn=role_arn,
                        Edition='DEVELOPER_EDITION'
                    )

                    index_id = response['Id']
                    print(f"SUCCESS (retry): Created Kendra index with ID: {index_id}")
                    return index_id

                except Exception as retry_error:
                    print(f"ERROR (retry): {retry_error}")
                    return False

            return False

    except Exception as e:
        print(f"FATAL ERROR: {e}")
        return False

if __name__ == "__main__":
    result = test_kendra_creation()
    if result:
        print(f"\nFINAL SUCCESS: Kendra index created with ID: {result}")
    else:
        print(f"\nFINAL FAILURE: Could not create Kendra index")