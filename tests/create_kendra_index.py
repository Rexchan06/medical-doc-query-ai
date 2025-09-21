#!/usr/bin/env python3
"""
Script to create a Kendra index for the medical app
"""

import boto3
import json
import os
from dotenv import load_dotenv
load_dotenv()

def create_kendra_index():
    """Create a new Kendra index"""
    try:
        print("Creating Kendra index...")

        # Get AWS account ID
        sts_client = boto3.client('sts', region_name='us-east-1')
        account_id = sts_client.get_caller_identity()['Account']
        print(f"AWS Account ID: {account_id}")

        # Create IAM role first if it doesn't exist
        iam_client = boto3.client('iam', region_name='us-east-1')

        role_name = "KendraServiceRole"
        role_arn = f"arn:aws:iam::{account_id}:role/{role_name}"

        try:
            # Check if role exists
            iam_client.get_role(RoleName=role_name)
            print(f"IAM role {role_name} already exists")
        except iam_client.exceptions.NoSuchEntityException:
            print(f"Creating IAM role {role_name}...")

            trust_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"Service": "kendra.amazonaws.com"},
                        "Action": "sts:AssumeRole"
                    }
                ]
            }

            iam_client.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Description="Service role for Amazon Kendra"
            )

            # Attach the managed policy for Kendra
            iam_client.attach_role_policy(
                RoleName=role_name,
                PolicyArn="arn:aws:iam::aws:policy/AmazonKendraFullAccess"
            )
            print(f"Created IAM role: {role_arn}")

        # Create Kendra index
        kendra_client = boto3.client('kendra', region_name='us-east-1')

        response = kendra_client.create_index(
            Name="medical-documents-index",
            Description="Index for medical document analysis",
            RoleArn=role_arn,
            Edition='DEVELOPER_EDITION'
        )

        index_id = response['Id']
        print(f"SUCCESS: Created Kendra index with ID: {index_id}")

        # Update .env file
        with open('.env', 'r') as f:
            content = f.read()

        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('KENDRA_INDEX_ID='):
                lines[i] = f'KENDRA_INDEX_ID="{index_id}"'
                break

        with open('.env', 'w') as f:
            f.write('\n'.join(lines))

        print(f"Updated .env file with new index ID: {index_id}")
        print("\nIndex creation initiated. It may take a few minutes to become active.")
        return True

    except Exception as e:
        print(f"ERROR: Failed to create Kendra index: {e}")
        print("\nThis could be due to:")
        print("1. Insufficient IAM permissions")
        print("2. Kendra not available in your region")
        print("3. AWS service limits")
        print("\nYou may need to create the index manually in the AWS console.")
        return False

if __name__ == "__main__":
    create_kendra_index()