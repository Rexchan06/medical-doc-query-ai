#!/usr/bin/env python3
"""
Simple script to fix Kendra index configuration
"""

import boto3
import os
from dotenv import load_dotenv
load_dotenv()

def fix_kendra_index():
    """Fix the Kendra index configuration"""
    try:
        print("Checking Kendra indices...")

        # Create Kendra client
        kendra_client = boto3.client('kendra', region_name='us-east-1')

        # List available indices
        response = kendra_client.list_indices()
        indices = response.get('IndexConfigurationSummaryItems', [])

        print(f"Found {len(indices)} Kendra indices:")
        for idx in indices:
            print(f"  - ID: {idx['Id']}, Name: {idx['Name']}, Status: {idx['Status']}")

        if indices:
            # Use the first available index
            index_id = indices[0]['Id']
            print(f"\nUsing index: {index_id}")

            # Read current .env file
            with open('.env', 'r') as f:
                content = f.read()

            # Update the KENDRA_INDEX_ID line
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.startswith('KENDRA_INDEX_ID='):
                    lines[i] = f'KENDRA_INDEX_ID="{index_id}"'
                    break

            # Write back to .env file
            with open('.env', 'w') as f:
                f.write('\n'.join(lines))

            print(f"SUCCESS: Updated .env file with index ID: {index_id}")
            return True
        else:
            print("ERROR: No Kendra indices found. You need to create one first.")
            return False

    except Exception as e:
        print(f"ERROR: {e}")
        return False

if __name__ == "__main__":
    fix_kendra_index()