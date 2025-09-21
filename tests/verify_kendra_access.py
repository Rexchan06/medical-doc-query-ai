#!/usr/bin/env python3
"""
Verify Kendra index access and permissions
"""

import boto3
import os
from dotenv import load_dotenv
load_dotenv()

def verify_kendra_access():
    """Test if we can actually access and use the Kendra index"""
    try:
        # Get region and index ID from env
        region = os.environ.get('AWS_DEFAULT_REGION', 'ap-southeast-1')
        index_id = os.environ.get('KENDRA_INDEX_ID')

        print(f"Testing Kendra access...")
        print(f"Region: {region}")
        print(f"Index ID: {index_id}")

        # Create Kendra client
        kendra_client = boto3.client('kendra', region_name=region)

        # Test 1: List indices (basic connectivity)
        print("\n1. Testing basic connectivity...")
        indices = kendra_client.list_indices()['IndexConfigurationSummaryItems']
        print(f"Found {len(indices)} indices")
        for idx in indices:
            print(f"  - {idx['Id']}: {idx['Name']} ({idx['Status']})")

        # Test 2: Describe specific index
        print(f"\n2. Testing access to your index: {index_id}")
        try:
            index_details = kendra_client.describe_index(Id=index_id)
            print(f"✅ Index exists: {index_details['Name']}")
            print(f"✅ Status: {index_details['Status']}")
            print(f"✅ Role ARN: {index_details.get('RoleArn', 'Not shown')}")
        except Exception as e:
            print(f"❌ Cannot access index: {e}")
            return False

        # Test 3: Try a simple document upload
        print(f"\n3. Testing document upload permissions...")
        test_doc = {
            'Id': 'test-doc-123',
            'Title': 'Test Document',
            'Blob': b'This is a test document for verification.',
            'ContentType': 'PLAIN_TEXT'
        }

        try:
            response = kendra_client.batch_put_document(
                IndexId=index_id,
                Documents=[test_doc]
            )
            print("✅ Document upload test successful")

            # Clean up test document
            try:
                kendra_client.batch_delete_document(
                    IndexId=index_id,
                    DocumentIdList=['test-doc-123']
                )
                print("✅ Test document cleaned up")
            except:
                print("⚠️ Could not clean up test document (not critical)")

        except Exception as e:
            print(f"❌ Document upload failed: {e}")

            # Check if it's a permissions issue
            if "AccessDenied" in str(e):
                print("💡 This looks like a permissions issue")
            elif "ValidationException" in str(e):
                print("💡 This looks like a data format issue")
            else:
                print("💡 Unknown issue type")

            return False

        print("\n🎉 All tests passed! Kendra index is accessible and writable.")
        return True

    except Exception as e:
        print(f"❌ Critical error: {e}")
        return False

if __name__ == "__main__":
    success = verify_kendra_access()
    if not success:
        print("\n🔧 Troubleshooting steps:")
        print("1. Check if the index exists in the correct region")
        print("2. Verify IAM permissions for your AWS credentials")
        print("3. Ensure the index status is ACTIVE")
        print("4. Check if there are any access policies on the index")