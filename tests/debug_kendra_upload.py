#!/usr/bin/env python3
"""
Debug Kendra BatchPutDocument to find why it fails silently
"""

import boto3
import json
import os
from dotenv import load_dotenv
load_dotenv()

def debug_kendra_upload():
    """Debug step by step why Kendra upload isn't working"""

    region = os.environ.get('AWS_DEFAULT_REGION', 'ap-southeast-1')
    index_id = os.environ.get('KENDRA_INDEX_ID')

    print(f"=== KENDRA UPLOAD DEBUG ===")
    print(f"Region: {region}")
    print(f"Index ID: {index_id}")

    kendra = boto3.client('kendra', region_name=region)

    # Step 1: Test minimal document upload
    print(f"\\n1. Testing MINIMAL document upload...")

    minimal_doc = {
        'Id': 'debug-test-001',
        'Title': 'Debug Test Document',
        'Blob': b'This is a simple test document.',
        'ContentType': 'PLAIN_TEXT'
    }

    print(f"Document structure:")
    print(f"  - Id: {minimal_doc['Id']}")
    print(f"  - Title: {minimal_doc['Title']}")
    print(f"  - Blob length: {len(minimal_doc['Blob'])} bytes")
    print(f"  - ContentType: {minimal_doc['ContentType']}")

    try:
        print(f"\\nCalling BatchPutDocument...")
        response = kendra.batch_put_document(
            IndexId=index_id,
            Documents=[minimal_doc]
        )

        print(f"API Response:")
        print(f"  - HTTP Status: SUCCESS")
        print(f"  - Response: {json.dumps(response, indent=2, default=str)}")

        # Check for any failed documents in response
        failed_docs = response.get('FailedDocuments', [])
        if failed_docs:
            print(f"\\nFAILED DOCUMENTS FOUND:")
            for fail in failed_docs:
                print(f"  - Doc ID: {fail.get('Id')}")
                print(f"  - Error Code: {fail.get('ErrorCode')}")
                print(f"  - Error Message: {fail.get('ErrorMessage')}")
        else:
            print(f"\\n✅ No failed documents reported")

    except Exception as e:
        print(f"\\n❌ BatchPutDocument FAILED: {e}")
        return False

    # Step 2: Wait and check if document appears
    print(f"\\n2. Checking if document was actually stored...")

    import time
    print(f"Waiting 10 seconds for indexing...")
    time.sleep(10)

    try:
        # Try to find our test document
        query_response = kendra.query(
            IndexId=index_id,
            QueryText="debug test document simple",
            PageSize=10
        )

        results = query_response.get('ResultItems', [])
        print(f"Query results: {len(results)} documents found")

        if results:
            for i, result in enumerate(results):
                print(f"  Result {i+1}:")
                print(f"    - Document ID: {result.get('DocumentId')}")
                print(f"    - Title: {result.get('DocumentTitle')}")
                print(f"    - Excerpt: {result.get('DocumentExcerpt', {}).get('Text', '')[:100]}...")
        else:
            print(f"❌ NO DOCUMENTS FOUND - Upload failed silently!")

    except Exception as e:
        print(f"❌ Query failed: {e}")

    # Step 3: Check index statistics
    print(f"\\n3. Checking index statistics...")
    try:
        index_stats = kendra.describe_index(Id=index_id)
        doc_count_stats = index_stats.get('DocumentMetadataConfigurations', [])
        print(f"Index status: {index_stats.get('Status')}")

        # Try to get document count another way
        list_response = kendra.query(
            IndexId=index_id,
            QueryText="*",  # Wildcard search
            PageSize=1
        )
        total_results = list_response.get('TotalNumberOfResults', 0)
        print(f"Total documents in index: {total_results}")

    except Exception as e:
        print(f"Could not get index stats: {e}")

    # Step 4: Test permissions by trying to delete
    print(f"\\n4. Testing delete permissions...")
    try:
        delete_response = kendra.batch_delete_document(
            IndexId=index_id,
            DocumentIdList=['debug-test-001']
        )
        print(f"Delete test: SUCCESS (confirms write permissions)")
        print(f"Delete response: {delete_response}")
    except Exception as e:
        print(f"Delete test FAILED: {e}")
        if "AccessDenied" in str(e):
            print(f"❌ PERMISSION ISSUE: You can read but not write to this index")

    print(f"\\n=== DEBUG COMPLETE ===")

if __name__ == "__main__":
    debug_kendra_upload()