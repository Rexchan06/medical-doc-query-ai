#!/usr/bin/env python3
"""
Check Kendra index configuration and data sources
"""

import boto3
import os
from dotenv import load_dotenv
load_dotenv()

def check_kendra_config():
    try:
        region = os.environ.get('AWS_DEFAULT_REGION', 'ap-southeast-1')
        index_id = os.environ.get('KENDRA_INDEX_ID')

        kendra = boto3.client('kendra', region_name=region)

        print(f"Checking Kendra index: {index_id}")

        # 1. Get index details
        index_info = kendra.describe_index(Id=index_id)
        print(f"Index Name: {index_info['Name']}")
        print(f"Status: {index_info['Status']}")
        print(f"Edition: {index_info.get('Edition', 'Unknown')}")

        # 2. Check data sources
        data_sources = kendra.list_data_sources(IndexId=index_id)
        sources = data_sources.get('SummaryItems', [])
        print(f"\\nData Sources: {len(sources)}")

        if sources:
            print("Your index has data sources - might need S3 approach:")
            for source in sources:
                print(f"  - {source['Name']} ({source['Type']}) - {source['Status']}")
        else:
            print("No data sources - direct upload should work")

        # 3. Check recent documents
        try:
            # Try to see if any documents exist
            query_result = kendra.query(
                IndexId=index_id,
                QueryText="test",
                PageSize=1
            )
            doc_count = len(query_result.get('ResultItems', []))
            print(f"\\nDocuments found in index: {doc_count}")

        except Exception as e:
            print(f"\\nCould not query index: {e}")

        return True

    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    check_kendra_config()