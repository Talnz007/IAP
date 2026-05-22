"""
Keep-alive script to prevent Supabase free tier database from pausing.
Runs a lightweight SELECT query.
"""
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

def main():
    print("="*50)
    print(f"Running Supabase Keep-Alive: {datetime.now().isoformat()}")
    print("="*50)
    
    load_dotenv()
    
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        print("Error: SUPABASE_URL and SUPABASE_KEY must be set.")
        sys.exit(1)
        
    try:
        from supabase import create_client, Client
        client: Client = create_client(supabase_url, supabase_key)
        
        # Execute a lightweight read to keep the database active
        # We limit to 1 to minimize data transfer
        response = client.table('aqi_features').select('id').limit(1).execute()
        
        print(f"✓ Successfully pinged Supabase database. Response length: {len(response.data)}")
        sys.exit(0)
    except Exception as e:
        print(f"✗ Failed to ping Supabase: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
