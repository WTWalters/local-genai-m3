"""
Test semantic search API endpoints
"""

import asyncio
import httpx
import json

async def test_semantic_search():
    """Test the semantic search endpoint."""
    
    base_url = "http://localhost:8000"
    
    # Test data - search request
    search_payload = {
        "query": "knee pain and arthroscopy",
        "search_types": ["patients", "notes", "procedures", "imaging"],
        "limit": 5,
        "similarity_threshold": 0.1
    }
    
    async with httpx.AsyncClient() as client:
        try:
            print("🔍 Testing semantic search...")
            
            # Test semantic search endpoint
            response = await client.post(
                f"{base_url}/api/v1/search/semantic",
                json=search_payload,
                timeout=30.0
            )
            
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Search successful!")
                print(f"Query: {data['query']}")
                print(f"Total results: {data['total_results']}")
                print(f"Search time: {data['search_time_ms']}ms")
                
                print("\n📋 Results:")
                for i, result in enumerate(data['results'][:3], 1):  # Show first 3 results
                    print(f"\n{i}. {result['title']} (Score: {result['score']:.3f})")
                    print(f"   Type: {result['type']}")
                    print(f"   Content: {result['content'][:100]}...")
                    if result['patient_name']:
                        print(f"   Patient: {result['patient_name']}")
                
            else:
                print(f"❌ Search failed: {response.text}")
                
        except Exception as e:
            print(f"❌ Error testing semantic search: {e}")
        
        try:
            print("\n🔍 Testing search suggestions...")
            
            # Test search suggestions endpoint
            response = await client.get(
                f"{base_url}/api/v1/search/suggestions",
                params={"query": "knee", "limit": 5},
                timeout=10.0
            )
            
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Suggestions successful!")
                print("Suggestions:", data['suggestions'])
            else:
                print(f"❌ Suggestions failed: {response.text}")
                
        except Exception as e:
            print(f"❌ Error testing suggestions: {e}")

if __name__ == "__main__":
    asyncio.run(test_semantic_search())