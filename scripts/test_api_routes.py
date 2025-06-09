#!/usr/bin/env python3
"""
Test script to verify API routes are properly organized and accessible.
"""

import os
import sys
import time
import requests

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

BASE_URL = "http://localhost:8000"

def test_endpoints():
    """Test all the new model endpoints to ensure they're accessible."""
    
    print("🔍 Testing API Route Organization...")
    print("=" * 50)
    
    # Test system endpoints
    endpoints_to_test = [
        ("/health", "System Health"),
        ("/", "Root Endpoint"),
        ("/docs", "API Documentation"),
        ("/api/v1/model/health", "Model Health"),
        ("/api/v1/model/info", "Model Information"),
        ("/api/v1/model/test/likelihood", "Test Likelihood"),
        ("/api/v1/model/test/recommendations", "Test Recommendations"),
        (
            "/api/v1/model/embeddings/student/5266e49a-5de0-43c2-b840-a66f4641f30d", 
            "Student Embedding"
        ),
    ]
    
    results = []
    
    for endpoint, description in endpoints_to_test:
        try:
            print(f"Testing {description}: {endpoint}")
            response = requests.get(f"{BASE_URL}{endpoint}", timeout=10)
            
            if response.status_code == 200:
                print(f"  ✅ SUCCESS: {response.status_code}")
                if endpoint == "/api/v1/model/info":
                    data = response.json()
                    print(f"     Architecture: {data.get('architecture', 'N/A')}")
                    print(f"     Status: {data.get('status', 'N/A')}")
                elif endpoint == "/api/v1/model/health":
                    data = response.json() 
                    print(f"     Model Health: {data.get('status', 'N/A')}")
                elif endpoint == "/health":
                    data = response.json()
                    print(f"     System Status: {data.get('status', 'N/A')}")
                    print(f"     Model Available: {data.get('model_health', {}).get('status', 'N/A')}")
                
                results.append(("✅", endpoint, response.status_code))
            else:
                error_text = response.text[:100] if response.text else "No content"
                print(f"  ❌ FAILED: {response.status_code} - {error_text}")
                results.append(("❌", endpoint, response.status_code))
                
        except requests.exceptions.ConnectionError:
            print(f"  🔌 CONNECTION ERROR: Server not running at {BASE_URL}")
            results.append(("🔌", endpoint, "No Connection"))
        except Exception as e:
            print(f"  ⚠️ ERROR: {str(e)}")
            results.append(("⚠️", endpoint, str(e)))
        
        print()
        time.sleep(0.5)  # Brief pause between requests
    
    # Summary
    print("📊 SUMMARY")
    print("=" * 50)
    
    success_count = sum(1 for status, _, _ in results if status == "✅")
    total_count = len(results)
    
    print(f"Total Endpoints Tested: {total_count}")
    print(f"Successful: {success_count}")
    print(f"Failed: {total_count - success_count}")
    print()
    
    for status, endpoint, code in results:
        print(f"{status} {endpoint:<50} {code}")
    
    if success_count == total_count:
        print("\n🎉 All endpoints are working correctly!")
        print("✅ API routes are properly organized and accessible to frontend")
    else:
        print(f"\n⚠️ {total_count - success_count} endpoints need attention")
        
    return success_count == total_count

if __name__ == "__main__":
    print("🚀 API Route Testing Script")
    print(f"📍 Testing against: {BASE_URL}")
    print(f"⏰ Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    success = test_endpoints()
    
    print("\n" + "=" * 50)
    if success:
        print("🎯 RESULT: All API routes working correctly!")
        print("🔗 Frontend can now access all model endpoints")
    else:
        print("🔧 RESULT: Some endpoints need attention")
        print("💡 Make sure the FastAPI server is running: uvicorn api.main:app --reload")
    
    print(f"⏰ Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}") 