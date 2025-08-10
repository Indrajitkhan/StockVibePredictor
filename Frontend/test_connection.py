#!/usr/bin/env python3
"""
Quick test to verify frontend-backend connectivity
Run this after starting both servers to test the connection
"""

import requests
import json

def test_backend_connection():
    """Test if backend API is accessible"""
    try:
        # Test basic API endpoint
        response = requests.get("http://localhost:8000/api/redis-check/", timeout=5)
        
        if response.status_code == 200:
            print("✅ Backend API is accessible!")
            print(f"Response: {response.json()}")
            return True
        else:
            print(f"❌ Backend API returned status code: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend. Make sure Django server is running on port 8000")
        return False
    except requests.exceptions.Timeout:
        print("❌ Backend request timed out")
        return False
    except Exception as e:
        print(f"❌ Error connecting to backend: {str(e)}")
        return False

def test_prediction_endpoint():
    """Test the main prediction endpoint"""
    try:
        test_data = {"ticker": "AAPL"}
        response = requests.post(
            "http://localhost:8000/api/predict/", 
            json=test_data,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Prediction endpoint is working!")
            print(f"Sample prediction for AAPL: {data.get('prediction', {}).get('direction', 'N/A')}")
            return True
        else:
            print(f"❌ Prediction endpoint returned status code: {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing prediction endpoint: {str(e)}")
        return False

if __name__ == "__main__":
    print("🔍 Testing StockVibePredictor Connection...")
    print("=" * 50)
    
    backend_ok = test_backend_connection()
    prediction_ok = False
    
    if backend_ok:
        print("\n" + "=" * 50)
        prediction_ok = test_prediction_endpoint()
    
    print("\n" + "=" * 50)
    print("📊 Connection Test Summary:")
    print(f"Backend API: {'✅ Working' if backend_ok else '❌ Failed'}")
    print(f"Prediction API: {'✅ Working' if prediction_ok else '❌ Failed'}")
    
    if backend_ok and prediction_ok:
        print("\n🎉 All connections are working! Your frontend and backend are properly connected.")
        print("\nTo start your application:")
        print("1. Backend: cd Backend && python manage.py runserver")
        print("2. Frontend: cd Frontend && npm start")
    else:
        print("\n⚠️ Some connections failed. Check if your Django server is running.")
