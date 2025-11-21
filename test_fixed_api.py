import requests
import json

def test_url(url, expected_risk):
    """Test a URL and show results"""
    print(f"\n{'='*60}")
    print(f"Testing: {url}")
    print(f"Expected: {expected_risk}")
    print('='*60)
    
    try:
        response = requests.post(
            "http://localhost:8000/analyze",
            json={"url": url},
            timeout=15
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Phishing Score: {result['phishing_score']}%")
            print(f"✅ Risk Level: {result['risk_level']}")
            print(f"✅ Analysis Time: {result['analysis_time']}s")
            
            # Show some key features
            features = result['features']
            print(f"\n📊 Key Features:")
            print(f"   - URL Length: {features.get('url_length')}")
            print(f"   - Has HTTPS: {features.get('is_https')}")
            print(f"   - Suspicious Keywords: {features.get('suspicious_keyword_count')}")
            print(f"   - Entropy: {features.get('entropy', 'N/A')}")
            print(f"   - Has IP: {features.get('has_ip')}")
            
            return result['risk_level']
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Connection Error: {e}")

# TEST 1: Legitimate Sites (Should be LOW)
print("\n🟢 TESTING LEGITIMATE SITES (Should be LOW risk)")
test_url("https://www.google.com", "LOW")
test_url("https://github.com", "LOW")
test_url("https://www.amazon.com", "LOW")

# TEST 2: Suspicious Sites (Should be HIGH)
print("\n\n🔴 TESTING SUSPICIOUS PATTERNS (Should be HIGH risk)")
test_url("http://secure-paypal-verify.suspicious.com/login", "HIGH")
test_url("http://192.168.1.1/banking/login.php", "HIGH")
test_url("http://account-verify-urgent.tk/signin", "HIGH")

# TEST 3: Your choice
print("\n\n🔵 TEST YOUR OWN URL:")
custom_url = input("Enter a URL to test: ")
if custom_url:
    test_url(custom_url, "UNKNOWN")

print("\n" + "="*60)
print("✅ Testing Complete!")
print("="*60)