import requests
import json

def test(url):
    response = requests.post(
        "http://localhost:8000/analyze",
        json={"url": url}
    )
    data = response.json()
    
    print(f"\nURL: {url}")
    print(f"Prediction: {data['prediction']}")
    print(f"Risk: {data['risk_level']}")
    print(f"Phishing Probability: {data['features']['phishing_probability']}%")
    print("-" * 50)

# Test different URLs
test("https://www.google.com")
test("http://secure-paypal.xyz/login")
test("https://fake-domain-12345.void")
test("https://www.microsoft.com")
test("https://www.amazon.in")
test("http://paypal-login-verification.com/login")
test("http://secure-account-update.net/login")
test("https://bit.ly/3xF4kK")
test("http://login-update.xyz")
test("http://185.244.25.12/login")
test("http://%70%61%79%70%61%6c-login.com/auth")
test("https://www.geeksforgeeks.org")



# URL: https://www.google.com
# Prediction: LEGITIMATE (Trusted Domain)
# Risk: SAFE
# Phishing Probability: 5.0%
# --------------------------------------------------

# URL: http://secure-paypal.xyz/login
# Prediction: SUSPICIOUS (Domain Unreachable)
# Risk: HIGH
# Phishing Probability: 90.0%
# --------------------------------------------------

# URL: https://fake-domain-12345.void
# Prediction: SUSPICIOUS (Domain Unreachable)
# Risk: HIGH
# Phishing Probability: 90.0%
# --------------------------------------------------

# URL: https://www.microsoft.com
# Prediction: LEGITIMATE (Trusted Domain)
# Risk: SAFE
# Phishing Probability: 5.0%
# --------------------------------------------------

# URL: https://www.amazon.in
# Prediction: LEGITIMATE
# Risk: SAFE
# Phishing Probability: 6.29%
# --------------------------------------------------

# URL: http://paypal-login-verification.com/login
# Prediction: SUSPICIOUS (Domain Unreachable)
# Risk: HIGH
# Phishing Probability: 90.0%
# --------------------------------------------------

# URL: http://secure-account-update.net/login
# Prediction: SUSPICIOUS (Domain Unreachable)
# Risk: HIGH
# Phishing Probability: 90.0%
# --------------------------------------------------

# URL: https://bit.ly/3xF4kK
# Prediction: LEGITIMATE (Page Not Accessible)
# Risk: MEDIUM
# Phishing Probability: 60.0%
# --------------------------------------------------

# URL: http://login-update.xyz
# Prediction: SUSPICIOUS (Domain Unreachable)
# Risk: HIGH
# Phishing Probability: 90.0%
# --------------------------------------------------

# URL: http://185.244.25.12/login
# Prediction: PHISHING
# Risk: CRITICAL
# Phishing Probability: 86.59%
# --------------------------------------------------

# URL: http://%70%61%79%70%61%6c-login.com/auth
# Prediction: SUSPICIOUS (Domain Unreachable)
# Risk: HIGH
# Phishing Probability: 90.0%
# --------------------------------------------------

# URL: https://www.geeksforgeeks.org
# Prediction: LEGITIMATE
# Risk: SAFE
# Phishing Probability: 9.65%
# --------------------------------------------------