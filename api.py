# ======================================================================
# PHASE 4: FASTAPI BACKEND FOR PHISHING DETECTION (FIXED VERSION)
# ======================================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import requests
import re
import numpy as np
from urllib.parse import urlparse
import tldextract
import time

# ======================================================================
# INITIALIZE FASTAPI APP
# ======================================================================

app = FastAPI(
    title="Phishing URL Detector API",
    description="AI-powered phishing detection with 5 security layers",
    version="1.0.0"
)

# Allow CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================================================
# LOAD ML MODEL AND COMPONENTS
# ======================================================================

print("Loading ML model and components...")
try:
    model = joblib.load('phishing_model.pkl')
    scaler = joblib.load('feature_scaler.pkl')
    feature_names = joblib.load('feature_names.pkl')
    print(f"✅ Model loaded successfully! Expects {len(feature_names)} features")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    scaler = None
    feature_names = None

# ======================================================================
# REQUEST/RESPONSE MODELS
# ======================================================================

class URLRequest(BaseModel):
    url: str

class AnalysisResponse(BaseModel):
    url: str
    phishing_score: float
    risk_level: str
    features: dict
    redirect_chain: list
    final_destination: str
    page_info: dict
    analysis_time: float

# ======================================================================
# FEATURE EXTRACTION - MATCHES TRAINING EXACTLY
# ======================================================================

class URLFeatureExtractor:
    """Extract features from URLs - MUST MATCH TRAINING CODE"""
    
    def __init__(self):
        self.suspicious_keywords = [
            'verify', 'account', 'secure', 'update', 'login', 
            'banking', 'confirm', 'signin', 'ebayisapi', 'webscr',
            'password', 'credential', 'suspend', 'restricted', 'verify',
            'click', 'urgent', 'expire', 'suspended'
        ]
        
        self.shortening_services = [
            'bit.ly', 'goo.gl', 'shorte.st', 'go2l.ink', 'x.co', 
            'ow.ly', 't.co', 'tinyurl', 'tr.im', 'is.gd', 'cli.gs',
            'yfrog.com', 'migre.me', 'ff.im', 'tiny.cc', 'url4.eu',
            'twit.ac', 'su.pr', 'twurl.nl', 'snipurl.com', 'short.to'
        ]
    
    def extract_all_features(self, url):
        """Extract all features - EXACTLY AS IN TRAINING"""
        try:
            features = {}
            
            parsed = urlparse(url)
            ext = tldextract.extract(url)
            
            # 1. URL LENGTH FEATURES
            features['url_length'] = len(url)
            features['hostname_length'] = len(parsed.netloc)
            features['path_length'] = len(parsed.path)
            features['query_length'] = len(parsed.query)
            
            # 2. CHARACTER COUNT FEATURES (EXACT NAMES FROM TRAINING!)
            features['dot_count'] = url.count('.')
            features['hyphen_count'] = url.count('-')
            features['underscore_count'] = url.count('_')
            features['slash_count'] = url.count('/')
            features['questionmark_count'] = url.count('?')  # NOT question_count!
            features['equal_count'] = url.count('=')
            features['at_count'] = url.count('@')
            features['ampersand_count'] = url.count('&')
            features['exclamation_count'] = url.count('!')
            features['space_count'] = url.count(' ')
            features['tilde_count'] = url.count('~')
            features['comma_count'] = url.count(',')
            features['plus_count'] = url.count('+')
            features['asterisk_count'] = url.count('*')  # NOT star_count!
            features['hashtag_count'] = url.count('#')
            features['dollar_count'] = url.count('$')
            features['percent_count'] = url.count('%')
            
            # 3. SUBDOMAIN FEATURES
            subdomains = ext.subdomain.split('.') if ext.subdomain else []
            features['subdomain_count'] = len([s for s in subdomains if s])
            
            # 4. DOMAIN FEATURES
            domain = ext.domain
            features['domain_length'] = len(domain)
            features['domain_digit_count'] = sum(c.isdigit() for c in domain)
            features['domain_digit_ratio'] = features['domain_digit_count'] / max(len(domain), 1)
            
            # 5. PROTOCOL FEATURES
            features['is_https'] = 1 if parsed.scheme == 'https' else 0
            features['is_http'] = 1 if parsed.scheme == 'http' else 0
            
            # 6. IP ADDRESS DETECTION
            features['has_ip'] = self._has_ip_address(parsed.netloc)
            
            # 7. SUSPICIOUS KEYWORDS
            url_lower = url.lower()
            keyword_count = sum(1 for keyword in self.suspicious_keywords if keyword in url_lower)
            features['suspicious_keyword_count'] = keyword_count
            features['has_suspicious_keywords'] = 1 if keyword_count > 0 else 0
            
            # 8. URL SHORTENING SERVICE
            features['is_shortened'] = 1 if any(service in url_lower for service in self.shortening_services) else 0
            
            # 9. DOUBLE SLASH IN PATH
            features['double_slash_in_path'] = 1 if '//' in parsed.path else 0
            
            # 10. PREFIX/SUFFIX IN DOMAIN
            features['has_prefix_suffix'] = 1 if '-' in domain else 0
            
            # 11. SPECIAL CHARACTER RATIOS
            special_chars = sum(1 for c in url if not c.isalnum())
            features['special_char_ratio'] = special_chars / max(len(url), 1)
            
            # 12. DIGIT RATIO IN URL
            digit_count = sum(c.isdigit() for c in url)
            features['digit_ratio'] = digit_count / max(len(url), 1)
            
            # 13. LETTER RATIO IN URL
            letter_count = sum(c.isalpha() for c in url)
            features['letter_ratio'] = letter_count / max(len(url), 1)
            
            # 14. TLD LENGTH
            features['tld_length'] = len(ext.suffix)
            
            # 15. ENTROPY (CRITICAL - WAS MISSING!)
            features['entropy'] = self._calculate_entropy(url)
            
            # 16. CONSECUTIVE CONSONANTS (CRITICAL - WAS MISSING!)
            features['max_consecutive_consonants'] = self._max_consecutive_consonants(domain)
            
            # 17. VOWEL RATIO IN DOMAIN (CRITICAL - WAS MISSING!)
            vowels = 'aeiou'
            vowel_count = sum(1 for c in domain.lower() if c in vowels)
            features['vowel_ratio'] = vowel_count / max(len(domain), 1)
            
            # 18. HEX CHARACTERS (WAS MISSING!)
            features['has_hex'] = 1 if re.search(r'%[0-9a-fA-F]{2}', url) else 0
            
            # 19. PUNYCODE (WAS MISSING!)
            features['is_punycode'] = 1 if 'xn--' in url.lower() else 0
            
            # 20. PORT NUMBER
            features['has_port'] = 1 if ':' in parsed.netloc and parsed.port else 0
            
            return features
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def _has_ip_address(self, netloc):
        """Check if URL contains IP address"""
        ipv4_pattern = re.compile(r'^(\d{1,3}\.){3}\d{1,3}$')
        host = netloc.split(':')[0]
        
        if ipv4_pattern.match(host):
            return 1
        if '[' in host and ']' in host:
            return 1
        return 0
    
    def _calculate_entropy(self, text):
        """Calculate Shannon entropy - CRITICAL FOR DETECTION"""
        if not text:
            return 0
        
        freq = {}
        for char in text:
            freq[char] = freq.get(char, 0) + 1
        
        entropy = 0
        text_len = len(text)
        for count in freq.values():
            probability = count / text_len
            entropy -= probability * np.log2(probability)
        
        return entropy
    
    def _max_consecutive_consonants(self, text):
        """Find maximum consecutive consonants"""
        consonants = 'bcdfghjklmnpqrstvwxyz'
        max_count = 0
        current_count = 0
        
        for char in text.lower():
            if char in consonants:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
        
        return max_count

# Initialize the feature extractor globally
feature_extractor = URLFeatureExtractor()

def extract_features(url: str) -> dict:
    """Wrapper function - uses the class to extract features"""
    return feature_extractor.extract_all_features(url)

# ======================================================================
# REDIRECT CHAIN ANALYZER
# ======================================================================

def analyze_redirects(url: str, timeout: int = 10) -> dict:
    """Follow redirects and track the chain"""
    
    redirect_chain = [url]
    final_destination = url
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(
            url, 
            headers=headers, 
            timeout=timeout, 
            allow_redirects=True,
            verify=False
        )
        
        if response.history:
            redirect_chain = [r.url for r in response.history] + [response.url]
        
        final_destination = response.url
        
    except requests.exceptions.Timeout:
        pass
    except requests.exceptions.RequestException as e:
        print(f"Redirect analysis error: {e}")
    
    return {
        "redirect_chain": redirect_chain,
        "final_destination": final_destination,
        "redirect_count": len(redirect_chain) - 1
    }

# ======================================================================
# SECURE PAGE FETCHER
# ======================================================================

def fetch_page_info(url: str, timeout: int = 10) -> dict:
    """Safely fetch page information without executing scripts"""
    
    page_info = {
        "title": "Unknown",
        "status_code": 0,
        "content_type": "Unknown",
        "server": "Unknown",
        "response_time": 0,
        "error": None
    }
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        start_time = time.time()
        response = requests.get(
            url, 
            headers=headers, 
            timeout=timeout,
            verify=False
        )
        page_info["response_time"] = round(time.time() - start_time, 2)
        
        page_info["status_code"] = response.status_code
        page_info["content_type"] = response.headers.get('Content-Type', 'Unknown')
        page_info["server"] = response.headers.get('Server', 'Unknown')
        
        title_match = re.search(r'<title[^>]*>([^<]+)</title>', response.text, re.IGNORECASE)
        if title_match:
            page_info["title"] = title_match.group(1).strip()[:100]
            
    except requests.exceptions.Timeout:
        page_info["error"] = "Connection timed out"
    except requests.exceptions.RequestException as e:
        page_info["error"] = str(e)[:100]
    
    return page_info

# ======================================================================
# ML PREDICTION
# ======================================================================

def get_phishing_score(features: dict) -> float:
    """Get phishing probability from ML model"""
    
    if model is None or scaler is None or features is None:
        return 50.0
    
    try:
        # Create feature vector in EXACT order as training
        feature_vector = []
        for name in feature_names:
            feature_vector.append(features.get(name, 0))
        
        # Scale features
        feature_array = np.array(feature_vector).reshape(1, -1)
        scaled_features = scaler.transform(feature_array)
        
        # Get prediction probability
        proba = model.predict_proba(scaled_features)[0]
        phishing_prob = proba[1] * 100
        
        return round(phishing_prob, 2)
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return 50.0

def get_risk_level(score: float) -> str:
    """Convert score to risk level"""
    if score >= 70:
        return "HIGH"
    elif score >= 40:
        return "MEDIUM"
    else:
        return "LOW"

# ======================================================================
# API ENDPOINTS
# ======================================================================

@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "message": "Phishing URL Detector API is running",
        "model_loaded": model is not None
    }

@app.post("/analyze", response_model=AnalysisResponse)
def analyze_url(request: URLRequest):
    """Main endpoint to analyze a URL for phishing"""
    
    start_time = time.time()
    url = request.url.strip()
    
    if not url:
        raise HTTPException(status_code=400, detail="URL cannot be empty")
    
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url
    
    # Step 1: Extract features (NOW MATCHES TRAINING!)
    features = extract_features(url)
    
    # Step 2: Get ML phishing score
    phishing_score = get_phishing_score(features)
    risk_level = get_risk_level(phishing_score)
    
    # Step 3: Analyze redirects
    redirect_info = analyze_redirects(url)
    
    # Step 4: Fetch page info
    page_info = fetch_page_info(url)
    
    analysis_time = round(time.time() - start_time, 2)
    
    return AnalysisResponse(
        url=url,
        phishing_score=phishing_score,
        risk_level=risk_level,
        features=features,
        redirect_chain=redirect_info["redirect_chain"],
        final_destination=redirect_info["final_destination"],
        page_info=page_info,
        analysis_time=analysis_time
    )

@app.get("/health")
def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "features_count": len(feature_names) if feature_names else 0
    }

# ======================================================================
# RUN SERVER
# ======================================================================

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("🚀 Starting Phishing Detection API Server")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)





# 🟢 TESTING LEGITIMATE SITES (Should be LOW risk)

# ============================================================
# Testing: https://www.google.com
# Expected: LOW
# ============================================================
# ✅ Phishing Score: 57.0%
# ✅ Risk Level: MEDIUM
# ✅ Analysis Time: 1.96s

# 📊 Key Features:
#    - URL Length: 22
#    - Has HTTPS: 1
#    - Suspicious Keywords: 0
#    - Entropy: 3.663532754804254
#    - Has IP: 0

# ============================================================
# Testing: https://github.com
# Expected: LOW
# ============================================================
# ✅ Phishing Score: 60.0%
# ✅ Risk Level: MEDIUM
# ✅ Analysis Time: 1.49s

# 📊 Key Features:
#    - URL Length: 18
#    - Has HTTPS: 1
#    - Suspicious Keywords: 0
#    - Entropy: 3.6835423624332306
#    - Has IP: 0

# ============================================================
# Testing: https://www.amazon.com
# Expected: LOW
# ============================================================
# ✅ Phishing Score: 57.0%
# ✅ Risk Level: MEDIUM
# ✅ Analysis Time: 1.7s

# 📊 Key Features:
#    - URL Length: 22
#    - Has HTTPS: 1
#    - Suspicious Keywords: 0
#    - Entropy: 3.6978458230844122
#    - Has IP: 0


# 🔴 TESTING SUSPICIOUS PATTERNS (Should be HIGH risk)

# ============================================================
# Testing: http://secure-paypal-verify.suspicious.com/login
# Expected: HIGH
# ============================================================
# ✅ Phishing Score: 65.0%
# ✅ Risk Level: MEDIUM
# ✅ Analysis Time: 0.1s

# 📊 Key Features:
#    - URL Length: 48
#    - Has HTTPS: 0
#    - Suspicious Keywords: 4
#    - Entropy: 4.29799505257913
#    - Has IP: 0

# ============================================================
# Testing: http://192.168.1.1/banking/login.php
# Expected: HIGH
# ============================================================
# ❌ Connection Error: HTTPConnectionPool(host='localhost', port=8000): Read timed out. (read timeout=15)

# ============================================================
# Testing: http://account-verify-urgent.tk/signin
# Expected: HIGH
# ============================================================
# ✅ Phishing Score: 74.0%
# ✅ Risk Level: HIGH
# ✅ Analysis Time: 0.44s

# 📊 Key Features:
#    - URL Length: 38
#    - Has HTTPS: 0
#    - Suspicious Keywords: 5
#    - Entropy: 4.165837632423487
#    - Has IP: 0


# 🔵 TEST YOUR OWN URL:
# Enter a URL to test: http://secure-paypal-verify.suspicious.com/login

# ============================================================
# Testing: http://secure-paypal-verify.suspicious.com/login
# Expected: UNKNOWN
# ============================================================
# ✅ Phishing Score: 65.0%
# ✅ Risk Level: MEDIUM
# ✅ Analysis Time: 0.07s

# 📊 Key Features:
#    - URL Length: 48
#    - Has HTTPS: 0
#    - Suspicious Keywords: 4
#    - Entropy: 4.29799505257913
#    - Has IP: 0

# ============================================================
# ✅ Testing Complete!
# ============================================================
