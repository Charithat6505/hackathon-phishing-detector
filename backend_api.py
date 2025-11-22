"""
Phase 4: Backend API with FastAPI
Real-time phishing URL detection API with trusted domain whitelist
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
import joblib
import numpy as np
import requests
from urllib.parse import urlparse
import time
from datetime import datetime
import sys
import os
import socket

# Import feature extractor from Phase 2
sys.path.append(os.path.dirname(__file__))
from feature_extraction import URLFeatureExtractor

# Trusted domains whitelist - known legitimate sites
TRUSTED_DOMAINS = {
    'google.com', 'www.google.com', 'youtube.com', 'www.youtube.com',
    'github.com', 'www.github.com', 'gitlab.com', 'www.gitlab.com',
    'stackoverflow.com', 'www.stackoverflow.com', 'stackexchange.com',
    'microsoft.com', 'www.microsoft.com', 'azure.com', 'office.com',
    'apple.com', 'www.apple.com', 'icloud.com', 
    'amazon.com', 'www.amazon.com', 'aws.amazon.com',
    'facebook.com', 'www.facebook.com', 'meta.com',
    'twitter.com', 'www.twitter.com', 'x.com',
    'linkedin.com', 'www.linkedin.com',
    'reddit.com', 'www.reddit.com',
    'wikipedia.org', 'www.wikipedia.org', 'en.wikipedia.org',
    'instagram.com', 'www.instagram.com',
    'netflix.com', 'www.netflix.com',
    'paypal.com', 'www.paypal.com',
    'dropbox.com', 'www.dropbox.com',
    'zoom.us', 'www.zoom.us',
    'adobe.com', 'www.adobe.com',
    'salesforce.com', 'www.salesforce.com',
    'npmjs.com', 'www.npmjs.com',
    'pypi.org', 'www.pypi.org'
}

def is_trusted_domain(url: str) -> bool:
    """
    Check if URL is from a trusted, well-known domain
    This prevents false positives on major legitimate sites
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        # Remove port if present
        if ':' in domain:
            domain = domain.split(':')[0]
        
        # Check exact match or subdomain match
        if domain in TRUSTED_DOMAINS:
            return True
        
        # Check if it's a subdomain of a trusted domain
        for trusted in TRUSTED_DOMAINS:
            if domain.endswith('.' + trusted):
                return True
        
        return False
    except:
        return False

def domain_exists(url: str) -> bool:
    """
    Check if the domain actually exists (DNS resolution)
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc
        
        # Remove port if present
        if ':' in domain:
            domain = domain.split(':')[0]
        
        # Try to resolve DNS
        socket.gethostbyname(domain)
        return True
    except (socket.gaierror, socket.error):
        return False
    except:
        return False

# Initialize FastAPI app
app = FastAPI(
    title="Phishing URL Detection API",
    description="AI-powered phishing detection with accessibility checks + trusted domain whitelist",
    version="1.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
scaler = None
feature_names = None
metadata = None
extractor = None

# Request/Response models
class URLRequest(BaseModel):
    url: str
    
class AnalysisResponse(BaseModel):
    url: str
    prediction: str
    confidence: float
    risk_level: str
    features: dict
    redirect_chain: list
    final_destination: str
    page_info: dict
    analysis_time: float
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    is_model_loaded: bool
    accuracy: float
    uptime: str

class ModelInfoResponse(BaseModel):
    name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    feature_count: int
    training_samples: int

# Load model on startup
@app.on_event("startup")
async def load_model():
    """Load ML model and preprocessing components"""
    global model, scaler, feature_names, metadata, extractor
    
    try:
        print("Loading ML model and components...")
        
        model = joblib.load('phishing_model.pkl')
        print("✅ Model loaded")
        
        scaler = joblib.load('feature_scaler.pkl')
        print("✅ Scaler loaded")
        
        feature_names = joblib.load('feature_names.pkl')
        print("✅ Feature names loaded")
        
        metadata = joblib.load('model_metadata.pkl')
        print("✅ Metadata loaded")
        
        extractor = URLFeatureExtractor()
        print("✅ Feature extractor initialized")
        
        print(f"\n🚀 API Ready! Model: {metadata['model_name']} | Accuracy: {metadata['accuracy']*100:.2f}%")
        print(f"✅ Trusted domains whitelist: {len(TRUSTED_DOMAINS)} domains")
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        print("Make sure you've run Phase 3 (retrain_model.py) first!")
        raise

@app.get("/")
async def root():
    """API welcome message"""
    return {
        "message": "Phishing URL Detection API",
        "version": "1.1.0",
        "accuracy": f"{metadata['accuracy']*100:.2f}%" if metadata else "N/A",
        "trusted_domains": len(TRUSTED_DOMAINS),
        "endpoints": {
            "analyze": "/analyze (POST)",
            "health": "/health (GET)",
            "model_info": "/model-info (GET)",
            "docs": "/docs (Interactive API documentation)"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health status"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "is_model_loaded": model is not None,
        "accuracy": metadata['accuracy'] if metadata else 0.0,
        "uptime": "running"
    }

@app.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get trained model information"""
    if metadata is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "name": metadata['model_name'],
        "accuracy": metadata['accuracy'],
        "precision": metadata['precision'],
        "recall": metadata['recall'],
        "f1_score": metadata['f1_score'],
        "feature_count": metadata['feature_count'],
        "training_samples": metadata['training_samples']
    }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_url(request: URLRequest):
    """
    Analyze a URL for phishing threats
    
    Uses hybrid approach:
    - Whitelist for known trusted domains
    - DNS check for domain existence
    - ML model for unknown/suspicious URLs
    - Accessibility verification
    """
    start_time = time.time()
    
    if model is None or extractor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    url = request.url.strip()
    
    if not url:
        raise HTTPException(status_code=400, detail="URL cannot be empty")
    
    try:
        # Check if trusted domain (whitelist)
        is_trusted = is_trusted_domain(url)
        
        # Check if domain exists (DNS resolution)
        exists = domain_exists(url)
        
        # Extract features
        features = extractor.extract_all_features(url)
        
        if features is None:
            raise HTTPException(status_code=400, detail="Failed to extract features from URL")
        
        # Prepare features for prediction
        feature_values = [features.get(fname, 0) for fname in feature_names]
        feature_array = np.array(feature_values).reshape(1, -1)
        
        # Scale features
        feature_scaled = scaler.transform(feature_array)
        
        # Make prediction
        raw_prediction = model.predict(feature_scaled)[0]
        probability = model.predict_proba(feature_scaled)[0]
        
        # Get ML prediction values
        ml_confidence = float(probability[raw_prediction])
        phishing_prob = float(probability[1])
        
        # Override logic with multiple checks
        if is_trusted:
            # Trusted domain - always safe
            prediction = 0
            confidence = 0.95
            phishing_prob = 0.05
            risk_level = "SAFE"
            prediction_text = "LEGITIMATE (Trusted Domain)"
        elif not exists:
            # Domain doesn't exist - highly suspicious
            prediction = 1
            confidence = 0.90
            phishing_prob = 0.90
            risk_level = "HIGH"
            prediction_text = "SUSPICIOUS (Domain Unreachable)"
        else:
            # Use ML prediction for unknown domains
            prediction = raw_prediction
            confidence = ml_confidence
            prediction_text = "PHISHING" if prediction == 1 else "LEGITIMATE"
            
            # Determine risk level based on phishing probability
            if phishing_prob >= 0.8:
                risk_level = "CRITICAL"
            elif phishing_prob >= 0.6:
                risk_level = "HIGH"
            elif phishing_prob >= 0.4:
                risk_level = "MEDIUM"
            elif phishing_prob >= 0.2:
                risk_level = "LOW"
            else:
                risk_level = "SAFE"
        
        # Follow redirects
        redirect_chain, final_destination = await get_redirect_chain(url)
        
        # Get page info
        page_info = await get_page_info(final_destination or url)
        
        # Adjust risk if page is not accessible but ML says legitimate
        if not page_info["accessible"] and prediction == 0 and not is_trusted:
            # Page not accessible - increase suspicion
            if phishing_prob < 0.5:
                phishing_prob = 0.6  # Boost to medium risk
                risk_level = "MEDIUM"
                prediction_text += " (Page Not Accessible)"
        
        # Calculate analysis time
        analysis_time = time.time() - start_time
        
        return {
            "url": url,
            "prediction": prediction_text,
            "confidence": confidence,
            "risk_level": risk_level,
            "features": {
                "url_length": features.get('url_length', 0),
                "is_https": bool(features.get('is_https', 0)),
                "has_ip": bool(features.get('has_ip', 0)),
                "dot_count": features.get('dot_count', 0),
                "subdomain_count": features.get('subdomain_count', 0),
                "suspicious_keyword_count": features.get('suspicious_keyword_count', 0),
                "entropy": round(features.get('entropy', 0), 2),
                "is_shortened": bool(features.get('is_shortened', 0)),
                "has_hex": bool(features.get('has_hex', 0)),
                "phishing_probability": round(phishing_prob * 100, 2),
                "is_trusted_domain": is_trusted,
                "domain_exists": exists,
                "page_accessible": page_info["accessible"]
            },
            "redirect_chain": redirect_chain,
            "final_destination": final_destination or url,
            "page_info": page_info,
            "analysis_time": round(analysis_time, 3),
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

async def get_redirect_chain(url: str, max_redirects: int = 5, timeout: int = 5):
    """Follow URL redirects and return the chain"""
    redirect_chain = [url]
    final_url = url
    
    try:
        response = requests.head(
            url,
            allow_redirects=True,
            timeout=timeout,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        
        for resp in response.history:
            if resp.url not in redirect_chain:
                redirect_chain.append(resp.url)
        
        if response.url not in redirect_chain:
            redirect_chain.append(response.url)
        
        final_url = response.url
        
    except requests.exceptions.Timeout:
        redirect_chain.append("(Timeout - URL not reachable)")
    except requests.exceptions.RequestException as e:
        redirect_chain.append(f"(Error: {str(e)[:50]})")
    except Exception as e:
        redirect_chain.append(f"(Error: {str(e)[:50]})")
    
    return redirect_chain, final_url

async def get_page_info(url: str, timeout: int = 5):
    """Safely fetch page information"""
    page_info = {
        "title": "N/A",
        "status_code": None,
        "server": "N/A",
        "content_type": "N/A",
        "accessible": False
    }
    
    try:
        response = requests.get(
            url,
            timeout=timeout,
            allow_redirects=True,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        
        page_info["status_code"] = response.status_code
        page_info["server"] = response.headers.get('Server', 'N/A')
        page_info["content_type"] = response.headers.get('Content-Type', 'N/A')
        page_info["accessible"] = response.status_code == 200
        
        if 'text/html' in page_info["content_type"]:
            import re
            title_match = re.search(r'<title[^>]*>([^<]+)</title>', response.text, re.IGNORECASE)
            if title_match:
                page_info["title"] = title_match.group(1).strip()[:100]
        
    except requests.exceptions.Timeout:
        page_info["status_code"] = "Timeout"
    except requests.exceptions.RequestException as e:
        page_info["status_code"] = f"Error: {str(e)[:30]}"
    except Exception as e:
        page_info["status_code"] = f"Error: {str(e)[:30]}"
    
    return page_info

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*70)
    print("STARTING PHISHING DETECTION API (v1.1 - WITH DNS CHECKS)")
    print("="*70)
    print("\n🚀 Starting server on http://localhost:8000")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    print(f"✅ Trusted domains: {len(TRUSTED_DOMAINS)}")
    print("✅ DNS verification enabled")
    print("\nPress CTRL+C to stop the server\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")