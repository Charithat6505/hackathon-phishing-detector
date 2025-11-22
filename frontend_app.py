"""
Phase 5: Streamlit Frontend
Beautiful UI for Phishing URL Detection
"""

import streamlit as st
import requests
import os
import plotly.graph_objects as go
import time
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Phishing URL Detector",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 1.2rem;
        padding: 0.5rem;
        border-radius: 10px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #0d5a8f;
    }
    .risk-safe {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
    }
    .risk-low {
        background-color: #d1ecf1;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #17a2b8;
    }
    .risk-medium {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
    }
    .risk-high {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
    }
    .risk-critical {
        background-color: #f5c6cb;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #8b0000;
    }
    </style>
""", unsafe_allow_html=True)


# API Configuration - works in both local and production
API_URL = os.getenv("BACKEND_URL", "https://hackathon-phishing-detector-api.onrender.com")

# Initialize session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

def check_api_health():
    """Check if the backend API is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def analyze_url(url: str):
    """Send URL to backend API for analysis"""
    try:
        response = requests.post(
            f"{API_URL}/analyze",
            json={"url": url},
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"Error {response.status_code}: {response.text}"
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to API. Make sure backend_api.py is running!"
    except requests.exceptions.Timeout:
        return None, "Request timed out. The URL might be taking too long to analyze."
    except Exception as e:
        return None, f"Error: {str(e)}"

def create_risk_gauge(phishing_prob: float, risk_level: str):
    """Create a circular gauge showing phishing probability"""
    
    # Color mapping
    color_map = {
        "SAFE": "#28a745",
        "LOW": "#17a2b8",
        "MEDIUM": "#ffc107",
        "HIGH": "#dc3545",
        "CRITICAL": "#8b0000"
    }
    
    color = color_map.get(risk_level, "#666")
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = phishing_prob,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Phishing Probability", 'font': {'size': 24}},
        number = {'suffix': "%", 'font': {'size': 40}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 20], 'color': '#d4edda'},
                {'range': [20, 40], 'color': '#d1ecf1'},
                {'range': [40, 60], 'color': '#fff3cd'},
                {'range': [60, 80], 'color': '#f8d7da'},
                {'range': [80, 100], 'color': '#f5c6cb'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="white",
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

def display_results(data: dict):
    """Display analysis results in a beautiful format"""
    
    # Extract key information
    url = data['url']
    prediction = data['prediction']
    confidence = data['confidence']
    risk_level = data['risk_level']
    features = data['features']
    phishing_prob = features['phishing_probability']
    
    # Header with risk level
    st.markdown("---")
    st.markdown("## 📊 Analysis Results")
    
    # Risk indicator and gauge
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Risk level box
        risk_class = f"risk-{risk_level.lower()}"
        
        risk_emoji = {
            "SAFE": "✅",
            "LOW": "✅",
            "MEDIUM": "⚠️",
            "HIGH": "🚨",
            "CRITICAL": "🛑"
        }
        
        st.markdown(f"""
            <div class="{risk_class}">
                <h2>{risk_emoji.get(risk_level, '⚠️')} {risk_level} RISK</h2>
                <h3>{prediction}</h3>
                <p style="font-size: 1.1rem;">Confidence: {confidence*100:.1f}%</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Recommendation
        st.markdown("### 💡 Recommendation")
        
        if risk_level in ["SAFE", "LOW"]:
            st.success("✅ This link appears to be legitimate. However, always stay vigilant when clicking links.")
        elif risk_level == "MEDIUM":
            st.warning("⚠️ Exercise caution with this link. Verify the sender and destination before proceeding.")
        else:  # HIGH or CRITICAL
            st.error("🛑 DO NOT visit this link! It shows strong indicators of being a phishing attempt.")
    
    with col2:
        # Gauge chart
        fig = create_risk_gauge(phishing_prob, risk_level)
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed analysis tabs
    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 URL Features", "🔄 Redirect Chain", "📄 Page Info", "📊 Technical Details"])
    
    with tab1:
        st.markdown("### URL Feature Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("URL Length", f"{features['url_length']} chars")
            st.metric("Dots Count", features['dot_count'])
            st.metric("Subdomains", features['subdomain_count'])
        
        with col2:
            st.metric("HTTPS", "✅ Yes" if features['is_https'] else "❌ No")
            st.metric("Has IP Address", "⚠️ Yes" if features['has_ip'] else "✅ No")
            st.metric("URL Shortener", "⚠️ Yes" if features['is_shortened'] else "✅ No")
        
        with col3:
            st.metric("Suspicious Keywords", features['suspicious_keyword_count'])
            st.metric("Entropy", f"{features['entropy']:.2f}")
            st.metric("Has Hex", "⚠️ Yes" if features['has_hex'] else "✅ No")
        
        # Domain checks
        st.markdown("#### 🌐 Domain Verification")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if features['is_trusted_domain']:
                st.success("✅ Trusted Domain (Whitelisted)")
            else:
                st.info("ℹ️ Not in trusted domains list")
        
        with col2:
            if features['domain_exists']:
                st.success("✅ Domain Exists (DNS Verified)")
            else:
                st.error("❌ Domain Does Not Exist")
        
        with col3:
            if features['page_accessible']:
                st.success("✅ Page Accessible")
            else:
                st.warning("⚠️ Page Not Accessible")
    
    with tab2:
        st.markdown("### Redirect Chain Analysis")
        
        redirect_chain = data['redirect_chain']
        
        if len(redirect_chain) == 1:
            st.info("✅ No redirects detected - URL goes directly to destination")
        else:
            st.warning(f"⚠️ Found {len(redirect_chain)-1} redirect(s)")
        
        st.markdown("#### Redirect Path:")
        for i, redirect in enumerate(redirect_chain):
            if i == 0:
                st.markdown(f"**🔵 Original URL:**")
            elif i == len(redirect_chain) - 1:
                st.markdown(f"**🎯 Final Destination:**")
            else:
                st.markdown(f"**↪️ Redirect {i}:**")
            
            st.code(redirect, language=None)
    
    with tab3:
        st.markdown("### Page Information")
        
        page_info = data['page_info']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Page Title:**")
            st.info(page_info['title'])
            
            st.markdown("**Status Code:**")
            status = page_info['status_code']
            if status == 200:
                st.success(f"✅ {status} - OK")
            elif isinstance(status, int) and 300 <= status < 400:
                st.warning(f"⚠️ {status} - Redirect")
            elif isinstance(status, int) and status >= 400:
                st.error(f"❌ {status} - Error")
            else:
                st.warning(f"⚠️ {status}")
        
        with col2:
            st.markdown("**Server:**")
            st.info(page_info['server'])
            
            st.markdown("**Content Type:**")
            st.info(page_info['content_type'])
        
        st.markdown("**Accessibility:**")
        if page_info['accessible']:
            st.success("✅ Page is accessible and loaded successfully")
        else:
            st.error("❌ Page could not be accessed")
    
    with tab4:
        st.markdown("### Technical Analysis Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Analysis Metadata:**")
            st.json({
                "Analysis Time": f"{data['analysis_time']:.3f}s",
                "Timestamp": data['timestamp'],
                "API Version": "1.1.0"
            })
        
        with col2:
            st.markdown("**ML Model Prediction:**")
            st.json({
                "Prediction": prediction,
                "Confidence": f"{confidence*100:.2f}%",
                "Phishing Probability": f"{phishing_prob:.2f}%",
                "Risk Level": risk_level
            })

# Main UI
def main():
    # Header
    st.markdown('<h1 class="main-header">🔒 Phishing URL Detector</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered URL Analysis with 86% Accuracy</p>', unsafe_allow_html=True)
    
    # Check API health
    api_healthy = check_api_health()
    
    if not api_healthy:
        st.error("⚠️ **Backend API is not running!**")
        st.info("Please start the backend API first:")
        st.code("python backend_api.py", language="bash")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📋 About")
        st.info("""
        This tool analyzes URLs for phishing threats using:
        
        - 🤖 Machine Learning (86% accuracy)
        - 🔍 43 URL features analysis
        - 🌐 DNS verification
        - 🔄 Redirect chain tracking
        - ✅ Trusted domain whitelist
        """)
        
        st.markdown("## 🎯 Sample URLs")
        
        sample_urls = {
            "✅ Google": "https://www.google.com",
            "✅ GitHub": "https://github.com",
            "✅ Wikipedia": "https://www.wikipedia.org",
            "⚠️ Suspicious PayPal": "http://secure-paypal-verify.xyz/login",
            "⚠️ IP Address": "http://192.168.1.1/admin/login.php",
            "⚠️ Fake Domain": "https://this-site-does-not-exist-12345.void"
        }
        
        for label, url in sample_urls.items():
            if st.button(label, key=url):
                st.session_state.test_url = url
        
        st.markdown("---")
        st.markdown("## 📊 Statistics")
        if st.session_state.analysis_history:
            total = len(st.session_state.analysis_history)
            safe = sum(1 for r in st.session_state.analysis_history if r['risk_level'] in ['SAFE', 'LOW'])
            dangerous = total - safe
            
            st.metric("Total Analyzed", total)
            st.metric("Safe URLs", safe)
            st.metric("Dangerous URLs", dangerous)
    
    # Main content
    st.markdown("## 🔍 Enter URL to Analyze")
    
    # URL input
    col1, col2 = st.columns([4, 1])
    
    with col1:
        # Check if sample URL was clicked
        default_url = st.session_state.get('test_url', '')
        url_input = st.text_input(
            "Enter the URL you want to check:",
            value=default_url,
            placeholder="https://example.com",
            label_visibility="collapsed"
        )
        # Clear the test URL after using it
        if 'test_url' in st.session_state:
            del st.session_state.test_url
    
    with col2:
        analyze_button = st.button("🔍 Analyze", use_container_width=True)
    
    # Analyze URL when button is clicked
    if analyze_button:
        if not url_input:
            st.warning("⚠️ Please enter a URL to analyze")
        else:
            with st.spinner("🔄 Analyzing URL... Please wait..."):
                # Add a progress bar for better UX
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                # Analyze the URL
                result, error = analyze_url(url_input)
                
                if error:
                    st.error(f"❌ {error}")
                else:
                    # Save to history
                    st.session_state.analysis_history.append({
                        'url': url_input,
                        'risk_level': result['risk_level'],
                        'timestamp': datetime.now()
                    })
                    
                    # Display results
                    display_results(result)
                    
                    # Success message
                    st.balloons()
    
    # Show analysis history
    if st.session_state.analysis_history:
        st.markdown("---")
        st.markdown("## 📜 Analysis History")
        
        # Show last 5 analyses
        recent = st.session_state.analysis_history[-5:][::-1]
        
        for i, record in enumerate(recent):
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.text(record['url'][:60] + ('...' if len(record['url']) > 60 else ''))
            
            with col2:
                risk_color = {
                    'SAFE': '🟢',
                    'LOW': '🟢',
                    'MEDIUM': '🟡',
                    'HIGH': '🔴',
                    'CRITICAL': '🔴'
                }
                st.text(f"{risk_color.get(record['risk_level'], '⚪')} {record['risk_level']}")
            
            with col3:
                st.text(record['timestamp'].strftime("%H:%M:%S"))

if __name__ == "__main__":
    main()


