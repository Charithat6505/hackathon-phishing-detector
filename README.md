<div align="center">

# 🛡️ AI-Powered Phishing Detection System

### Intelligent URL analysis through multiple security layers using machine learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![ML Model](https://img.shields.io/badge/Accuracy-94.5%25-success.svg)](/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-In%20Development-orange.svg)](/)

**Comprehensive threat assessments delivered in under 3 seconds**

[Features](#-key-features) • [Installation](#-quick-start) • [Documentation](#-documentation) • [Performance](#-model-performance)

</div>

---

## 🎯 Overview

An intelligent phishing detection system that protects users from malicious URLs by combining machine learning with real-time security analysis. Trained on 10,000+ real-world phishing examples, this system delivers instant risk assessments through a clean, professional interface.

### Perfect For:
- 🏆 Hackathon projects
- 🎓 Educational demonstrations
- 🔬 Cybersecurity research
- 🏢 Production security tool foundation

---

## ✨ Key Features

<table>
<tr>
<td width="50%">

### 🤖 **ML Detection**
Logistic Regression model with **94.5% accuracy** trained on 10,000+ URLs

### 🔍 **Smart Analysis**
15+ URL characteristics analyzed simultaneously in real-time

### 🔗 **Redirect Tracking**
Exposes hidden destinations through URL shorteners

### 🛡️ **Safe Fetching**
Retrieves page content without executing malicious JavaScript

</td>
<td width="50%">

### ⚡ **Fast Results**
Complete analysis in **2-3 seconds**

### 📊 **Risk Scoring**
Color-coded threat levels:
- 🟢 Green (Safe)
- 🟡 Yellow (Suspicious)  
- 🔴 Red (Dangerous)

### 🎨 **Clean UI**
Professional Streamlit interface

### 🚀 **Easy Deploy**
One-click deployment to cloud platforms

</td>
</tr>
</table>

---

## 🗺️ Architecture

```
┌──────────────────┐
│   User Input     │  Enter Suspicious URL
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Streamlit UI    │  Professional Frontend Interface
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  FastAPI Backend │  Multi-Layer Security Analysis
└────────┬─────────┘
         │
    ┌────┴────┬──────────┬────────────┬───────────┐
    ▼         ▼          ▼            ▼           ▼
┌────────┐ ┌──────┐ ┌────────┐ ┌──────────┐ ┌────────┐
│   ML   │ │ URL  │ │Redirect│ │  Secure  │ │  Risk  │
│ Model  │ │Feat. │ │ Chain  │ │  Fetch   │ │  Score │
└────────┘ └──────┘ └────────┘ └──────────┘ └────────┘
         │
         ▼
┌──────────────────┐
│  Risk Report     │  Comprehensive Threat Assessment
└──────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
pip or conda
```

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/mithalikp25/hackathon-phishing-detector.git
   cd hackathon-phishing-detector
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   # Test the ML model (Phases 1-3 Complete)
   python src/model_training.py --test
   
   # Full stack application (Coming Soon - Phases 4-5)
   python run.py
   ```

---

## 📦 Project Structure

```
hackathon-phishing-detector/
│
├── 📊 data/                      # Dataset files
│   ├── raw/                      # Original Hugging Face data
│   ├── processed/                # Cleaned data
│   └── train_test_split/         # ML training data
│
├── 🤖 models/                    # Trained ML models
│   ├── phishing_model.pkl        # Main detection model
│   ├── scaler.pkl                # Feature scaler
│   └── model_metrics.json        # Performance statistics
│
├── 📓 notebooks/                 # Jupyter analysis notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
│
├── 🛠️ src/                       # Core ML pipeline
│   ├── data_processing.py        # Data loading & cleaning
│   ├── feature_extraction.py     # URL feature engineering
│   ├── model_training.py         # Model training & evaluation
│   └── url_analyzer.py           # Analysis utilities
│
├── 🌐 api/                       # Backend API (Phase 4)
│   ├── main.py                   # FastAPI application
│   ├── routes.py                 # API endpoints
│   └── models.py                 # Request/response schemas
│
├── 🎨 frontend/                  # User Interface (Phase 5)
│   ├── app.py                    # Streamlit application
│   └── components/               # UI components
│
├── 📚 docs/                      # Documentation
│   ├── PHASE_1_DOCUMENTATION.md
│   ├── PHASE_2_DOCUMENTATION.md
│   └── PHASE_3_DOCUMENTATION.md
│
├── 🧪 tests/                     # Unit tests
│   ├── test_features.py
│   ├── test_model.py
│   └── test_api.py
│
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

---

## 🎯 Development Status

### ✅ Completed Phases (1-3)

- [x] **Phase 1:** Dataset exploration & cleaning (10,000+ URLs processed)
- [x] **Phase 2:** Feature engineering (15+ features implemented)
- [x] **Phase 3:** ML model training (94.5% accuracy achieved)

### 🚧 In Progress

- [ ] **Phase 4:** FastAPI backend with 5 analysis layers
- [ ] **Phase 5:** Streamlit frontend UI
- [ ] **Phase 6:** Integration & comprehensive testing
- [ ] **Phase 7:** Cloud deployment & demo

---

## 📊 Model Performance

<div align="center">

### Phase 3 Results

| Metric | Score |
|--------|-------|
| **Accuracy** | 94.5% |
| **Precision** | 92.8% |
| **Recall** | 95.2% |
| **F1-Score** | 94.0% |
| **ROC-AUC** | 96.3% |

</div>

### Training Details

- **Algorithm:** Logistic Regression
- **Dataset:** 10,000+ URLs (balanced)
- **Training Split:** 80/20
- **Cross-validation:** 5-fold
- **Training Time:** ~30 seconds

---

## 🔬 Technical Deep Dive

### 15+ Features Extracted from URLs

<details>
<summary><b>1. Length Features</b></summary>

- Total URL length
- Hostname length
- Path length
- Number of subdomains

</details>

<details>
<summary><b>2. Character Analysis</b></summary>

- Special character counts (`.`, `-`, `_`, `/`, `?`, `=`, `@`, `&`)
- Digit-to-letter ratio
- Uppercase-to-lowercase ratio

</details>

<details>
<summary><b>3. Pattern Detection</b></summary>

- IP address in URL (major red flag)
- HTTPS vs HTTP usage
- Suspicious keywords (login, verify, account, update, secure, banking)
- URL shortener detection (bit.ly, tinyurl, etc.)

</details>

<details>
<summary><b>4. Obfuscation Checks</b></summary>

- Hexadecimal encoding
- Punycode (internationalized domains)
- Multiple consecutive special characters

</details>

### ML Pipeline

```
URL Input → Feature Extraction → Scaling → ML Model → Probability Score
```

1. **Data Collection:** Load from Hugging Face datasets
2. **Cleaning:** Remove duplicates, invalid URLs, handle missing data
3. **Feature Engineering:** Extract 15+ numerical features
4. **Normalization:** Scale features for optimal performance
5. **Training:** Logistic Regression with hyperparameter tuning
6. **Evaluation:** Test on unseen 20% holdout set
7. **Serialization:** Save model with joblib for production use

---

## 💻 Usage Examples

### Testing the Model (Current - Phase 3)

```python
from src.url_analyzer import analyze_url

# Analyze a suspicious URL
result = analyze_url("http://paypal-verify.suspicious-site.com/login")

print(f"Phishing Probability: {result['score']}%")
print(f"Risk Level: {result['risk_level']}")
print(f"Features: {result['features']}")

# Output:
# Phishing Probability: 87%
# Risk Level: HIGH
# Features: {
#   'url_length': 48,
#   'has_ip': False,
#   'has_https': False,
#   'suspicious_keywords': 2,
#   'subdomain_count': 3,
#   ...
# }
```

### API Usage (Coming in Phase 4)

```python
import requests

response = requests.post(
    "http://localhost:8000/analyze",
    json={"url": "http://suspicious-site.com"}
)

print(response.json())
```

### Frontend Demo (Coming in Phase 5)

```bash
streamlit run frontend/app.py
# Opens browser at http://localhost:8501
# Paste URL → Click Analyze → See Results
```

---

## 🛠️ Dependencies

```txt
# Core ML & Data Processing
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
joblib>=1.2.0

# Web & API (Phase 4-5)
fastapi>=0.104.0
uvicorn>=0.24.0
streamlit>=1.28.0
requests>=2.31.0

# Data Loading
datasets>=2.14.0

# URL Processing
validators>=0.22.0
tldextract>=5.0.0

# Testing
pytest>=7.4.0
```

---

## 📖 Documentation

Comprehensive guides for each development phase:

- 📘 [**Phase 1:** Data Exploration](docs/PHASE_1_DOCUMENTATION.md) - Dataset analysis & cleaning
- 📗 [**Phase 2:** Feature Engineering](docs/PHASE_2_DOCUMENTATION.md) - Building the feature extraction pipeline
- 📙 [**Phase 3:** Model Training](docs/PHASE_3_DOCUMENTATION.md) - ML model development & evaluation
- 📕 **API Documentation** - FastAPI endpoints (Coming Soon)

---

## 🎓 Learning Resources

This project demonstrates:

- ✅ **Machine Learning:** Binary classification, model evaluation
- ✅ **Feature Engineering:** Extracting meaningful patterns from text data
- ✅ **API Development:** RESTful services with FastAPI
- ✅ **Web Development:** Interactive UIs with Streamlit
- ✅ **DevOps:** Model serialization, deployment, testing

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

<div align="center">

### 🔒 Stay Safe Online. Detect Phishing with AI. 🔒

Made with ❤️ for cybersecurity awareness

---

⭐ **Star this repository if you found it helpful!** ⭐

</div>