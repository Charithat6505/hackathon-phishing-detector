"""
RETRAIN MODEL WITH BETTER DATASET
This script uses the Kaggle Phishing Site URLs dataset which has better
representation of real-world legitimate and phishing URLs
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import joblib
import os
from feature_extraction import URLFeatureExtractor

print("="*70)
print("RETRAINING MODEL WITH BETTER DATASET")
print("="*70)

# ============================================================
# STEP 1: Download and Load Better Dataset
# ============================================================
print("\n[1/6] Loading dataset...")

# Try to load from HuggingFace
try:
    from datasets import load_dataset
    
    print("   Downloading from HuggingFace (shawhin/phishing-site-classification)...")
    dataset = load_dataset("shawhin/phishing-site-classification", split="train")
    df = dataset.to_pandas()
    print(f"   ✅ Loaded {len(df)} URLs from HuggingFace")
    
    # Check column names
    print(f"   Columns: {df.columns.tolist()}")
    
    # Rename columns to match expected format
    if 'text' in df.columns:
        df = df.rename(columns={'text': 'url'})
    if 'URL' in df.columns:
        df = df.rename(columns={'URL': 'url'})
    
    # Fix label column name (plural to singular)
    if 'labels' in df.columns:
        df = df.rename(columns={'labels': 'label'})
    
    # Ensure label exists and is numeric
    if 'label' not in df.columns:
        print("   ❌ Error: No label column found!")
        exit(1)
    
    # Convert labels to numeric if needed
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    
    # Remove rows with NaN labels
    df = df.dropna(subset=['label'])
    
    print(f"   ✅ After cleaning: {len(df)} URLs with valid labels")
    print(f"   Label distribution: {df['label'].value_counts().to_dict()}")
    
except Exception as e:
    print(f"   ⚠️ HuggingFace failed: {e}")
    print("   Trying alternative: Kaggle dataset...")
    
    # Alternative: Use existing clean dataset if available
    if os.path.exists('phishing_dataset_clean.csv'):
        df = pd.read_csv('phishing_dataset_clean.csv')
        print(f"   ✅ Loaded {len(df)} URLs from local file")
    else:
        print("   ❌ No dataset found! Please download manually.")
        exit(1)

# ============================================================
# STEP 2: Add Real Legitimate URLs
# ============================================================
print("\n[2/6] Adding verified legitimate URLs...")

# Real legitimate URLs to ensure model learns correct patterns
legitimate_urls = [
    # Search engines
    "https://www.google.com", "https://google.com", "https://www.bing.com",
    "https://duckduckgo.com", "https://www.duckduckgo.com", "https://yahoo.com",
    "https://www.yahoo.com", "https://search.yahoo.com",
    
    # Social media
    "https://www.facebook.com", "https://facebook.com", "https://www.twitter.com",
    "https://twitter.com", "https://x.com", "https://www.instagram.com",
    "https://instagram.com", "https://www.linkedin.com", "https://linkedin.com",
    "https://www.reddit.com", "https://reddit.com", "https://www.tiktok.com",
    "https://www.pinterest.com", "https://www.snapchat.com",
    
    # Tech companies
    "https://www.microsoft.com", "https://microsoft.com", "https://www.apple.com",
    "https://apple.com", "https://www.amazon.com", "https://amazon.com",
    "https://www.netflix.com", "https://netflix.com", "https://www.spotify.com",
    "https://www.github.com", "https://github.com", "https://gitlab.com",
    
    # News sites
    "https://www.bbc.com", "https://www.cnn.com", "https://www.nytimes.com",
    "https://www.theguardian.com", "https://www.reuters.com", "https://www.forbes.com",
    
    # Education
    "https://www.wikipedia.org", "https://en.wikipedia.org", "https://www.khanacademy.org",
    "https://www.coursera.org", "https://www.udemy.com", "https://www.edx.org",
    
    # E-commerce & Finance
    "https://www.paypal.com", "https://paypal.com", "https://www.ebay.com",
    "https://www.etsy.com", "https://www.shopify.com", "https://www.stripe.com",
    
    # Productivity
    "https://www.zoom.us", "https://zoom.us", "https://www.slack.com",
    "https://slack.com", "https://www.notion.so", "https://www.trello.com",
    "https://www.dropbox.com", "https://drive.google.com", "https://docs.google.com",
    
    # Developer tools
    "https://stackoverflow.com", "https://www.stackoverflow.com",
    "https://npmjs.com", "https://www.npmjs.com", "https://pypi.org",
    "https://www.docker.com", "https://hub.docker.com",
    
    # Cloud services
    "https://aws.amazon.com", "https://cloud.google.com", "https://azure.microsoft.com",
    
    # More common sites
    "https://www.whatsapp.com", "https://web.whatsapp.com",
    "https://www.telegram.org", "https://www.discord.com",
    "https://www.twitch.tv", "https://www.youtube.com", "https://youtube.com",
    "https://www.vimeo.com", "https://www.dailymotion.com",
    
    # Short legitimate URLs (important!)
    "https://t.me", "https://wa.me", "https://g.co", "https://fb.com",
    "https://youtu.be", "https://amzn.to",
]

# Create dataframe for legitimate URLs
legit_df = pd.DataFrame({
    'url': legitimate_urls,
    'label': [0] * len(legitimate_urls)  # 0 = legitimate
})

print(f"   Added {len(legitimate_urls)} verified legitimate URLs")

# Combine with existing dataset
df = pd.concat([df, legit_df], ignore_index=True)

# Remove duplicates
df = df.drop_duplicates(subset=['url'])

# Ensure no NaN labels after combining
df = df.dropna(subset=['label'])

print(f"   Total URLs after combining: {len(df)}")
print(f"   Class distribution:")
print(f"   Legitimate (0): {sum(df['label']==0)} ({sum(df['label']==0)/len(df)*100:.1f}%)")
print(f"   Phishing (1):   {sum(df['label']==1)} ({sum(df['label']==1)/len(df)*100:.1f}%)")

# ============================================================
# STEP 3: Extract Features
# ============================================================
print("\n[3/6] Extracting features from URLs...")

extractor = URLFeatureExtractor()
features_list = []
valid_indices = []

for idx, row in df.iterrows():
    if idx % 5000 == 0:
        print(f"   Processing {idx}/{len(df)}...")
    
    url = row['url']
    features = extractor.extract_all_features(url)
    
    if features:
        features['label'] = row['label']
        features_list.append(features)
        valid_indices.append(idx)

features_df = pd.DataFrame(features_list)
print(f"   ✅ Extracted features for {len(features_df)} URLs")

# ============================================================
# STEP 4: Prepare Training Data
# ============================================================
print("\n[4/6] Preparing training data...")

# Separate features and labels
feature_columns = [col for col in features_df.columns if col not in ['label', 'url']]
X = features_df[feature_columns]
y = features_df['label']

# Handle missing values
X = X.fillna(0)

# Final check: remove any NaN labels
valid_mask = ~y.isna()
X = X[valid_mask]
y = y[valid_mask]

print(f"   Features: {len(feature_columns)}")
print(f"   Samples: {len(X)}")
print(f"\n   Class distribution:")
print(f"   Legitimate (0): {sum(y==0)} ({sum(y==0)/len(y)*100:.1f}%)")
print(f"   Phishing (1):   {sum(y==1)} ({sum(y==1)/len(y)*100:.1f}%)")

# Check if we have both classes
if len(y.unique()) < 2:
    print("\n   ❌ ERROR: Need both legitimate and phishing samples!")
    print(f"   Only found class(es): {y.unique()}")
    exit(1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n   Training set: {len(X_train)}")
print(f"   Test set: {len(X_test)}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# STEP 5: Train Model with Better Parameters
# ============================================================
print("\n[5/6] Training Random Forest model...")

model = RandomForestClassifier(
    n_estimators=200,           # More trees for better accuracy
    max_depth=20,               # Limit depth to prevent overfitting
    min_samples_split=5,        # Minimum samples to split
    min_samples_leaf=2,         # Minimum samples in leaf
    class_weight='balanced',    # Handle class imbalance
    random_state=42,
    n_jobs=-1                   # Use all CPU cores
)

model.fit(X_train_scaled, y_train)
print("   ✅ Model trained!")

# ============================================================
# STEP 6: Evaluate Model
# ============================================================
print("\n[6/6] Evaluating model...")

y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\n   📊 Performance Metrics:")
print(f"   Accuracy:  {accuracy*100:.2f}%")
print(f"   Precision: {precision*100:.2f}%")
print(f"   Recall:    {recall*100:.2f}%")
print(f"   F1-Score:  {f1*100:.2f}%")

print(f"\n   Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"\n   Confusion Matrix:")
print(f"   True Negatives (Legit correctly identified):  {tn}")
print(f"   False Positives (Legit marked as Phishing):   {fp}")
print(f"   False Negatives (Phishing marked as Legit):   {fn}")
print(f"   True Positives (Phishing correctly caught):   {tp}")

# False positive rate
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
print(f"\n   ⚠️ False Positive Rate: {fpr*100:.2f}%")

# ============================================================
# STEP 7: Test on Known URLs
# ============================================================
print("\n" + "="*70)
print("TESTING ON KNOWN URLs")
print("="*70)

test_urls = [
    ("https://www.google.com", "Should be LEGITIMATE"),
    ("https://duckduckgo.com", "Should be LEGITIMATE"),
    ("https://www.facebook.com", "Should be LEGITIMATE"),
    ("https://github.com", "Should be LEGITIMATE"),
    ("http://192.168.1.1/login.php", "Should be PHISHING"),
    ("http://secure-paypal-login.fake.com/verify", "Should be PHISHING"),
    ("http://account-update-required.xyz/signin", "Should be PHISHING"),
]

print("\nTesting individual URLs:")
for url, expected in test_urls:
    features = extractor.extract_all_features(url)
    if features:
        feature_vector = [features.get(col, 0) for col in feature_columns]
        feature_scaled = scaler.transform([feature_vector])
        pred = model.predict(feature_scaled)[0]
        proba = model.predict_proba(feature_scaled)[0]
        
        result = "LEGITIMATE" if pred == 0 else "PHISHING"
        confidence = proba[int(pred)] * 100
        
        status = "✅" if (pred == 0 and "LEGITIMATE" in expected) or (pred == 1 and "PHISHING" in expected) else "❌"
        
        print(f"\n{status} {url}")
        print(f"   Prediction: {result} ({confidence:.1f}% confidence)")
        print(f"   Expected: {expected}")

# ============================================================
# STEP 8: Save Model
# ============================================================
print("\n" + "="*70)
print("SAVING MODEL")
print("="*70)

# Backup old model
if os.path.exists('phishing_model.pkl'):
    os.rename('phishing_model.pkl', 'phishing_model_backup.pkl')
    print("   📦 Old model backed up as phishing_model_backup.pkl")

# Save new model
joblib.dump(model, 'phishing_model.pkl')
joblib.dump(scaler, 'feature_scaler.pkl')
joblib.dump(feature_columns, 'feature_names.pkl')

# Save metadata
metadata = {
    'model_name': 'Random Forest',
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'feature_count': len(feature_columns),
    'training_samples': len(X_train),
    'false_positive_rate': fpr
}
joblib.dump(metadata, 'model_metadata.pkl')

print("   ✅ Model saved: phishing_model.pkl")
print("   ✅ Scaler saved: feature_scaler.pkl")
print("   ✅ Feature names saved: feature_names.pkl")
print("   ✅ Metadata saved: model_metadata.pkl")

print("\n" + "="*70)
print("🎉 RETRAINING COMPLETE!")
print("="*70)
print(f"\n   New model accuracy: {accuracy*100:.2f}%")
print(f"   False positive rate: {fpr*100:.2f}%")
print("\n   Now restart your API: python backend_api.py")
print("="*70)



# ======================================================================
# RETRAINING MODEL WITH BETTER DATASET
# ======================================================================

# [1/6] Loading dataset...
#    Downloading from HuggingFace (shawhin/phishing-site-classification)...
#    ✅ Loaded 2100 URLs from HuggingFace
#    Columns: ['text', 'labels']
#    ✅ After cleaning: 2100 URLs with valid labels
#    Label distribution: {0: 1054, 1: 1046}

# [2/6] Adding verified legitimate URLs...
#    Added 86 verified legitimate URLs
#    Total URLs after combining: 2186
#    Class distribution:
#    Legitimate (0): 1140 (52.2%)
#    Phishing (1):   1046 (47.8%)

# [3/6] Extracting features from URLs...
#    Processing 0/2186...
#    ✅ Extracted features for 2186 URLs

# [4/6] Preparing training data...
#    Features: 43
#    Samples: 2186

#    Class distribution:
#    Legitimate (0): 1140 (52.2%)
#    Phishing (1):   1046 (47.8%)

#    Training set: 1748
#    Test set: 438

# [5/6] Training Random Forest model...
#    ✅ Model trained!

# [6/6] Evaluating model...

#    📊 Performance Metrics:
#    Accuracy:  86.07%
#    Precision: 88.21%
#    Recall:    81.90%
#    F1-Score:  84.94%

#    Classification Report:
#               precision    recall  f1-score   support

#   Legitimate       0.84      0.90      0.87       228
#     Phishing       0.88      0.82      0.85       210

#     accuracy                           0.86       438
#    macro avg       0.86      0.86      0.86       438
# weighted avg       0.86      0.86      0.86       438


#    Confusion Matrix:
#    True Negatives (Legit correctly identified):  205
#    False Positives (Legit marked as Phishing):   23
#    False Negatives (Phishing marked as Legit):   38
#    True Positives (Phishing correctly caught):   172

#    ⚠️ False Positive Rate: 10.09%

# ======================================================================
# TESTING ON KNOWN URLs
# ======================================================================

# Testing individual URLs:
# C:\Users\Geetha\OneDrive\Desktop\hackathon-phishing-detector\venv\Lib\site-packages\sklearn\utils\validation.py:2749: UserWarning: X does not have valid feature names, but StandardScaler was fitted with feature names
#   warnings.warn(

# ✅ https://www.google.com
#    Prediction: LEGITIMATE (99.4% confidence)
#    Expected: Should be LEGITIMATE
# C:\Users\Geetha\OneDrive\Desktop\hackathon-phishing-detector\venv\Lib\site-packages\sklearn\utils\validation.py:2749: UserWarning: X does not have valid feature names, but StandardScaler was fitted with feature names
#   warnings.warn(

# ✅ https://duckduckgo.com
#    Prediction: LEGITIMATE (98.3% confidence)
#    Expected: Should be LEGITIMATE
# C:\Users\Geetha\OneDrive\Desktop\hackathon-phishing-detector\venv\Lib\site-packages\sklearn\utils\validation.py:2749: UserWarning: X does not have valid feature names, but StandardScaler was fitted with feature names
#   warnings.warn(

# ✅ https://www.facebook.com
#    Prediction: LEGITIMATE (99.7% confidence)
#    Expected: Should be LEGITIMATE
# C:\Users\Geetha\OneDrive\Desktop\hackathon-phishing-detector\venv\Lib\site-packages\sklearn\utils\validation.py:2749: UserWarning: X does not have valid feature names, but StandardScaler was fitted with feature names
#   warnings.warn(

# ✅ https://github.com
#    Prediction: LEGITIMATE (100.0% confidence)
#    Expected: Should be LEGITIMATE
# C:\Users\Geetha\OneDrive\Desktop\hackathon-phishing-detector\venv\Lib\site-packages\sklearn\utils\validation.py:2749: UserWarning: X does not have valid feature names, but StandardScaler was fitted with feature names
#   warnings.warn(

# ✅ http://192.168.1.1/login.php
#    Prediction: PHISHING (91.7% confidence)
#    Expected: Should be PHISHING
# C:\Users\Geetha\OneDrive\Desktop\hackathon-phishing-detector\venv\Lib\site-packages\sklearn\utils\validation.py:2749: UserWarning: X does not have valid feature names, but StandardScaler was fitted with feature names
#   warnings.warn(

# ✅ http://secure-paypal-login.fake.com/verify
#    Prediction: PHISHING (77.5% confidence)
#    Expected: Should be PHISHING
# C:\Users\Geetha\OneDrive\Desktop\hackathon-phishing-detector\venv\Lib\site-packages\sklearn\utils\validation.py:2749: UserWarning: X does not have valid feature names, but StandardScaler was fitted with feature names
#   warnings.warn(

# ✅ http://account-update-required.xyz/signin
#    Prediction: PHISHING (78.5% confidence)
#    Expected: Should be PHISHING

# ======================================================================
# SAVING MODEL
# ======================================================================
#    📦 Old model backed up as phishing_model_backup.pkl
#    ✅ Model saved: phishing_model.pkl
#    ✅ Scaler saved: feature_scaler.pkl
#    ✅ Feature names saved: feature_names.pkl
#    ✅ Metadata saved: model_metadata.pkl

# ======================================================================
# 🎉 RETRAINING COMPLETE!
# ======================================================================

#    New model accuracy: 86.07%
#    False positive rate: 10.09%

#    Now restart your API: python backend_api.py
# ======================================================================