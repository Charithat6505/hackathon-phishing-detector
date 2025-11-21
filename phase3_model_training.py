"""
Phishing Detection - Phase 3: Model Training
This script trains multiple ML models and selects the best one
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc, roc_auc_score)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("PHASE 3: MACHINE LEARNING MODEL TRAINING")
print("=" * 70)

# Step 1: Load the feature dataset
print("\n[1/7] Loading feature dataset...")

# Try different possible filenames
possible_files = ['phishing_features.csv', 'phishing_dataset_features.csv', 'features.csv']
df = None

for filename in possible_files:
    try:
        df = pd.read_csv(filename)
        print(f"✅ Loaded {len(df):,} URLs from '{filename}'")
        print(f"   Total columns: {len(df.columns)}")
        break
    except FileNotFoundError:
        continue

if df is None:
    print(f"❌ Could not find feature dataset file")
    print(f"   Tried: {possible_files}")
    print("💡 Make sure you ran Phase 2 (apply_feature_extraction.py)")
    exit(1)

# Step 2: Prepare features and labels
print("\n[2/7] Preparing data for training...")

# Find the URL column (could be 'text', 'url', or 'URL')
url_column = None
for col in ['text', 'url', 'URL', 'urls']:
    if col in df.columns:
        url_column = col
        break

if url_column is None:
    print(f"❌ Could not find URL column. Available columns: {df.columns.tolist()}")
    exit(1)

print(f"   Found URL column: '{url_column}'")

# Separate features (X) and labels (y)
X = df.drop([url_column, 'label'], axis=1)
y = df['label']

print(f"   Features shape: {X.shape}")
print(f"   Labels shape: {y.shape}")
print(f"   Feature columns: {list(X.columns[:5])}... (showing first 5)")

# Check class distribution
class_dist = y.value_counts()
print(f"\n   Class distribution:")
print(f"   Legitimate (0): {class_dist[0]:,} ({class_dist[0]/len(y)*100:.1f}%)")
print(f"   Phishing (1):   {class_dist[1]:,} ({class_dist[1]/len(y)*100:.1f}%)")

# Step 3: Split data into train and test sets
print("\n[3/7] Splitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   Training set: {len(X_train):,} samples")
print(f"   Test set:     {len(X_test):,} samples")

# Step 4: Feature scaling (important for Logistic Regression)
print("\n[4/7] Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("   ✅ Features scaled using StandardScaler")

# Step 5: Train multiple models
print("\n[5/7] Training machine learning models...")
print("   This may take 2-5 minutes...")

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\n   Training {name}...")
    start_time = time.time()
    
    # Use scaled data for Logistic Regression, original for tree-based
    if name == 'Logistic Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    training_time = time.time() - start_time
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'training_time': training_time
    }
    
    print(f"      ✅ Trained in {training_time:.2f}s")
    print(f"      Accuracy:  {accuracy*100:.2f}%")
    print(f"      Precision: {precision*100:.2f}%")
    print(f"      Recall:    {recall*100:.2f}%")
    print(f"      F1-Score:  {f1*100:.2f}%")

# Step 6: Select the best model
print("\n[6/7] Selecting best model...")

# Sort by F1-score (balance between precision and recall)
best_model_name = max(results, key=lambda x: results[x]['f1'])
best_model_info = results[best_model_name]
best_model = best_model_info['model']

print(f"\n   🏆 Best Model: {best_model_name}")
print(f"   📊 Performance Metrics:")
print(f"      Accuracy:  {best_model_info['accuracy']*100:.2f}%")
print(f"      Precision: {best_model_info['precision']*100:.2f}%")
print(f"      Recall:    {best_model_info['recall']*100:.2f}%")
print(f"      F1-Score:  {best_model_info['f1']*100:.2f}%")
print(f"      ROC-AUC:   {best_model_info['roc_auc']:.4f}")

# Step 7: Save the model and scaler
print("\n[7/7] Saving model and preprocessing components...")

# Save the best model
joblib.dump(best_model, 'phishing_model.pkl')
print(f"   ✅ Model saved: phishing_model.pkl")

# Save the scaler
joblib.dump(scaler, 'feature_scaler.pkl')
print(f"   ✅ Scaler saved: feature_scaler.pkl")

# Save feature names for later use
feature_names = list(X.columns)
joblib.dump(feature_names, 'feature_names.pkl')
print(f"   ✅ Feature names saved: feature_names.pkl")

# Save model metadata
metadata = {
    'model_name': best_model_name,
    'accuracy': best_model_info['accuracy'],
    'precision': best_model_info['precision'],
    'recall': best_model_info['recall'],
    'f1_score': best_model_info['f1'],
    'roc_auc': best_model_info['roc_auc'],
    'training_time': best_model_info['training_time'],
    'feature_count': len(feature_names),
    'training_samples': len(X_train),
    'test_samples': len(X_test)
}
joblib.dump(metadata, 'model_metadata.pkl')
print(f"   ✅ Metadata saved: model_metadata.pkl")

# Step 8: Create visualizations
print("\n" + "=" * 70)
print("CREATING PERFORMANCE VISUALIZATIONS")
print("=" * 70)

fig = plt.figure(figsize=(20, 12))

# 1. Model Comparison Bar Chart
ax1 = plt.subplot(2, 3, 1)
metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
x = np.arange(len(models))
width = 0.15

for i, metric in enumerate(metrics):
    values = [results[model][metric] for model in models.keys()]
    ax1.bar(x + i*width, values, width, label=metric.upper())

ax1.set_xlabel('Models')
ax1.set_ylabel('Score')
ax1.set_title('Model Performance Comparison', fontweight='bold', fontsize=14)
ax1.set_xticks(x + width * 2)
ax1.set_xticklabels(models.keys(), rotation=15, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim([0, 1.05])

# 2. Confusion Matrix for Best Model
ax2 = plt.subplot(2, 3, 2)
cm = confusion_matrix(y_test, best_model_info['predictions'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
ax2.set_title(f'Confusion Matrix - {best_model_name}', fontweight='bold', fontsize=14)
ax2.set_ylabel('True Label')
ax2.set_xlabel('Predicted Label')
ax2.set_xticklabels(['Legitimate (0)', 'Phishing (1)'])
ax2.set_yticklabels(['Legitimate (0)', 'Phishing (1)'])

# Add percentage annotations
total = cm.sum()
for i in range(2):
    for j in range(2):
        percentage = cm[i, j] / total * 100
        ax2.text(j+0.5, i+0.7, f'({percentage:.1f}%)', 
                ha='center', va='center', fontsize=9, color='gray')

# 3. ROC Curves
ax3 = plt.subplot(2, 3, 3)
for name, result in results.items():
    fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
    ax3.plot(fpr, tpr, label=f"{name} (AUC = {result['roc_auc']:.4f})")

ax3.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
ax3.set_xlabel('False Positive Rate')
ax3.set_ylabel('True Positive Rate')
ax3.set_title('ROC Curves', fontweight='bold', fontsize=14)
ax3.legend()
ax3.grid(alpha=0.3)

# 4. Feature Importance (for best model if it's tree-based)
ax4 = plt.subplot(2, 3, 4)
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    feature_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False).head(15)
    
    ax4.barh(range(len(feature_imp)), feature_imp['importance'], color='steelblue')
    ax4.set_yticks(range(len(feature_imp)))
    ax4.set_yticklabels(feature_imp['feature'])
    ax4.set_xlabel('Importance')
    ax4.set_title('Top 15 Feature Importances', fontweight='bold', fontsize=14)
    ax4.invert_yaxis()
    ax4.grid(axis='x', alpha=0.3)
else:
    ax4.text(0.5, 0.5, 'Feature importance\nnot available for\nLogistic Regression', 
            ha='center', va='center', fontsize=12)
    ax4.set_title('Feature Importances', fontweight='bold', fontsize=14)
    ax4.axis('off')

# 5. Metrics Comparison Table
ax5 = plt.subplot(2, 3, 5)
ax5.axis('off')

table_data = []
for name, result in results.items():
    table_data.append([
        name,
        f"{result['accuracy']*100:.2f}%",
        f"{result['precision']*100:.2f}%",
        f"{result['recall']*100:.2f}%",
        f"{result['f1']*100:.2f}%",
        f"{result['training_time']:.2f}s"
    ])

table = ax5.table(cellText=table_data,
                  colLabels=['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'Time'],
                  cellLoc='center',
                  loc='center',
                  bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Highlight best model row
for i, name in enumerate(models.keys()):
    if name == best_model_name:
        for j in range(6):
            table[(i+1, j)].set_facecolor('#90EE90')

ax5.set_title('Performance Summary', fontweight='bold', fontsize=14, pad=20)

# 6. Prediction Distribution
ax6 = plt.subplot(2, 3, 6)
bins = np.linspace(0, 1, 50)
ax6.hist(best_model_info['probabilities'][y_test == 0], bins=bins, alpha=0.5, 
         label='Legitimate URLs', color='green')
ax6.hist(best_model_info['probabilities'][y_test == 1], bins=bins, alpha=0.5, 
         label='Phishing URLs', color='red')
ax6.axvline(x=0.5, color='black', linestyle='--', label='Decision Threshold')
ax6.set_xlabel('Predicted Probability (Phishing)')
ax6.set_ylabel('Frequency')
ax6.set_title('Prediction Probability Distribution', fontweight='bold', fontsize=14)
ax6.legend()
ax6.grid(axis='y', alpha=0.3)

plt.suptitle('Phishing Detection Model - Performance Analysis', 
             fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
print("\n✅ Visualization saved: model_performance.png")

# Print detailed classification report
print("\n" + "=" * 70)
print(f"DETAILED CLASSIFICATION REPORT - {best_model_name}")
print("=" * 70)
print(classification_report(y_test, best_model_info['predictions'], 
                          target_names=['Legitimate', 'Phishing']))

# Print confusion matrix details
print("\n" + "=" * 70)
print("CONFUSION MATRIX BREAKDOWN")
print("=" * 70)
tn, fp, fn, tp = cm.ravel()
print(f"True Negatives (Legitimate correctly identified):  {tn:,} ({tn/len(y_test)*100:.2f}%)")
print(f"False Positives (Legitimate marked as Phishing):   {fp:,} ({fp/len(y_test)*100:.2f}%)")
print(f"False Negatives (Phishing marked as Legitimate):   {fn:,} ({fn/len(y_test)*100:.2f}%)")
print(f"True Positives (Phishing correctly identified):    {tp:,} ({tp/len(y_test)*100:.2f}%)")

# Final summary
print("\n" + "=" * 70)
print("PHASE 3 COMPLETE! 🎉")
print("=" * 70)
print(f"\n✅ Best Model: {best_model_name}")
print(f"✅ Accuracy: {best_model_info['accuracy']*100:.2f}%")
print(f"✅ Model saved and ready for deployment")
print(f"\n📁 Files created:")
print(f"   • phishing_model.pkl (trained model)")
print(f"   • feature_scaler.pkl (preprocessing)")
print(f"   • feature_names.pkl (feature list)")
print(f"   • model_metadata.pkl (model info)")
print(f"   • model_performance.png (visualizations)")
print(f"\n🚀 Ready for Phase 4: Backend API Development!")


# ======================================================================
# PHASE 3: MACHINE LEARNING MODEL TRAINING
# ======================================================================

# [1/7] Loading feature dataset...
# ✅ Loaded 100,000 URLs from 'phishing_features.csv'
#    Total columns: 45

# [2/7] Preparing data for training...
#    Found URL column: 'url'
#    Features shape: (100000, 43)
#    Labels shape: (100000,)
#    Feature columns: ['url_length', 'hostname_length', 'path_length', 'query_length', 'dot_count']... (showing first 5)

#    Class distribution:
#    Legitimate (0): 53,580 (53.6%)
#    Phishing (1):   46,420 (46.4%)

# [3/7] Splitting data (80% train, 20% test)...
#    Training set: 80,000 samples
#    Test set:     20,000 samples

# [4/7] Scaling features...
#    ✅ Features scaled using StandardScaler

# [5/7] Training machine learning models...
#    This may take 2-5 minutes...

#    Training Logistic Regression...
#       ✅ Trained in 0.57s
#       Accuracy:  83.80%
#       Precision: 85.52%
#       Recall:    78.38%
#       F1-Score:  81.80%

#    Training Random Forest...
#       ✅ Trained in 2.61s
#       Accuracy:  93.95%
#       Precision: 94.89%
#       Recall:    91.92%
#       F1-Score:  93.38%

#    Training Gradient Boosting...
#       ✅ Trained in 11.65s
#       Accuracy:  90.81%
#       Precision: 93.13%
#       Recall:    86.59%
#       F1-Score:  89.74%

# [6/7] Selecting best model...

#    🏆 Best Model: Random Forest
#    📊 Performance Metrics:
#       Accuracy:  93.95%
#       Precision: 94.89%
#       Recall:    91.92%
#       F1-Score:  93.38%
#       ROC-AUC:   0.9841

# [7/7] Saving model and preprocessing components...
#    ✅ Model saved: phishing_model.pkl
#    ✅ Scaler saved: feature_scaler.pkl
#    ✅ Feature names saved: feature_names.pkl
#    ✅ Metadata saved: model_metadata.pkl

# ======================================================================
# CREATING PERFORMANCE VISUALIZATIONS
# ======================================================================

# ✅ Visualization saved: model_performance.png

# ======================================================================
# DETAILED CLASSIFICATION REPORT - Random Forest
# ======================================================================
#               precision    recall  f1-score   support

#   Legitimate       0.93      0.96      0.94     10716
#     Phishing       0.95      0.92      0.93      9284

#     accuracy                           0.94     20000
#    macro avg       0.94      0.94      0.94     20000
# weighted avg       0.94      0.94      0.94     20000


# ======================================================================
# CONFUSION MATRIX BREAKDOWN
# ======================================================================
# True Negatives (Legitimate correctly identified):  10,256 (51.28%)
# False Positives (Legitimate marked as Phishing):   460 (2.30%)
# False Negatives (Phishing marked as Legitimate):   750 (3.75%)
# True Positives (Phishing correctly identified):    8,534 (42.67%)

# ======================================================================
# PHASE 3 COMPLETE! 🎉
# ======================================================================

# ✅ Best Model: Random Forest
# ✅ Accuracy: 93.95%
# ✅ Model saved and ready for deployment

# 📁 Files created:
#    • phishing_model.pkl (trained model)
#    • feature_scaler.pkl (preprocessing)
#    • feature_names.pkl (feature list)
#    • model_metadata.pkl (model info)
#    • model_performance.png (visualizations)

# 🚀 Ready for Phase 4: Backend API Development!