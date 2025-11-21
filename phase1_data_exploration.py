"""
Phishing Detection - Phase 1: Data Exploration
This script downloads the JSON dataset directly from Hugging Face
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import urllib.request
import os

print("=" * 60)
print("PHASE 1: PHISHING DATASET EXPLORATION")
print("=" * 60)

# Step 1: Download the dataset directly from Hugging Face
print("\n📥 Downloading URL dataset from Hugging Face...")
print("   (This is a 73MB file, may take 1-2 minutes...)")

dataset_url = "https://huggingface.co/datasets/ealvaradob/phishing-dataset/resolve/main/urls.json"
json_file = "phishing_urls.json"

# Check if already downloaded
if os.path.exists(json_file):
    print(f"✅ Dataset already downloaded: {json_file}")
else:
    try:
        urllib.request.urlretrieve(dataset_url, json_file)
        print(f"✅ Dataset downloaded successfully: {json_file}")
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("\n💡 Try downloading manually from:")
        print("   https://huggingface.co/datasets/ealvaradob/phishing-dataset/tree/main")
        print("   Download 'urls.json' and place it in the project folder")
        exit(1)

# Step 2: Load the JSON data
print("\n📂 Loading JSON data...")
try:
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    print("✅ Data loaded successfully!")
except Exception as e:
    print(f"❌ Error loading JSON: {e}")
    exit(1)

# Step 3: Explore dataset structure
print("\n" + "=" * 60)
print("DATASET STRUCTURE")
print("=" * 60)

print(f"\n📊 Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\n📋 Column names: {list(df.columns)}")

# Display first few rows
print("\n" + "=" * 60)
print("SAMPLE DATA (First 5 rows)")
print("=" * 60)
print(df.head())

# Step 4: Check data types and missing values
print("\n" + "=" * 60)
print("DATA QUALITY CHECK")
print("=" * 60)
print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
missing = df.isnull().sum()
print(missing)
print(f"\n✅ Total missing values: {missing.sum()}")

# Step 5: Analyze the labels
print("\n" + "=" * 60)
print("LABEL DISTRIBUTION")
print("=" * 60)

# The dataset has 'text' (URL) and 'label' (0=legitimate, 1=phishing)
label_column = 'label'
url_column = 'text'

if label_column in df.columns:
    print(f"\n📌 Label column: '{label_column}'")
    label_counts = df[label_column].value_counts().sort_index()
    print(f"\nLabel distribution:")
    print(label_counts)
    
    # Calculate percentages
    total = len(df)
    print(f"\nPercentages:")
    for label, count in label_counts.items():
        percentage = (count / total) * 100
        label_name = "Phishing" if label == 1 else "Legitimate"
        print(f"  {label_name} ({label}): {count:,} ({percentage:.2f}%)")
    
    # Check if balanced
    balance_ratio = min(label_counts) / max(label_counts)
    if balance_ratio > 0.8:
        print(f"\n✅ Dataset is well-balanced (ratio: {balance_ratio:.2f})")
    else:
        print(f"\n⚠️  Dataset is imbalanced (ratio: {balance_ratio:.2f})")
        print("   This is okay - we'll handle it during training!")

# Step 6: Analyze URLs
print("\n" + "=" * 60)
print("URL ANALYSIS")
print("=" * 60)

if url_column in df.columns:
    print(f"\n📌 URL column: '{url_column}'")
    
    # Basic statistics
    df['url_length'] = df[url_column].astype(str).str.len()
    print(f"\nURL length statistics:")
    print(df['url_length'].describe())
    
    # Check for duplicates
    duplicates = df[url_column].duplicated().sum()
    print(f"\n🔍 Duplicate URLs: {duplicates:,}")
    if duplicates > 0:
        print(f"   ({(duplicates/len(df)*100):.2f}% of dataset)")
    
    # Check for broken/incomplete URLs
    empty_urls = df[url_column].isnull().sum()
    very_short_urls = (df['url_length'] < 10).sum()
    print(f"\n🔍 Empty URLs: {empty_urls}")
    print(f"🔍 Very short URLs (<10 chars): {very_short_urls}")
    
    # Protocol analysis
    df['has_https'] = df[url_column].astype(str).str.contains('https://', case=False, na=False)
    df['has_http'] = df[url_column].astype(str).str.contains('http://', case=False, na=False)
    
    print(f"\n🔒 URLs with HTTPS: {df['has_https'].sum():,} ({(df['has_https'].sum()/len(df)*100):.1f}%)")
    print(f"🔓 URLs with HTTP: {df['has_http'].sum():,} ({(df['has_http'].sum()/len(df)*100):.1f}%)")
    
    # Show sample URLs
    print(f"\n📝 Sample URLs:")
    print(f"\n✅ Legitimate URLs (label=0):")
    legit_samples = df[df[label_column] == 0][url_column].head(3)
    for i, url in enumerate(legit_samples, 1):
        print(f"  {i}. {url[:80]}{'...' if len(url) > 80 else ''}")
    
    print(f"\n⚠️  Phishing URLs (label=1):")
    phishing_samples = df[df[label_column] == 1][url_column].head(3)
    for i, url in enumerate(phishing_samples, 1):
        print(f"  {i}. {url[:80]}{'...' if len(url) > 80 else ''}")

# Step 7: Create visualizations
print("\n" + "=" * 60)
print("CREATING VISUALIZATIONS")
print("=" * 60)

# Create a figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Phishing URL Dataset Analysis', fontsize=16, fontweight='bold')

# 1. Label distribution pie chart
label_counts.plot(kind='pie', ax=axes[0, 0], autopct='%1.1f%%', 
                  labels=['Legitimate', 'Phishing'],
                  colors=['#2ecc71', '#e74c3c'],
                  startangle=90)
axes[0, 0].set_title('Label Distribution', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('')

# 2. Label distribution bar chart
label_counts.plot(kind='bar', ax=axes[0, 1], color=['#2ecc71', '#e74c3c'])
axes[0, 1].set_title('URL Count by Type', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Label')
axes[0, 1].set_ylabel('Count')
axes[0, 1].set_xticklabels(['Legitimate (0)', 'Phishing (1)'], rotation=45)
axes[0, 1].grid(axis='y', alpha=0.3)

# Add count labels on bars
for i, (label, count) in enumerate(label_counts.items()):
    axes[0, 1].text(i, count, f'{count:,}', ha='center', va='bottom')

# 3. URL length distribution by label
df.boxplot(column='url_length', by=label_column, ax=axes[1, 0], 
           patch_artist=True)
axes[1, 0].set_title('URL Length Distribution by Type', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Label')
axes[1, 0].set_ylabel('URL Length (characters)')
axes[1, 0].set_xticklabels(['Legitimate (0)', 'Phishing (1)'])
plt.sca(axes[1, 0])
plt.xticks(rotation=0)

# 4. URL length histogram
axes[1, 1].hist([df[df[label_column]==0]['url_length'], 
                 df[df[label_column]==1]['url_length']], 
                label=['Legitimate', 'Phishing'], 
                bins=50, alpha=0.7, color=['#2ecc71', '#e74c3c'])
axes[1, 1].set_title('URL Length Distribution', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('URL Length (characters)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].legend()
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('dataset_analysis.png', dpi=300, bbox_inches='tight')
print("✅ Visualization saved as 'dataset_analysis.png'")
print("   📊 Open this file in VS Code to view the charts!")

# Step 8: Data Cleaning Summary
print("\n" + "=" * 60)
print("DATA CLEANING RECOMMENDATIONS")
print("=" * 60)

cleaning_needed = []
if duplicates > 0:
    cleaning_needed.append(f"Remove {duplicates:,} duplicate URLs")
if empty_urls > 0:
    cleaning_needed.append(f"Remove {empty_urls:,} empty URLs")
if very_short_urls > 0:
    cleaning_needed.append(f"Investigate {very_short_urls:,} very short URLs")

if cleaning_needed:
    print("\n⚠️  Cleaning steps needed:")
    for step in cleaning_needed:
        print(f"   • {step}")
else:
    print("\n✅ Dataset appears clean! No major issues detected.")

# Step 9: Save cleaned dataset
print("\n" + "=" * 60)
print("SAVING PROCESSED DATA")
print("=" * 60)

print(f"\n🧹 Cleaning data...")
# Remove duplicates
df_clean = df.drop_duplicates(subset=[url_column])
removed_dups = len(df) - len(df_clean)
print(f"   • Removed {removed_dups:,} duplicate rows")

# Remove empty/null URLs
df_clean = df_clean[df_clean[url_column].notna()]
df_clean = df_clean[df_clean['url_length'] >= 10]

# Remove rows with very short URLs (likely broken)
initial_clean = len(df_clean)
df_clean = df_clean[df_clean['url_length'] >= 15]
removed_short = initial_clean - len(df_clean)
print(f"   • Removed {removed_short:,} very short URLs")

print(f"\n📊 Final dataset size: {len(df_clean):,} rows")

# Keep only essential columns
df_clean = df_clean[[url_column, label_column]]

# Save to CSV
df_clean.to_csv('phishing_dataset_clean.csv', index=False)
print(f"\n✅ Cleaned dataset saved as 'phishing_dataset_clean.csv'")

# Final summary
print("\n" + "=" * 60)
print("PHASE 1 COMPLETE! 🎉")
print("=" * 60)

legit_count = len(df_clean[df_clean[label_column]==0])
phishing_count = len(df_clean[df_clean[label_column]==1])

print(f"\n📊 Summary:")
print(f"   • Original dataset: {len(df):,} URLs")
print(f"   • Cleaned dataset: {len(df_clean):,} URLs")
print(f"   • Legitimate URLs: {legit_count:,} ({(legit_count/len(df_clean)*100):.1f}%)")
print(f"   • Phishing URLs: {phishing_count:,} ({(phishing_count/len(df_clean)*100):.1f}%)")
print(f"   • Balance ratio: {min(legit_count, phishing_count)/max(legit_count, phishing_count):.2f}")

print(f"\n📁 Files created:")
print(f"   ✓ phishing_urls.json (raw data)")
print(f"   ✓ phishing_dataset_clean.csv (cleaned data - ready for Phase 2!)")
print(f"   ✓ dataset_analysis.png (visualizations)")

print(f"\n✅ Ready for Phase 2: Feature Engineering!")
print(f"\n💡 Next step: Run phase2_feature_engineering.py")




# ============================================================
# PHASE 1: PHISHING DATASET EXPLORATION
# ============================================================

# 📥 Downloading URL dataset from Hugging Face...
#    (This is a 73MB file, may take 1-2 minutes...)
# ✅ Dataset downloaded successfully: phishing_urls.json

# 📂 Loading JSON data...
# ✅ Data loaded successfully!

# ============================================================
# DATASET STRUCTURE
# ============================================================

# 📊 Dataset shape: 835697 rows × 2 columns

# 📋 Column names: ['text', 'label']

# ============================================================
# SAMPLE DATA (First 5 rows)
# ============================================================
#                                                 text  label
# 0      http://webmail-brinkster.com/ex/?email=%20%0%      1
# 1                         billsportsmaps.com/?p=1206      0
# 2  www.sanelyurdu.com/language/homebank.tsbbank.c...      1
# 3                          ee-billing.limited323.com      1
# 4                   indiadaily.com/bolly_archive.htm      0

# ============================================================
# DATA QUALITY CHECK
# ============================================================

# Data types:
# text     object
# label     int64
# dtype: object

# Missing values:
# text     0
# label    0
# dtype: int64

# ✅ Total missing values: 0

# ============================================================
# LABEL DISTRIBUTION
# ============================================================

# 📌 Label column: 'label'

# Label distribution:
# label
# 0    444933
# 1    390764
# Name: count, dtype: int64

# Percentages:
#   Legitimate (0): 444,933 (53.24%)
#   Phishing (1): 390,764 (46.76%)

# ✅ Dataset is well-balanced (ratio: 0.88)

# ============================================================
# URL ANALYSIS
# ============================================================

# 📌 URL column: 'text'

# URL length statistics:
# count    835697.000000
# mean         47.484266
# std          42.333049
# min           1.000000
# 25%          26.000000
# 50%          37.000000
# 75%          56.000000
# max        3992.000000
# Name: url_length, dtype: float64

# 🔍 Duplicate URLs: 14,295
#    (1.71% of dataset)

# 🔍 Empty URLs: 0
# 🔍 Very short URLs (<10 chars): 4991

# 🔒 URLs with HTTPS: 143,582 (17.2%)
# 🔓 URLs with HTTP: 85,138 (10.2%)

# 📝 Sample URLs:

# ✅ Legitimate URLs (label=0):
#   1. billsportsmaps.com/?p=1206
#   2. indiadaily.com/bolly_archive.htm
#   3. homepage.esoterica.pt/~jrfsousa/fortran.html

# ⚠️  Phishing URLs (label=1):
#   1. http://webmail-brinkster.com/ex/?email=%20%0%
#   2. www.sanelyurdu.com/language/homebank.tsbbank.co.nz/SignOn.htm
#   3. ee-billing.limited323.com

# ============================================================
# CREATING VISUALIZATIONS
# ============================================================
# ✅ Visualization saved as 'dataset_analysis.png'
#    📊 Open this file in VS Code to view the charts!

# ============================================================
# DATA CLEANING RECOMMENDATIONS
# ============================================================

# ⚠️  Cleaning steps needed:
#    • Remove 14,295 duplicate URLs
#    • Investigate 4,991 very short URLs

# ============================================================
# SAVING PROCESSED DATA
# ============================================================

# 🧹 Cleaning data...
#    • Removed 14,295 duplicate rows
#    • Removed 34,029 very short URLs

# 📊 Final dataset size: 782,382 rows

# ✅ Cleaned dataset saved as 'phishing_dataset_clean.csv'

# ============================================================
# PHASE 1 COMPLETE! 🎉
# ============================================================

# 📊 Summary:
#    • Original dataset: 835,697 URLs
#    • Cleaned dataset: 782,382 URLs
#    • Legitimate URLs: 427,459 (54.6%)
#    • Phishing URLs: 354,923 (45.4%)
#    • Balance ratio: 0.83

# 📁 Files created:
#    ✓ phishing_urls.json (raw data)
#    ✓ phishing_dataset_clean.csv (cleaned data - ready for Phase 2!)
#    ✓ dataset_analysis.png (visualizations)

# ✅ Ready for Phase 2: Feature Engineering!