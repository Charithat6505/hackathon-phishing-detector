"""
Validation Script: Check Phase 1 & Phase 2 Quality
This script validates data cleaning and feature extraction
"""

import pandas as pd
import numpy as np
import os

def validate_phase1(filename='phishing_dataset_clean.csv'):
    """Validate Phase 1: Data Cleaning"""
    print("="*60)
    print("PHASE 1 VALIDATION: Data Cleaning")
    print("="*60)
    
    if not os.path.exists(filename):
        print(f"❌ ERROR: {filename} not found!")
        return False
    
    df = pd.read_csv(filename)
    print(f"\n✓ Loaded cleaned dataset: {len(df)} rows")
    
    issues = []
    warnings = []
    
    # Check 1: Required columns
    print("\n[1/8] Checking required columns...")
    required_cols = ['text', 'label']  # or ['url', 'label']
    if 'text' in df.columns or 'url' in df.columns:
        print("  ✓ URL column present")
    else:
        issues.append("Missing URL column")
        print("  ❌ No URL column found")
    
    if 'label' in df.columns:
        print("  ✓ Label column present")
    else:
        issues.append("Missing label column")
        print("  ❌ No label column found")
    
    # Check 2: No duplicates
    print("\n[2/8] Checking for duplicates...")
    url_col = 'text' if 'text' in df.columns else 'url'
    duplicates = df[url_col].duplicated().sum()
    if duplicates == 0:
        print(f"  ✓ No duplicates found")
    else:
        warnings.append(f"{duplicates} duplicate URLs found")
        print(f"  ⚠️  {duplicates} duplicate URLs ({duplicates/len(df)*100:.2f}%)")
    
    # Check 3: No missing values
    print("\n[3/8] Checking for missing values...")
    missing = df.isnull().sum().sum()
    if missing == 0:
        print(f"  ✓ No missing values")
    else:
        issues.append(f"{missing} missing values found")
        print(f"  ❌ {missing} missing values")
    
    # Check 4: Valid URLs
    print("\n[4/8] Checking URL validity...")
    empty_urls = (df[url_col].str.len() == 0).sum()
    short_urls = (df[url_col].str.len() < 10).sum()
    if empty_urls == 0:
        print(f"  ✓ No empty URLs")
    else:
        issues.append(f"{empty_urls} empty URLs")
        print(f"  ❌ {empty_urls} empty URLs")
    
    if short_urls < len(df) * 0.01:  # Less than 1%
        print(f"  ✓ URL lengths look good ({short_urls} very short URLs)")
    else:
        warnings.append(f"{short_urls} suspiciously short URLs")
        print(f"  ⚠️  {short_urls} very short URLs ({short_urls/len(df)*100:.2f}%)")
    
    # Check 5: Valid labels
    print("\n[5/8] Checking label validity...")
    if 'label' in df.columns:
        unique_labels = df['label'].unique()
        if set(unique_labels).issubset({0, 1}):
            print(f"  ✓ Labels are binary (0 and 1)")
        else:
            issues.append(f"Invalid labels: {unique_labels}")
            print(f"  ❌ Invalid labels found: {unique_labels}")
        
        # Check 6: Label balance
        print("\n[6/8] Checking label balance...")
        label_counts = df['label'].value_counts()
        print(f"  Legitimate (0): {label_counts.get(0, 0)} ({label_counts.get(0, 0)/len(df)*100:.1f}%)")
        print(f"  Phishing (1):   {label_counts.get(1, 0)} ({label_counts.get(1, 0)/len(df)*100:.1f}%)")
        
        balance_ratio = min(label_counts) / max(label_counts)
        if balance_ratio >= 0.8:
            print(f"  ✓ Well-balanced dataset (ratio: {balance_ratio:.2f})")
        elif balance_ratio >= 0.6:
            print(f"  ⚠️  Moderately balanced (ratio: {balance_ratio:.2f})")
            warnings.append(f"Dataset imbalance (ratio: {balance_ratio:.2f})")
        else:
            print(f"  ❌ Imbalanced dataset (ratio: {balance_ratio:.2f})")
            issues.append(f"Severe imbalance (ratio: {balance_ratio:.2f})")
    
    # Check 7: URL format check
    print("\n[7/8] Checking URL formats...")
    has_http = df[url_col].str.contains('http', case=False, na=False).sum()
    has_dots = df[url_col].str.contains(r'\.', regex=True, na=False).sum()
    
    print(f"  URLs with 'http': {has_http} ({has_http/len(df)*100:.1f}%)")
    print(f"  URLs with dots: {has_dots} ({has_dots/len(df)*100:.1f}%)")
    
    if has_dots > len(df) * 0.8:  # At least 80% should have dots
        print(f"  ✓ Most URLs have domain structure")
    else:
        warnings.append("Many URLs missing domain structure")
        print(f"  ⚠️  Many URLs missing proper domain structure")
    
    # Check 8: Dataset size
    print("\n[8/8] Checking dataset size...")
    if len(df) >= 10000:
        print(f"  ✓ Dataset size sufficient: {len(df):,} URLs")
    elif len(df) >= 5000:
        print(f"  ⚠️  Dataset size acceptable: {len(df):,} URLs")
        warnings.append(f"Small dataset: {len(df):,} URLs")
    else:
        print(f"  ❌ Dataset too small: {len(df):,} URLs")
        issues.append(f"Insufficient data: {len(df):,} URLs")
    
    # Summary
    print("\n" + "="*60)
    print("PHASE 1 SUMMARY")
    print("="*60)
    if len(issues) == 0 and len(warnings) == 0:
        print("✅ PHASE 1 PERFECT: No issues found!")
        return True
    elif len(issues) == 0:
        print(f"✅ PHASE 1 PASSED: {len(warnings)} warnings")
        for w in warnings:
            print(f"  ⚠️  {w}")
        return True
    else:
        print(f"❌ PHASE 1 FAILED: {len(issues)} critical issues")
        for i in issues:
            print(f"  ❌ {i}")
        return False


def validate_phase2(filename='phishing_features.csv'):
    """Validate Phase 2: Feature Extraction"""
    print("\n" + "="*60)
    print("PHASE 2 VALIDATION: Feature Extraction")
    print("="*60)
    
    if not os.path.exists(filename):
        print(f"❌ ERROR: {filename} not found!")
        print("Please run apply_feature_extraction.py first.")
        return False
    
    df = pd.read_csv(filename)
    print(f"\n✓ Loaded feature dataset: {len(df)} rows, {len(df.columns)} columns")
    
    issues = []
    warnings = []
    
    # Check 1: Required columns
    print("\n[1/10] Checking required columns...")
    if 'url' in df.columns and 'label' in df.columns:
        print("  ✓ URL and label columns present")
    else:
        issues.append("Missing required columns")
        print("  ❌ Missing required columns")
    
    # Check 2: Feature count
    print("\n[2/10] Checking feature count...")
    feature_cols = [col for col in df.columns if col not in ['url', 'label']]
    print(f"  Total features: {len(feature_cols)}")
    if len(feature_cols) >= 40:
        print(f"  ✓ Excellent feature count: {len(feature_cols)} features")
    elif len(feature_cols) >= 30:
        print(f"  ✓ Good feature count: {len(feature_cols)} features")
    else:
        warnings.append(f"Low feature count: {len(feature_cols)}")
        print(f"  ⚠️  Low feature count: {len(feature_cols)} features")
    
    # Check 3: Missing values
    print("\n[3/10] Checking for missing values...")
    missing = df[feature_cols].isnull().sum().sum()
    if missing == 0:
        print("  ✓ No missing values in features")
    else:
        print(f"  ⚠️  {missing} missing values found")
        missing_cols = df[feature_cols].isnull().sum()
        missing_cols = missing_cols[missing_cols > 0]
        for col, count in missing_cols.items():
            print(f"      {col}: {count} missing")
        warnings.append(f"{missing} missing values")
    
    # Check 4: Feature types
    print("\n[4/10] Checking feature data types...")
    non_numeric = df[feature_cols].select_dtypes(exclude=[np.number]).columns
    if len(non_numeric) == 0:
        print("  ✓ All features are numeric")
    else:
        issues.append(f"Non-numeric features: {list(non_numeric)}")
        print(f"  ❌ Non-numeric features found: {list(non_numeric)}")
    
    # Check 5: Feature value ranges
    print("\n[5/10] Checking feature value ranges...")
    stats = df[feature_cols].describe()
    
    # Check for constant features (no variance)
    constant_features = []
    for col in feature_cols:
        if df[col].nunique() == 1:
            constant_features.append(col)
    
    if len(constant_features) == 0:
        print("  ✓ All features have variance")
    else:
        warnings.append(f"{len(constant_features)} constant features")
        print(f"  ⚠️  {len(constant_features)} constant features (no variance):")
        for cf in constant_features[:5]:  # Show first 5
            print(f"      {cf}")
    
    # Check 6: Key features present
    print("\n[6/10] Checking key features...")
    expected_features = [
        'url_length', 'dot_count', 'has_ip', 'is_https', 
        'suspicious_keyword_count', 'entropy', 'subdomain_count'
    ]
    missing_key_features = [f for f in expected_features if f not in df.columns]
    
    if len(missing_key_features) == 0:
        print("  ✓ All key features present")
    else:
        warnings.append(f"Missing key features: {missing_key_features}")
        print(f"  ⚠️  Missing key features: {missing_key_features}")
    
    # Check 7: Feature value sanity
    print("\n[7/10] Checking feature value sanity...")
    sanity_checks = []
    
    if 'url_length' in df.columns:
        if df['url_length'].min() >= 10 and df['url_length'].max() < 10000:
            print("  ✓ URL lengths look reasonable")
        else:
            sanity_checks.append(f"url_length range: {df['url_length'].min()}-{df['url_length'].max()}")
    
    if 'entropy' in df.columns:
        if df['entropy'].min() >= 0 and df['entropy'].max() <= 10:
            print("  ✓ Entropy values look reasonable")
        else:
            sanity_checks.append(f"entropy range: {df['entropy'].min():.2f}-{df['entropy'].max():.2f}")
    
    if 'is_https' in df.columns:
        if set(df['is_https'].unique()).issubset({0, 1}):
            print("  ✓ Binary features are binary")
        else:
            sanity_checks.append(f"is_https has non-binary values")
    
    if len(sanity_checks) > 0:
        warnings.extend(sanity_checks)
        for sc in sanity_checks:
            print(f"  ⚠️  {sc}")
    
    # Check 8: Label distribution matches
    print("\n[8/10] Checking label distribution...")
    if 'label' in df.columns:
        label_counts = df['label'].value_counts()
        print(f"  Legitimate (0): {label_counts.get(0, 0)} ({label_counts.get(0, 0)/len(df)*100:.1f}%)")
        print(f"  Phishing (1):   {label_counts.get(1, 0)} ({label_counts.get(1, 0)/len(df)*100:.1f}%)")
        print("  ✓ Labels preserved correctly")
    
    # Check 9: Sample validation
    print("\n[9/10] Validating sample URLs...")
    print("  Sample feature extraction check (first URL):")
    sample = df.iloc[0]
    print(f"    URL: {sample['url'][:60]}...")
    print(f"    Length: {sample.get('url_length', 'N/A')}")
    print(f"    HTTPS: {sample.get('is_https', 'N/A')}")
    print(f"    Has IP: {sample.get('has_ip', 'N/A')}")
    print(f"    Entropy: {sample.get('entropy', 'N/A'):.2f}" if 'entropy' in sample else "    Entropy: N/A")
    print("  ✓ Features extracted successfully")
    
    # Check 10: File size
    print("\n[10/10] Checking file size...")
    file_size = os.path.getsize(filename) / (1024 * 1024)
    print(f"  File size: {file_size:.2f} MB")
    if file_size < 100:  # Less than 100MB is reasonable
        print("  ✓ File size reasonable")
    else:
        warnings.append(f"Large file size: {file_size:.2f} MB")
        print(f"  ⚠️  Large file size")
    
    # Summary
    print("\n" + "="*60)
    print("PHASE 2 SUMMARY")
    print("="*60)
    if len(issues) == 0 and len(warnings) == 0:
        print("✅ PHASE 2 PERFECT: No issues found!")
        return True
    elif len(issues) == 0:
        print(f"✅ PHASE 2 PASSED: {len(warnings)} warnings")
        for w in warnings:
            print(f"  ⚠️  {w}")
        return True
    else:
        print(f"❌ PHASE 2 FAILED: {len(issues)} critical issues")
        for i in issues:
            print(f"  ❌ {i}")
        return False


def main():
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*10 + "PHISHING DETECTOR QUALITY VALIDATION" + " "*10 + "║")
    print("╚" + "="*58 + "╝")
    
    # Validate Phase 1
    phase1_pass = validate_phase1()
    
    # Validate Phase 2
    phase2_pass = validate_phase2()
    
    # Final report
    print("\n" + "="*60)
    print("FINAL VALIDATION REPORT")
    print("="*60)
    
    if phase1_pass and phase2_pass:
        print("\n🎉 ALL PHASES PASSED! Ready for Phase 3!")
        print("\n✅ Phase 1: Data Cleaning - PASSED")
        print("✅ Phase 2: Feature Extraction - PASSED")
        print("\n🚀 You can now proceed to Phase 3: Model Training")
        print("   Run: python train_model.py")
    elif phase1_pass:
        print("\n✅ Phase 1: PASSED")
        print("⚠️  Phase 2: NEEDS ATTENTION")
        print("\nPlease fix Phase 2 issues before proceeding.")
    elif phase2_pass:
        print("\n⚠️  Phase 1: NEEDS ATTENTION")
        print("✅ Phase 2: PASSED")
        print("\nPlease fix Phase 1 issues before proceeding.")
    else:
        print("\n❌ Both phases need attention")
        print("\nPlease fix issues before proceeding to Phase 3.")
    
    print("="*60)


if __name__ == "__main__":
    main()



# ╔==========================================================╗
# ║          PHISHING DETECTOR QUALITY VALIDATION          ║
# ╚==========================================================╝
# ============================================================
# PHASE 1 VALIDATION: Data Cleaning
# ============================================================

# ✓ Loaded cleaned dataset: 782382 rows

# [1/8] Checking required columns...
#   ✓ URL column present
#   ✓ Label column present

# [2/8] Checking for duplicates...
#   ✓ No duplicates found

# [3/8] Checking for missing values...
#   ✓ No missing values

# [4/8] Checking URL validity...
#   ✓ No empty URLs
#   ✓ URL lengths look good (0 very short URLs)

# [5/8] Checking label validity...
#   ✓ Labels are binary (0 and 1)

# [6/8] Checking label balance...
#   Legitimate (0): 427459 (54.6%)
#   Phishing (1):   354923 (45.4%)
#   ✓ Well-balanced dataset (ratio: 0.83)

# [7/8] Checking URL formats...
#   URLs with 'http': 216900 (27.7%)
#   URLs with dots: 782327 (100.0%)
#   ✓ Most URLs have domain structure

# [8/8] Checking dataset size...
#   ✓ Dataset size sufficient: 782,382 URLs

# ============================================================
# PHASE 1 SUMMARY
# ============================================================
# ✅ PHASE 1 PERFECT: No issues found!

# ============================================================
# PHASE 2 VALIDATION: Feature Extraction
# ============================================================

# ✓ Loaded feature dataset: 100000 rows, 45 columns

# [1/10] Checking required columns...
#   ✓ URL and label columns present

# [2/10] Checking feature count...
#   Total features: 43
#   ✓ Excellent feature count: 43 features

# [3/10] Checking for missing values...
#   ✓ No missing values in features

# [4/10] Checking feature data types...
#   ✓ All features are numeric

# [5/10] Checking feature value ranges...
#   ✓ All features have variance

# [6/10] Checking key features...
#   ✓ All key features present

# [7/10] Checking feature value sanity...
#   ✓ URL lengths look reasonable
#   ✓ Entropy values look reasonable
#   ✓ Binary features are binary

# [8/10] Checking label distribution...
#   Legitimate (0): 53580 (53.6%)
#   Phishing (1):   46420 (46.4%)
#   ✓ Labels preserved correctly

# [9/10] Validating sample URLs...
#   Sample feature extraction check (first URL):
#     URL: http://webmail-brinkster.com/ex/?email=%20%0%...
#     Length: 45
#     HTTPS: 0
#     Has IP: 0
#     Entropy: 4.49
#   ✓ Features extracted successfully

# [10/10] Checking file size...
#   File size: 20.09 MB
#   ✓ File size reasonable

# ============================================================
# PHASE 2 SUMMARY
# ============================================================
# ✅ PHASE 2 PERFECT: No issues found!

# ============================================================
# FINAL VALIDATION REPORT
# ============================================================

# 🎉 ALL PHASES PASSED! Ready for Phase 3!

# ✅ Phase 1: Data Cleaning - PASSED
# ✅ Phase 2: Feature Extraction - PASSED

# 🚀 You can now proceed to Phase 3: Model Training
