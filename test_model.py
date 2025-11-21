import joblib

model = joblib.load('phishing_model.pkl')
scaler = joblib.load('feature_scaler.pkl')
features = joblib.load('feature_names.pkl')

print(f"Model expects {len(features)} features")
print(f"Scaler fitted on {scaler.n_features_in_} features")
print(f"Match: {len(features) == scaler.n_features_in_}")

# output:
# Model expects 43 features
# Scaler fitted on 43 features
# Match: True