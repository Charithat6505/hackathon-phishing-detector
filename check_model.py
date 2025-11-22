import joblib
import numpy as np

model = joblib.load('phishing_model.pkl')
feature_names = joblib.load('feature_names.pkl')

print('Feature importances (top 10):')
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

for i in range(10):
    idx = indices[i]
    print(f'  {feature_names[idx]}: {importances[idx]:.4f}')