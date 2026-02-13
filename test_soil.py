import joblib
import numpy as np

# Load the model
model = joblib.load('models/xgboost_soil_health_model.pkl')

# Test with healthy values
healthy_data = [6.3, 0.3, 1.0, 300, 30, 280, 15, 1.8, 0.7, 8, 4, 1.0]  # PH, EC, OC, N, P, K, S, Zn, B, Fe, Mn, Cu
healthy_pred = model.predict_proba([healthy_data])[0]
print(f"Healthy input → Prediction: {healthy_pred} → Label: {'healthy' if np.argmax(healthy_pred) == 0 else 'diseased'}")

# Test with unhealthy values
unhealthy_data = [4.5, 0.8, 0.4, 100, 10, 150, 5, 0.3, 0.2, 2, 1, 0.1]  # Outside normal ranges
unhealthy_pred = model.predict_proba([unhealthy_data])[0]
print(f"Unhealthy input → Prediction: {unhealthy_pred} → Label: {'healthy' if np.argmax(healthy_pred) == 0 else 'diseased'}")