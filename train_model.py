import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load your processed dataset
df = pd.read_csv("AI_Lifecycle_Analyzer_Updated.csv")  # Make sure this file exists

# Prepare training data
X = df.drop(['Sustainability_Rating', 'Product_ID'], axis=1, errors='ignore')
y = df['Sustainability_Rating']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/sustainability_rf_model.pkl")

print("✅ Model successfully trained and saved at models/sustainability_rf_model.pkl")
