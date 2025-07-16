import os
import joblib
import pandas as pd
import streamlit as st

# ======================
# Step 1: Load Trained Model
# ======================
model_path = "models/sustainability_rf_model.pkl"

try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error(f"❌ Model file not found at {model_path}. Please train and save it first.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading the model: {e}")
    st.stop()

# ======================
# Streamlit UI Setup
# ======================
st.title("🌱 AI-Powered Lifecycle Analyzer")
st.markdown("Predict the sustainability rating of a product based on lifecycle data.")

# Dropdown options
product_options = ["Smartphone", "Laptop", "T-Shirt", "Plastic Bottle", "LED Bulb", "Desk Chair", "Refrigerator", "Sneakers"]
material_options = ["Plastic", "Metal", "Cotton", "Glass", "Polyester", "Leather", "Wood"]
country_options = ["India", "China", "Germany", "USA", "Vietnam", "Bangladesh"]
disposal_options = ["Recycled", "Landfill", "Incinerated", "Reused"]

# Input Widgets
product_name = st.selectbox("Product Name", product_options)
material = st.selectbox("Material", material_options)
manufacturing_country = st.selectbox("Manufacturing Country", country_options)
disposal_method = st.selectbox("Disposal Method", disposal_options)
weight = st.slider("Weight (kg)", 0.1, 20.0, 1.0)
energy = st.slider("Energy Consumption (kWh)", 5.0, 2000.0, 100.0)
water = st.slider("Water Usage (Liters)", 10.0, 5000.0, 500.0)
carbon = st.slider("Carbon Footprint (kgCO₂)", 1.0, 1000.0, 50.0)
waste = st.slider("Waste Generated (kg)", 0.05, 15.0, 1.0)
lifespan = st.slider("Lifespan (Years)", 1, 15, 5)

# Prepare DataFrame for prediction
df = pd.DataFrame([{
    "Product_Name": product_name,
    "Material": material,
    "Weight_kg": weight,
    "Energy_Consumption_kWh": energy,
    "Water_Usage_Liters": water,
    "Carbon_Footprint_kgCO2": carbon,
    "Waste_Generated_kg": waste,
    "Manufacturing_Country": manufacturing_country,
    "Disposal_Method": disposal_method,
    "Lifespan_Years": lifespan,
}])

# Feature engineering
df['Impact_per_Year'] = df['Carbon_Footprint_kgCO2'] / df['Lifespan_Years']

# One-hot encode the input to match training features
df_encoded = pd.get_dummies(df)

# Match features with model's trained features
model_features = model.feature_names_in_
for col in model_features:
    if col not in df_encoded.columns:
        df_encoded[col] = 0
df_encoded = df_encoded[model_features]

# ======================
# Predict and Display
# ======================
if st.button("Predict Sustainability Rating"):
    try:
        rating = model.predict(df_encoded)[0]
        st.success(f"🌿 Predicted Sustainability Rating: {round(rating, 2)} / 5")
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
