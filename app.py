# -*- coding: utf-8 -*-
"""Refined app.py"""

import os
import joblib
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

# ======================
# Step 1: Model Creation and Saving (Once)
# ======================
# Create the 'models' directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

model_path = "models/sustainability_rf_model.pkl"

# Only save the model if it doesn't already exist
if not os.path.isfile(model_path):
    # Create a placeholder model (replace with your trained model)
    model = LinearRegression()
    joblib.dump(model, model_path)
    print(f"✅ Placeholder model saved at {model_path}")

# ======================
# Step 2: Load Model
# ======================
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error(f"Model file not found at {model_path}")
    st.stop()
except Exception as e:
    st.error(f"Error loading the model: {e}")
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

# One-hot encoding for categorical variables
df_encoded = pd.get_dummies(df, columns=['Product_Name', 'Material', 'Manufacturing_Country', 'Disposal_Method'])

# Ensure all expected columns exist
all_possible_columns = [
    'Weight_kg', 'Energy_Consumption_kWh', 'Water_Usage_Liters',
    'Carbon_Footprint_kgCO2', 'Waste_Generated_kg', 'Lifespan_Years', 'Impact_per_Year'
]
for option in product_options:
    all_possible_columns.append(f'Product_Name_{option}')
for option in material_options:
    all_possible_columns.append(f'Material_{option}')
for option in country_options:
    all_possible_columns.append(f'Manufacturing_Country_{option}')
for option in disposal_options:
    all_possible_columns.append(f'Disposal_Method_{option}')

# Add missing columns and set to 0
for col in all_possible_columns:
    if col not in df_encoded.columns:
        df_encoded[col] = 0

# Sort columns alphabetically (for placeholder model)
df_encoded = df_encoded[sorted(df_encoded.columns)]

# ======================
# Predict and Display
# ======================
if st.button("Predict Sustainability Rating"):
    try:
        rating = model.predict(df_encoded)[0]
        st.success(f"🌿 Predicted Sustainability Rating: {round(rating, 2)} / 5")
    except Exception as e:
        st.error(f"Prediction error: {e}")
