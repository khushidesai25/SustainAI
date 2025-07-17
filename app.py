# -*- coding: utf-8 -*-
"""Refined app.py with easy-to-understand visualizations and clean imports"""

import os
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
import numpy as np

# ======================
# Step 1: Model Creation and Saving (Once)
# ======================
# Create the 'models' directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

model_path = "models/sustainability_rf_model.pkl"

# Only save the model if it doesn't already exist
if not os.path.isfile(model_path):
    # Create sample data for training the model
    np.random.seed(42)
    n_samples = 1000
    
    # Generate sample features
    sample_data = {
        'Weight_kg': np.random.uniform(0.1, 20.0, n_samples),
        'Energy_Consumption_kWh': np.random.uniform(5.0, 2000.0, n_samples),
        'Water_Usage_Liters': np.random.uniform(10.0, 5000.0, n_samples),
        'Carbon_Footprint_kgCO2': np.random.uniform(1.0, 1000.0, n_samples),
        'Waste_Generated_kg': np.random.uniform(0.05, 15.0, n_samples),
        'Lifespan_Years': np.random.uniform(1, 15, n_samples),
    }
    
    # Create sample DataFrame
    sample_df = pd.DataFrame(sample_data)
    
    # Create a simple sustainability score (target variable)
    # Lower values = more sustainable
    sustainability_score = (
        5 - (sample_df['Carbon_Footprint_kgCO2'] / 200) - 
        (sample_df['Energy_Consumption_kWh'] / 800) - 
        (sample_df['Water_Usage_Liters'] / 2000) - 
        (sample_df['Waste_Generated_kg'] / 5) + 
        (sample_df['Lifespan_Years'] / 10)
    ).clip(0, 5)
    
    # Create and fit the model
    model = LinearRegression()
    model.fit(sample_df, sustainability_score)
    
    # Save the fitted model
    joblib.dump(model, model_path)
    print(f"✅ Trained model saved at {model_path}")

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

# Sidebar for navigation
st.sidebar.title("📊 Navigation")
page = st.sidebar.selectbox("Choose a page", ["Input & Prediction", "Visualizations", "Impact Analysis"])

# Dropdown options
product_options = ["Smartphone", "Laptop", "T-Shirt", "Plastic Bottle", "LED Bulb", "Desk Chair", "Refrigerator", "Sneakers"]
material_options = ["Plastic", "Metal", "Cotton", "Glass", "Polyester", "Leather", "Wood"]
country_options = ["India", "China", "Germany", "USA", "Vietnam", "Bangladesh"]
disposal_options = ["Recycled", "Landfill", "Incinerated", "Reused"]

# Input Widgets
with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        product_name = st.selectbox("Product Name", product_options)
        material = st.selectbox("Material", material_options)
        manufacturing_country = st.selectbox("Manufacturing Country", country_options)
        disposal_method = st.selectbox("Disposal Method", disposal_options)
        weight = st.slider("Weight (kg)", 0.1, 20.0, 1.0)
        energy = st.slider("Energy Consumption (kWh)", 5.0, 2000.0, 100.0)
    
    with col2:
        water = st.slider("Water Usage (Liters)", 10.0, 5000.0, 500.0)
        carbon = st.slider("Carbon Footprint (kgCO₂)", 1.0, 1000.0, 50.0)
        waste = st.slider("Waste Generated (kg)", 0.05, 15.0, 1.0)
        lifespan = st.slider("Lifespan (Years)", 1, 15, 5)

# Prepare DataFrame for prediction
df = pd.DataFrame([{
    "Weight_kg": weight,
    "Energy_Consumption_kWh": energy,
    "Water_Usage_Liters": water,
    "Carbon_Footprint_kgCO2": carbon,
    "Waste_Generated_kg": waste,
    "Lifespan_Years": lifespan,
}])

# ======================
# Page Navigation
# ======================
if page == "Input & Prediction":
    st.header("🎯 Product Input & Prediction")
    
    # Predict and Display
    if st.button("Predict Sustainability Rating", type="primary"):
        try:
            rating = model.predict(df)[0]
            st.success(f"🌿 Predicted Sustainability Rating: {round(rating, 2)} / 5")
            
            # Display rating with a progress bar
            st.progress(rating / 5.0)
            
            # Rating interpretation
            if rating >= 4.0:
                st.info("🌟 Excellent! This product has a very high sustainability rating.")
            elif rating >= 3.0:
                st.info("✅ Good! This product has a moderate sustainability rating.")
            elif rating >= 2.0:
                st.warning("⚠️ Fair! This product could be more sustainable.")
            else:
                st.error("❌ Poor! This product has significant sustainability concerns.")
                
        except Exception as e:
            st.error(f"Prediction error: {e}")

elif page == "Visualizations":
    st.header("📊 Sustainability Visualizations")
    
    # 1. Environmental Impact Comparison (Sample Data)
    sample_products = ["Smartphone", "Laptop", "T-Shirt", "Plastic Bottle", "LED Bulb"]
    sample_data = pd.DataFrame({
        'Product': sample_products,
        'Carbon_Footprint': [45, 120, 8, 2, 15],
        'Energy_Consumption': [80, 200, 15, 5, 25],
        'Water_Usage': [400, 800, 2000, 10, 50],
        'Waste_Generated': [0.5, 2.0, 0.1, 0.05, 0.2]
    })
    
    st.subheader("🌍 Environmental Impact Comparison")
    fig_comparison = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Carbon Footprint (kgCO₂)', 'Energy Consumption (kWh)', 
                       'Water Usage (Liters)', 'Waste Generated (kg)'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    fig_comparison.add_trace(
        go.Bar(x=sample_data['Product'], y=sample_data['Carbon_Footprint'], 
               name='Carbon Footprint', marker_color='#FF6B6B'),
        row=1, col=1
    )
    fig_comparison.add_trace(
        go.Bar(x=sample_data['Product'], y=sample_data['Energy_Consumption'], 
               name='Energy Consumption', marker_color='#4ECDC4'),
        row=1, col=2
    )
    fig_comparison.add_trace(
        go.Bar(x=sample_data['Product'], y=sample_data['Water_Usage'], 
               name='Water Usage', marker_color='#45B7D1'),
        row=2, col=1
    )
    fig_comparison.add_trace(
        go.Bar(x=sample_data['Product'], y=sample_data['Waste_Generated'], 
               name='Waste Generated', marker_color='#96CEB4'),
        row=2, col=2
    )
    fig_comparison.update_layout(height=600, showlegend=False, title_text="Environmental Impact by Product")
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # 2. Current Product Impact Profile (Horizontal Bar Chart)
    st.subheader("🎯 Current Product Impact Profile")
    impact_factors = [
        'Carbon Footprint (kgCO₂)',
        'Energy Consumption (kWh)',
        'Water Usage (Liters)',
        'Waste Generated (kg)',
        'Weight (kg)'
    ]
    impact_values = [
        carbon,
        energy,
        water,
        waste,
        weight
    ]
    impact_df = pd.DataFrame({
        'Impact Factor': impact_factors,
        'Value': impact_values
    })
    fig_bar = px.bar(
        impact_df,
        x='Value',
        y='Impact Factor',
        orientation='h',
        title=f"Impact Profile: {product_name}",
        color='Impact Factor',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # 3. Material Impact Analysis (Grouped Bar Chart)
    st.subheader("🔬 Material Impact Analysis")
    material_impact = pd.DataFrame({
        'Material': material_options,
        'Carbon_Intensity': [2.5, 8.0, 1.2, 1.8, 3.2, 4.5, 0.8],
        'Energy_Intensity': [4.0, 12.0, 2.0, 3.0, 5.0, 6.0, 1.5],
        'Water_Intensity': [8.0, 20.0, 15.0, 5.0, 12.0, 25.0, 3.0]
    })
    material_impact_melted = material_impact.melt(id_vars='Material', var_name='Impact Type', value_name='Intensity')
    fig_material_bar = px.bar(
        material_impact_melted,
        x='Material',
        y='Intensity',
        color='Impact Type',
        barmode='group',
        title="Material Impact Comparison (Carbon, Energy, Water)"
    )
    st.plotly_chart(fig_material_bar, use_container_width=True)

elif page == "Impact Analysis":
    st.header("📈 Detailed Impact Analysis")
    
    # Create tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["📊 Impact Breakdown", "🌱 Sustainability Score", "📋 Recommendations"])
    
    with tab1:
        st.subheader("Environmental Impact Breakdown")
        total_impact = carbon + energy/10 + water/100 + waste*10
        impact_data = {
            'Impact Type': ['Carbon Footprint', 'Energy Consumption', 'Water Usage', 'Waste Generated'],
            'Value': [carbon, energy/10, water/100, waste*10],
            'Percentage': [
                (carbon/total_impact)*100,
                ((energy/10)/total_impact)*100,
                ((water/100)/total_impact)*100,
                ((waste*10)/total_impact)*100
            ]
        }
        impact_df = pd.DataFrame(impact_data)
        fig_pie = px.pie(
            impact_df, 
            values='Percentage', 
            names='Impact Type',
            title="Environmental Impact Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        fig_bar = px.bar(
            impact_df,
            x='Impact Type',
            y='Value',
            title="Absolute Impact Values",
            color='Impact Type',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with tab2:
        st.subheader("Sustainability Score Analysis")
        carbon_score = max(0, 5 - (carbon / 100))
        energy_score = max(0, 5 - (energy / 400))
        water_score = max(0, 5 - (water / 1000))
        waste_score = max(0, 5 - (waste / 3))
        lifespan_score = min(5, lifespan / 3)
        scores = {
            'Component': ['Carbon Footprint', 'Energy Efficiency', 'Water Usage', 'Waste Management', 'Lifespan'],
            'Score': [carbon_score, energy_score, water_score, waste_score, lifespan_score]
        }
        scores_df = pd.DataFrame(scores)
        fig_scores = px.bar(
            scores_df,
            x='Score',
            y='Component',
            orientation='h',
            title="Sustainability Score Breakdown",
            color='Score',
            color_continuous_scale='RdYlGn'
        )
        fig_scores.update_layout(xaxis_range=[0, 5])
        st.plotly_chart(fig_scores, use_container_width=True)
        overall_score = (carbon_score + energy_score + water_score + waste_score + lifespan_score) / 5
        st.metric("Overall Sustainability Score", f"{overall_score:.2f}/5")
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = overall_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Sustainability Rating"},
            delta = {'reference': 2.5},
            gauge = {
                'axis': {'range': [None, 5]},
                'bar': {'color': "darkgreen"},
                'steps': [
                    {'range': [0, 1], 'color': "lightgray"},
                    {'range': [1, 2], 'color': "yellow"},
                    {'range': [2, 3], 'color': "orange"},
                    {'range': [3, 4], 'color': "lightgreen"},
                    {'range': [4, 5], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 4.5
                }
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with tab3:
        st.subheader("🌿 Sustainability Recommendations")
        recommendations = []
        if carbon > 100:
            recommendations.append("🔴 High carbon footprint - Consider using renewable energy sources")
        if energy > 500:
            recommendations.append("🔴 High energy consumption - Look for energy-efficient alternatives")
        if water > 2000:
            recommendations.append("🔴 High water usage - Consider water-saving manufacturing processes")
        if waste > 5:
            recommendations.append("🔴 High waste generation - Implement waste reduction strategies")
        if lifespan < 3:
            recommendations.append("🔴 Short lifespan - Design for durability and repairability")
        if carbon <= 50 and energy <= 200 and water <= 1000 and waste <= 2 and lifespan >= 5:
            recommendations.append("🟢 Excellent sustainability profile!")
        if not recommendations:
            recommendations.append("🟡 Moderate sustainability - Room for improvement")
        for rec in recommendations:
            st.write(rec)
        improvement_data = {
            'Aspect': ['Carbon Footprint', 'Energy Usage', 'Water Usage', 'Waste Generation', 'Lifespan'],
            'Current': [carbon, energy, water, waste, lifespan],
            'Target': [50, 200, 1000, 2, 8]
        }
        improvement_df = pd.DataFrame(improvement_data)
        improvement_df['Improvement_Potential'] = (
            (improvement_df['Current'] - improvement_df['Target']) / improvement_df['Current'] * 100
        ).clip(lower=0)
        fig_improvement = px.bar(
            improvement_df,
            x='Aspect',
            y='Improvement_Potential',
            title="Improvement Potential (%)",
            color='Improvement_Potential',
            color_continuous_scale='RdYlGn_r'
        )
        st.plotly_chart(fig_improvement, use_container_width=True)
