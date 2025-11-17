import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="California Housing Predictor", page_icon="🏠", layout="wide")

@st.cache_resource
def load_model():
    try:
        with open('california_knn_pipeline.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Model file not found!")
        return None

model = load_model()
if model is None:
    st.stop()

st.title("🏠 California Housing Price Predictor")
st.markdown("### Predict house prices with KNN Machine Learning")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["Prediction", "Model Info", "Guide"])

with tab1:
    st.header("Enter Housing Features")
    col1, col2 = st.columns(2)
    
    with col1:
        medinc = st.slider("Median Income", 0.5, 15.0, 3.5)
        houseage = st.slider("House Age", 1, 52, 28)
        averoom = st.slider("Average Rooms", 0.8, 141.9, 5.4)
        avebdrm = st.slider("Average Bedrooms", 0.33, 34.07, 1.1)
    
    with col2:
        population = st.slider("Population", 3, 35682, 1425)
        aveocc = st.slider("Average Occupancy", 0.69, 1243.33, 3.07)
        lat = st.slider("Latitude", 32.54, 41.95, 35.63)
        lng = st.slider("Longitude", -124.35, -114.31, -119.57)
    
    if st.button("🎯 Predict Price", use_container_width=True):
        X = np.array([[medinc, houseage, averoom, avebdrm, population, aveocc, lat, lng]])
        price = model.predict(X)[0]
        st.markdown(f"<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; text-align: center; font-size: 28px; font-weight: bold;'>💰 Predicted Price: ${price:.4f} (x100,000)</div>", unsafe_allow_html=True)

with tab2:
    st.header("Model Performance")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R² Score", "0.7221")
    col2.metric("RMSE", "0.6034")
    col3.metric("Best CV R²", "0.7313")
    col4.metric("Algorithm", "KNN")
    
    st.markdown("---")
    st.subheader("Cross-Validation Results")
    cv_scores = [0.6958, 0.6980, 0.6646, 0.6665, 0.7217, 0.7244, 0.6861, 0.6897, 0.7257, 0.7296, 0.6913, 0.6958, 0.7267, 0.7313, 0.6922, 0.6972]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(1, len(cv_scores) + 1), cv_scores, color='#667eea', alpha=0.7)
    ax.axhline(y=np.mean(cv_scores), color='red', linestyle='--', label=f'Mean: {np.mean(cv_scores):.4f}')
    ax.set_xlabel('Fold #')
    ax.set_ylabel('R² Score')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    st.pyplot(fig, use_container_width=True)

with tab3:
    st.header("Feature Guide")
    st.markdown("""
    **Median Income:** Median income of households (in tens of thousands)
    
    **House Age:** Age of the house in years
    
    **Average Rooms:** Average number of rooms per household
    
    **Average Bedrooms:** Average number of bedrooms per household
    
    **Population:** Block population
    
    **Average Occupancy:** Average household occupancy
    
    **Latitude & Longitude:** Geographic coordinates for California locations
    """)

st.markdown("---")
st.markdown("Created for the California Housing Price Prediction bonus assignment")
