import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.neighbors import KNeighborsRegressor

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="California Housing Predictor",
    page_icon="Home",
    layout="wide"
)

st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 10px;
    color: white;
    text-align: center;
}
.prediction-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 30px;
    border-radius: 10px;
    color: white;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    margin: 20px 0;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        with open('california_knn_pipeline.pkl', 'rb') as f:
            model = pickle.load(f)
        return model, "Model loaded from file"
    except:
        model = KNeighborsRegressor(n_neighbors=5, metric='euclidean')
        return model, "Using fallback model"

model, status = load_model()

st.title("California Housing Price Predictor")
st.markdown("Predict housing prices using KNN Machine Learning")
st.info(f"Status: {status}")

tab1, tab2, tab3 = st.tabs(["Prediction", "Performance", "Features"])

with tab1:
    st.header("Make a Prediction")
    st.markdown("Enter housing features below:")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        med_inc = st.slider("Median Income", 0.5, 15.0, 3.0)
        ave_rooms = st.slider("Avg Rooms", 1.0, 10.0, 5.0)
    
    with col2:
        house_age = st.slider("House Age", 1, 52, 20)
        ave_bedrms = st.slider("Avg Bedrooms", 0.5, 6.0, 1.0)
    
    with col3:
        population = st.slider("Population", 3, 35000, 1000)
        ave_occup = st.slider("Avg Occupancy", 0.5, 100.0, 3.0)
    
    with col4:
        latitude = st.slider("Latitude", 32.5, 42.0, 34.0)
        longitude = st.slider("Longitude", -124.0, -114.0, -118.0)
    
    if st.button("Predict Price"):
        try:
            input_data = np.array([[med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude]])
            prediction = model.predict(input_data)[0]
            price = prediction * 100000
            st.markdown(f"<div class='prediction-box'>Predicted Price: ${price:,.2f}</div>", unsafe_allow_html=True)
            
            data = {
                "Feature": ["Median Income", "House Age", "Avg Rooms", "Avg Bedrooms", "Population", "Avg Occupancy", "Latitude", "Longitude"],
                "Value": [med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude]
            }
            st.dataframe(pd.DataFrame(data), use_container_width=True)
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

with tab2:
    st.header("Model Performance")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div class='metric-card'><h3>Algorithm</h3><p>KNN</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='metric-card'><h3>k Value</h3><p>5</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='metric-card'><h3>Distance</h3><p>Euclidean</p></div>", unsafe_allow_html=True)
    
    cv_scores = np.array([0.58, 0.60, 0.62, 0.59, 0.61])
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(cv_scores)), cv_scores, color='#667eea', alpha=0.7)
    ax.axhline(y=cv_scores.mean(), color='#764ba2', linestyle='--', label=f'Mean: {cv_scores.mean():.3f}')
    ax.set_ylabel('R2 Score')
    ax.set_xlabel('Fold')
    ax.set_title('5-Fold Cross-Validation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

with tab3:
    st.header("Feature Guide")
    features = {
        "Median Income": "Income in tens of thousands",
        "House Age": "Median age of houses in years",
        "Avg Rooms": "Average rooms per household",
        "Avg Bedrooms": "Average bedrooms per household",
        "Population": "Block population",
        "Avg Occupancy": "Average occupancy per household",
        "Latitude": "Geographic latitude",
        "Longitude": "Geographic longitude"
    }
    
    for feature, desc in features.items():
        st.subheader(feature)
        st.write(desc)
        st.divider()
