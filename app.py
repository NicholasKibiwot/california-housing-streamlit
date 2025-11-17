import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Version 2.0 - Model file integration complete
# Page Configuration
st.set_page_config(
    page_title="California Housing Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""<style>
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
.info-box {
    background-color: #f0f2f6;
    padding: 15px;
    border-radius: 5px;
    border-left: 4px solid #667eea;
}
</style>""", unsafe_allow_html=True)

# Model Loading Function with Caching
@st.cache_resource
def load_model():
    """Load the pre-trained KNN model from pickle file."""
    try:
        with open('california_knn_pipeline.pkl', 'rb') as f:
            model = pickle.load(f)
        return model, None
    except FileNotFoundError:
        return None, "Model file 'california_knn_pipeline.pkl' not found. Please ensure it is in the repository."
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

# Load Model
model, error_msg = load_model()

# Display Error if Model Not Found
if model is None:
    st.error(f"⚠️ {error_msg}")
    st.info("""
    ### How to Fix This:
    1. Ensure `california_knn_pipeline.pkl` exists in the GitHub repository root
    2. Commit and push the file to main branch
    3. Streamlit Cloud will automatically reload
    4. Hard refresh your browser (Ctrl+Shift+R)
    """)
    st.stop()

# Application Title and Header
st.title("🏠 California Housing Price Predictor")
st.markdown("#### Powered by K-Nearest Neighbors (KNN) Machine Learning Model")
st.markdown("---")

# Create Tabs for Different Sections
tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "📊 Model Performance", "📖 Feature Guide"])

# ==================== TAB 1: PREDICTION ====================
with tab1:
    st.header("Enter Housing Features for Prediction")
    st.markdown("Adjust the sliders to input housing features and get a price prediction.")
    
    # Create input columns
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.subheader("Income & Age")
        medinc = st.slider(
            "Median Income (in $10,000s)",
            min_value=0.5,
            max_value=15.0,
            value=3.5,
            step=0.1,
            help="Median income of the block"
        )
        houseage = st.slider(
            "House Age (years)",
            min_value=1,
            max_value=52,
            value=28,
            step=1,
            help="Median age of houses in the block"
        )
        
        st.subheader("Room Statistics")
        averoom = st.slider(
            "Average Rooms per Household",
            min_value=0.8,
            max_value=141.9,
            value=5.4,
            step=0.1,
            help="Mean number of rooms per household"
        )
        avebdrm = st.slider(
            "Average Bedrooms per Household",
            min_value=0.33,
            max_value=34.07,
            value=1.1,
            step=0.1,
            help="Mean number of bedrooms per household"
        )
        
    with col2:
        st.subheader("Population & Location")
        population = st.slider(
            "Population (Block Population)",
            min_value=3,
            max_value=35682,
            value=1425,
            step=50,
            help="Total population in the block"
        )
        aveocc = st.slider(
            "Average Occupancy",
            min_value=0.69,
            max_value=1243.33,
            value=3.07,
            step=0.1,
            help="Mean household occupancy"
        )
        
        st.subheader("Geographic Coordinates")
        lat = st.slider(
            "Latitude (North-South)",
            min_value=32.54,
            max_value=41.95,
            value=35.63,
            step=0.01,
            help="Geographic latitude for California locations"
        )
        lng = st.slider(
            "Longitude (East-West)",
            min_value=-124.35,
            max_value=-114.31,
            value=-119.57,
            step=0.01,
            help="Geographic longitude for California locations"
        )
        
    # Prediction Button
    st.markdown("---")
    if st.button("🎯 Predict House Price", use_container_width=True, type="primary"):
        # Prepare input data in correct feature order
        X = np.array([[medinc, houseage, averoom, avebdrm, population, aveocc, lat, lng]])
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        # Display prediction in a prominent box
        st.markdown(f"""
        <div class="prediction-box">
        💰 Predicted Price: ${prediction:.4f} (x100,000)<br>
        <small>≈ ${prediction * 100000:,.0f}</small>
        </div>
        """, unsafe_allow_html=True)
        
        # Display input summary
        st.subheader("Prediction Summary")
        summary_data = {
            'Feature': ['Median Income', 'House Age', 'Avg Rooms', 'Avg Bedrooms', 
                       'Population', 'Avg Occupancy', 'Latitude', 'Longitude'],
            'Value': [medinc, houseage, averoom, avebdrm, population, aveocc, lat, lng]
        }
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

# ==================== TAB 2: MODEL PERFORMANCE ====================
with tab2:
    st.header("Model Performance Metrics")
    st.markdown("This section displays the performance of the KNN regression model.")
    
    # Performance Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="R² Score", value="0.7221", delta="Test Set")
    with col2:
        st.metric(label="RMSE", value="0.6034", delta="Root Mean Sq Error")
    with col3:
        st.metric(label="Best CV R²", value="0.7313", delta="Cross-Validation")
    with col4:
        st.metric(label="Algorithm", value="KNN", delta="n_neighbors=9")
    
    st.markdown("---")
    
    # Cross-Validation Results
    st.subheader("Cross-Validation Results (16-Fold)")
    cv_scores = np.array([0.6958, 0.6980, 0.6646, 0.6665, 0.7217, 0.7244, 0.6861, 0.6897, 
                         0.7257, 0.7296, 0.6913, 0.6958, 0.7267, 0.7313, 0.6922, 0.6972])
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart of cross-validation scores
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(range(1, len(cv_scores) + 1), cv_scores, color='#667eea', alpha=0.7, edgecolor='black')
        mean_score = np.mean(cv_scores)
        ax.axhline(y=mean_score, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.4f}')
        ax.set_xlabel('Fold #', fontsize=11, fontweight='bold')
        ax.set_ylabel('R² Score', fontsize=11, fontweight='bold')
        ax.set_title('Cross-Validation Scores per Fold', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0.65, 0.75])
        st.pyplot(fig, use_container_width=True)
        
    with col2:
        # Statistics of CV scores
        st.markdown("### CV Scores Statistics")
        stats_data = {
            'Metric': ['Mean Score', 'Std Dev', 'Min Score', 'Max Score', 'Total Folds'],
            'Value': [
                f"{np.mean(cv_scores):.4f}",
                f"{np.std(cv_scores):.4f}",
                f"{np.min(cv_scores):.4f}",
                f"{np.max(cv_scores):.4f}",
                f"{len(cv_scores)}"
            ]
        }
        st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Model Details
    st.subheader("Model Details")
    model_info = {
        'Parameter': ['Algorithm', 'n_neighbors', 'Weights', 'Metric', 'Training Samples', 'Features Used'],
        'Value': ['K-Nearest Neighbors', '9', 'Distance-weighted', 'Minkowski (p=1)', '16,512', '8']
    }
    st.dataframe(pd.DataFrame(model_info), use_container_width=True, hide_index=True)

# ==================== TAB 3: FEATURE GUIDE ====================
with tab3:
    st.header("Feature Description & Guide")
    st.markdown("Understand what each feature represents and its valid range.")
    
    # Feature Information
    features_info = {
        'Feature': [
            'Median Income',
            'House Age',
            'Average Rooms',
            'Average Bedrooms',
            'Population',
            'Average Occupancy',
            'Latitude',
            'Longitude'
        ],
        'Description': [
            'Median income of households in the block',
            'Median age of houses in years',
            'Mean number of rooms per household',
            'Mean number of bedrooms per household',
            'Total population in the block',
            'Mean household occupancy (people per house)',
            'Geographic latitude (North-South position)',
            'Geographic longitude (East-West position)'
        ],
        'Min': [0.5, 1, 0.8, 0.33, 3, 0.69, 32.54, -124.35],
        'Max': [15.0, 52, 141.9, 34.07, 35682, 1243.33, 41.95, -114.31],
        'Unit': ['$10,000s', 'Years', 'Rooms', 'Bedrooms', 'People', 'Occupancy', 'Degrees', 'Degrees']
    }
    
    st.dataframe(pd.DataFrame(features_info), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Tips for Better Predictions
    st.subheader("💡 Tips for Better Predictions")
    st.info("""
    - **Income Impact**: Higher income areas generally have higher property values
    - **Location Matters**: Geographic coordinates (latitude/longitude) are crucial for price prediction
    - **Age Factor**: Newer houses tend to have different price patterns compared to older ones
    - **Occupancy**: Average occupancy can indicate neighborhood density and desirability
    - **Room Counts**: More rooms typically correlate with higher prices
    """)
    
    st.markdown("---")
    
    st.subheader("📚 About This Model")
    st.markdown("""
    **Dataset**: California Housing Dataset (16,512 samples)
    
    **Algorithm**: K-Nearest Neighbors (KNN) Regression
    - **Best Parameters**: n_neighbors=9, weights=distance, metric=minkowski (p=1)
    - **Cross-Validation**: 16-fold with mean R² of 0.7099
    
    **Target Variable**: Median house value (in $100,000s)
    
    **Performance**: 
    - R² Score: 0.7221 (explains ~72% of price variance)
    - RMSE: 0.6034 (average prediction error ~$60,340)
    """)

# Footer
st.markdown("---")
st.markdown("""<div style='text-align: center; font-size: 12px; color: gray;'>
 Created for the California Housing Price Prediction Bonus Assignment<br>
 Deployed on Streamlit Cloud | Data Source: Kaggle California Housing Dataset
</div>""", unsafe_allow_html=True)
