"""
House Price Prediction — Streamlit App
Author: [Your Name]
Date: 2024
Description: Interactive web app to predict house prices using a trained ML model.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# ============================================================
# PAGE CONFIGURATION (must be the first Streamlit command)
# ============================================================
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS — make it look professional
# ============================================================
st.markdown("""
    <style>
    /* Main container padding */
    .main {
        padding: 1rem 3rem;
    }

    /* Big title */
    .big-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Prediction box */
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }

    .prediction-price {
        font-size: 2.8rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }

    /* Metric cards */
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f4e79;
        margin: 0.5rem 0;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #999;
        font-size: 0.85rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #eee;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL & METADATA (cached — loads only once)
# ============================================================
@st.cache_resource
def load_model():
    """Load the trained pipeline."""
    model_path = 'house_price_model.pkl'
    if not os.path.exists(model_path):
        st.error(f"❌ Model file '{model_path}' not found. Please run the training notebook first.")
        st.stop()
    return joblib.load(model_path)


@st.cache_data
def load_metadata():
    """Load model metadata (features, categories, metrics)."""
    meta_path = 'model_metadata.json'
    if not os.path.exists(meta_path):
        # Fallback defaults if metadata missing
        return {
            'features': [
                'Area', 'Bedrooms', 'Bathrooms', 'Floors', 'YearBuilt',
                'Location', 'Condition', 'Garage',
                'HouseAge', 'TotalRooms', 'AreaPerRoom', 'BathBedRatio',
                'IsNew', 'ConditionScore'
            ],
            'locations': ['Downtown', 'Suburban', 'Urban', 'Rural'],
            'conditions': ['Poor', 'Fair', 'Good', 'Excellent'],
            'garage_options': ['Yes', 'No'],
            'model_metrics': {'RMSE': 0, 'MAE': 0, 'R2': 0}
        }
    with open(meta_path, 'r') as f:
        return json.load(f)


# Load resources
model = load_model()
metadata = load_metadata()


# ============================================================
# HEADER
# ============================================================
st.markdown('<div class="big-title">🏠 House Price Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Enter property details below to get an instant price estimate</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR — model info
# ============================================================
with st.sidebar:
    st.header("📊 Model Information")

    st.markdown("**Algorithm:** Random Forest Regressor")
    st.markdown("**Training rows:** 1,600")
    st.markdown("**Features used:** 14")

    st.markdown("---")

    st.markdown("### 📈 Performance Metrics")
    metrics = metadata.get('model_metrics', {})

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("R² Score", f"{metrics.get('R2', 0):.4f}")
    with col_b:
        st.metric("RMSE", f"₹{metrics.get('RMSE', 0):,.0f}")

    st.metric("MAE", f"₹{metrics.get('MAE', 0):,.0f}")

    st.markdown("---")

    st.info(
        "ℹ️ **Note:** This model was trained on a synthetic dataset "
        "with weak feature-target relationships. On real-world data "
        "(e.g., Ames Housing), accuracy would be significantly higher."
    )

    st.markdown("---")
    st.caption("Built with Streamlit • scikit-learn")


# ============================================================
# MAIN FORM — user inputs
# ============================================================
st.markdown("### 🏡 Property Details")

# Two-column layout for inputs
col1, col2 = st.columns(2)

with col1:
    area = st.number_input(
        "Area (sq ft)",
        min_value=500,
        max_value=5000,
        value=2500,
        step=50,
        help="Total built-up area in square feet"
    )

    bedrooms = st.slider(
        "Bedrooms",
        min_value=1,
        max_value=5,
        value=3,
        help="Number of bedrooms"
    )

    bathrooms = st.slider(
        "Bathrooms",
        min_value=1,
        max_value=4,
        value=2,
        help="Number of bathrooms"
    )

    floors = st.slider(
        "Floors",
        min_value=1,
        max_value=3,
        value=2,
        help="Number of floors"
    )

with col2:
    year_built = st.number_input(
        "Year Built",
        min_value=1900,
        max_value=2024,
        value=2010,
        step=1,
        help="Year the property was constructed"
    )

    location = st.selectbox(
        "Location",
        options=metadata.get('locations', ['Downtown', 'Suburban', 'Urban', 'Rural']),
        help="Neighborhood type"
    )

    condition = st.selectbox(
        "Condition",
        options=metadata.get('conditions', ['Poor', 'Fair', 'Good', 'Excellent']),
        help="Overall property condition"
    )

    garage = st.selectbox(
        "Garage",
        options=metadata.get('garage_options', ['Yes', 'No']),
        help="Does the property have a garage?"
    )


# ============================================================
# PREDICTION
# ============================================================
st.markdown("---")

col_btn_1, col_btn_2, col_btn_3 = st.columns([1, 1, 1])
with col_btn_2:
    predict_clicked = st.button("🔮 Predict Price", use_container_width=True, type="primary")


if predict_clicked:
    # Build the input DataFrame with all 14 features (raw + engineered)
    input_df = pd.DataFrame([{
        # Raw features
        'Area': area,
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'Floors': floors,
        'YearBuilt': year_built,
        'Location': location,
        'Condition': condition,
        'Garage': garage,
        # Engineered features (must match Cell 9 exactly)
        'HouseAge': 2024 - year_built,
        'TotalRooms': bedrooms + bathrooms,
        'AreaPerRoom': area / (bedrooms + bathrooms) if (bedrooms + bathrooms) > 0 else 0,
        'BathBedRatio': bathrooms / bedrooms if bedrooms > 0 else 0,
        'IsNew': int(year_built >= 2010),
        'ConditionScore': {'Poor': 0, 'Fair': 1, 'Good': 2, 'Excellent': 3}[condition]
    }])

    # Reorder columns to match training exactly
    expected_features = metadata.get('features', input_df.columns.tolist())
    input_df = input_df[expected_features]

    # Predict
    try:
        prediction = model.predict(input_df)[0]

        # Display the prediction prominently
        st.markdown(f"""
            <div class="prediction-box">
                <div style="font-size: 1.1rem; opacity: 0.9;">Estimated Price</div>
                <div class="prediction-price">₹{prediction:,.0f}</div>
                <div style="font-size: 0.9rem; opacity: 0.8;">
                    Based on the features you entered
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Confidence interval (approximate — based on test RMSE)
        rmse = metrics.get('RMSE', 0)
        if rmse > 0:
            lower = prediction - 1.96 * rmse
            upper = prediction + 1.96 * rmse
            st.info(
                f"📊 **Approximate 95% confidence interval:** "
                f"₹{lower:,.0f} – ₹{upper:,.0f}"
            )

        # Show input summary
        with st.expander("🔍 View input details & engineered features"):
            st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.exception(e)


# ============================================================
# FOOTER
# ============================================================
st.markdown("""
    <div class="footer">
        Built as a demonstration project — House Price Prediction using Random Forest.<br>
        ⚠️ Predictions are based on a synthetic dataset and should not be used for real decisions.
    </div>
""", unsafe_allow_html=True)