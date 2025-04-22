import streamlit as st
import pickle
import numpy as np
import time
import joblib
from sklearn.preprocessing import StandardScaler

# ──────────────────────────────────────────────────────────────
# Page Config - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Space Weather Forecasting System", 
    layout="wide",
    page_icon="🌌"
)
# ──────────────────────────────────────────────────────────────
# Custom CSS for styling
st.markdown("""
<style>
    /* Main background and text */
    .main {
        background-color: white !important;
    }
    body {
        color: black !important;
    }
    
    /* Headers and text elements */
    h1, h2, h3, h4, h5, h6, p, .stMarkdown, .stText {
        color: black !important;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(45deg, #6e48aa, #9d50bb);
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background: linear-gradient(45deg, #9d50bb, #6e48aa);
        color: white !important;
    }
    
    /* Input fields */
    .stTextInput>div>div>input, 
    .stNumberInput>div>div>input,
    .stTextArea>div>div>textarea,
    .stSelectbox>div>div>select {
        background-color: white !important;
        color: black !important;
        border: 1px solid #ccc;
    }
    
    /* Sliders */
    .stSlider>div>div>div>div {
        background: linear-gradient(45deg, #6e48aa, #9d50bb) !important;
    }
    .stSlider>div>div>div>div>div {
        background: white !important;
        border: 2px solid #6e48aa !important;
    }
    
    /* Alerts and info boxes */
    .stAlert {
        background-color: #f8f9fa !important;
        color: black !important;
        border-radius: 10px;
        border: 1px solid #dee2e6;
    }
    
    /* Tables */
    .stTable {
        color: black !important;
    }
    table {
        color: black !important;
    }
    
    /* Tabs */
    .stTabs>div>div>div>div {
        color: black !important;
    }
    .stTabs>div>div>div>div[aria-selected="true"] {
        color: #6e48aa !important;
        font-weight: bold;
    }
    
    /* Progress bars */
    .stProgress>div>div>div>div {
        background: linear-gradient(45deg, #6e48aa, #9d50bb) !important;
    }
    
    /* Remove all dark background classes */
    [class*="st-"] {
        background-color: transparent !important;
        color: black !important;
    }
</style>
""", unsafe_allow_html=True)
# ──────────────────────────────────────────────────────────────
# Load Models - after page config
with st.spinner('Loading storm model...'):
    storm_model = joblib.load("storm_pipeline.pkl")

with st.spinner('Loading solar wind model...'):
    with open("solar_wind_model.pkl", "rb") as f:
        solar_wind_model = pickle.load(f)

with st.spinner('Loading flare model...'):
    with open("solar_flare_rf_model.pkl", "rb") as f:
        flare_model = pickle.load(f)

# Sidebar
with st.sidebar:
    st.title("🌌 Space Weather")
    st.markdown("""
    ### About
    This dashboard predicts:
    - Geomagnetic storms
    - Solar wind conditions
    - Solar flare classes
    
    Using state-of-the-art machine learning models.
    """)
    
    st.markdown("---")
    st.markdown("""
    ### Quick Guide
    1. Select a prediction type
    2. Enter the required parameters
    3. Click predict button
    """)
    
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit")

# Main Content
st.title("🌌 Space Weather Forecasting System")
st.markdown("""
**Predict geomagnetic storms, solar wind conditions, and solar flare classes**  
using our trained machine learning models.
""")

# Tabs for different predictions
tab1, tab2, tab3 = st.tabs(["🌪️ Geomagnetic Storm", "🌬️ Solar Wind", "☀️ Solar Flare"])

with tab1:
    st.header("🌪️ Geomagnetic Storm Classification")
    st.markdown("Predict whether a geomagnetic storm will occur based on solar wind parameters.")
    
    with st.expander("ℹ️ About this model", expanded=False):
        st.markdown("""
        This model predicts geomagnetic storms using:
        - IMF Magnitude
        - Bz GSM component
        - Solar wind density
        - Solar wind speed
        - Solar wind pressure
        """)
    
    cols = st.columns(2)
    with cols[0]:
        st.subheader("Solar Wind Features")
        imf_mag = st.slider("IMF Magnitude (nT)", min_value=0.0, max_value=20.0, value=3.0, step=0.1)
        bz_gsm = st.slider("Bz GSM (nT)", min_value=-20.0, max_value=20.0, value=-0.5, step=0.1)
        density = st.slider("Density (protons/cm³)", min_value=0.1, max_value=20.0, value=2.5, step=0.1)
    
    with cols[1]:
        st.subheader("")
        st.subheader("")  # Spacer
        pressure = st.slider("Pressure (nPa)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        speed = st.slider("Speed (km/s)", min_value=100.0, max_value=1000.0, value=280.0, step=10.0)
    
    if st.button("⚡ Predict Storm", key="storm_btn"):
        with st.spinner('Analyzing solar wind data...'):
            time.sleep(1)  # Simulate processing time
            features = np.array([[imf_mag, bz_gsm, density, speed, pressure]])
            result = storm_model.predict(features)[0]
            
            if result == 1:
                st.error("""
                ## 🌩️ Storm Alert!
                **Geomagnetic storm detected.**  
                Potential impacts:
                - Satellite disruptions
                - Power grid fluctuations
                - Aurora visibility at lower latitudes
                """)
                st.image("https://cdn.pixabay.com/photo/2017/08/30/01/05/aurora-2695569_960_720.jpg", width=500)
            else:
                st.success("""
                ## ☀️ Quiet Conditions
                **No significant storm expected.**  
                Space weather conditions are nominal.
                """)
                st.image("https://cdn.pixabay.com/photo/2016/11/29/08/41/apple-1868496_960_720.jpg", width=500)

with tab2:
    st.header("🌬️ Solar Wind Forecasting")
    st.markdown("Predict future solar wind conditions based on current measurements.")
    
    with st.expander("ℹ️ About this model", expanded=False):
        st.markdown("""
        This model forecasts:
        - Solar wind speed (km/s)
        - Bz GSM component (nT)
        
        for the next 6 time steps based on current conditions.
        """)
    
    cols = st.columns(2)
    with cols[0]:
        st.subheader("Current Conditions")
        imf_mag_sw = st.slider("IMF Magnitude (nT)", min_value=0.0, max_value=20.0, value=3.0, key="sw1")
        density_sw = st.slider("Density (protons/cm³)", min_value=0.1, max_value=20.0, value=2.5, key="sw2")
        pressure_sw = st.slider("Pressure (nPa)", min_value=0.1, max_value=10.0, value=1.0, key="sw3")
    
    with cols[1]:
        st.subheader("")
        st.subheader("")  # Spacer
        speed_sw = st.slider("Speed (km/s)", min_value=100.0, max_value=1000.0, value=280.0, key="sw4")
        bz_gsm_sw = st.slider("Bz GSM (nT)", min_value=-20.0, max_value=20.0, value=-0.5, key="sw5")
    
    if st.button("🚀 Forecast Solar Wind", key="wind_btn"):
        with st.spinner('Running solar wind simulation...'):
            time.sleep(1)  # Simulate processing time
            user_input = np.array([[imf_mag_sw, density_sw, pressure_sw, speed_sw, bz_gsm_sw]])
            user_input_sequence = np.repeat(user_input, 24, axis=0)
            reshaped_input = user_input_sequence.reshape(1, 24, 5)
            
            try:
                forecast = solar_wind_model.predict(reshaped_input)
                forecast_steps = forecast[0]
                
                st.success("## 📊 Forecast Complete!")
                
                # Visualize the forecast
                import plotly.graph_objects as go
                
                # Prepare data for plotting
                steps = [f"Step {i}" for i in range(1, 7)]
                vsw_values = [step[0] for step in forecast_steps]
                bz_values = [step[1] for step in forecast_steps]
                
                # Create figure
                fig = go.Figure()
                
                # Add Vsw trace
                fig.add_trace(go.Scatter(
                    x=steps,
                    y=vsw_values,
                    name="Solar Wind Speed (km/s)",
                    line=dict(color='#6e48aa', width=4)
                ))
                
                # Add Bz trace
                fig.add_trace(go.Scatter(
                    x=steps,
                    y=bz_values,
                    name="Bz GSM (nT)",
                    line=dict(color='#9d50bb', width=4),
                    yaxis="y2"
                ))
                
                # Update layout
                fig.update_layout(
                    title="6-Step Solar Wind Forecast",
                    xaxis_title="Forecast Steps",
                    yaxis_title="Solar Wind Speed (km/s)",
                    yaxis2=dict(
                        title="Bz GSM (nT)",
                        overlaying="y",
                        side="right"
                    ),
                    template="plotly_dark",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display table
                st.write("### Detailed Forecast Values")
                forecast_data = {
                    "Step": [f"Step {i}" for i in range(1, 7)],
                    "Solar Wind Speed (km/s)": [f"{step[0]:.2f}" for step in forecast_steps],
                    "Bz GSM (nT)": [f"{step[1]:.2f}" for step in forecast_steps]
                }
                st.table(forecast_data)
                
            except Exception as e:
                st.error("""
                ## ❌ Forecasting Failed
                There was an error processing your request.
                """)
                st.exception(e)

with tab3:
    st.header("☀️ Solar Flare Prediction")
    st.markdown("Predict the class of potential solar flares based on X-ray flux features.")
    
    with st.expander("ℹ️ About this model", expanded=False):
        st.markdown("""
        This model predicts solar flare classes:
        - **C-class**: Small flares with few noticeable consequences
        - **M-class**: Medium-sized flares that can cause brief radio blackouts
        - **X-class**: Large flares with potential for planet-wide effects
        
        Based on X-ray flux time series features.
        """)
    
    cols = st.columns(2)
    with cols[0]:
        st.subheader("X-ray Flux Features")
        rolling_mean_10 = st.slider("Rolling Mean (10)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        rolling_std_10 = st.slider("Rolling Std (10)", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
        lag_1 = st.slider("Lag 1", min_value=0.0, max_value=1.0, value=0.4, step=0.01)
    
    with cols[1]:
        st.subheader("")
        st.subheader("")  # Spacer
        lag_3 = st.slider("Lag 3", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
        lag_5 = st.slider("Lag 5", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
        delta_flux = st.slider("Delta Flux", min_value=0.0, max_value=0.5, value=0.05, step=0.01)
        delta_flux_3 = st.slider("Delta Flux (3)", min_value=0.0, max_value=0.5, value=0.07, step=0.01)
    
    if st.button("🔮 Predict Flare Class", key="flare_btn"):
        with st.spinner('Analyzing X-ray flux patterns...'):
            time.sleep(1)  # Simulate processing time
        try:
            # Create input array with the exact feature order used during training
            flare_input = np.array([[rolling_mean_10, rolling_std_10, lag_1, lag_3, lag_5, delta_flux, delta_flux_3]])
    
            # Ensure the input has the same shape as training data (n_samples, n_features)
            if flare_input.shape != (1, 7):
                flare_input = flare_input.reshape(1, -1)
    
            # Get prediction and probabilities
            prediction = flare_model.predict(flare_input)[0]  # This will give the class (0, 1, etc.)
            prediction_proba = flare_model.predict_proba(flare_input)[0]  # Probability for each class
    
            # Convert numerical prediction back to original label
            flare_class = le.inverse_transform([prediction])[0]
    
            # Get confidence (probability of predicted class)
            confidence = prediction_proba[prediction] * 100
    
            # Display results
            st.success(f"Predicted Flare Class: {flare_class}")
            st.write(f"Confidence: {confidence:.2f}%")
    
            # Show probabilities for all classes
            st.write("### Class Probabilities:")
            for class_idx, class_name in enumerate(le.classes_):
                st.write(f"- {class_name}: {prediction_proba[class_idx]*100:.2f}%")
    
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.write("Debug Info:")
            st.write("Input array:", flare_input)
            st.write("Input shape:", flare_input.shape)
            st.write("Model classes:", le.classes_ if hasattr(le, 'classes_') else "LabelEncoder not available")
            
            label_map = {0: "C-class", 1: "M-class", 2: "X-class"}
            flare_class = label_map[prediction] if prediction in label_map else prediction
            
            if flare_class == "X-class":
                st.error(f"""
                ## ☀️ X-class Flare Warning!
                **Potential impacts:**
                - Radio blackouts on sunlit side of Earth
                - Radiation storms
                - Possible satellite damage
                - GPS and communication disruptions
                """)
                st.image("https://cdn.pixabay.com/photo/2017/08/30/01/05/aurora-2695569_960_720.jpg", width=500)
            elif flare_class == "M-class":
                st.warning(f"""
                ## ☀️ M-class Flare Detected
                **Potential impacts:**
                - Brief radio blackouts
                - Minor radiation storms
                - Possible auroras at high latitudes
                """)
                st.image("https://cdn.pixabay.com/photo/2016/11/29/08/41/apple-1868496_960_720.jpg", width=500)
            else:
                st.success(f"""
                ## ☀️ C-class Flare Detected
                **Minimal impacts expected:**
                - No significant effects on Earth
                - Normal space weather conditions
                """)
                st.image("https://cdn.pixabay.com/photo/2016/11/29/08/41/apple-1868496_960_720.jpg", width=500)

