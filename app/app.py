import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from scr.inference import make_prediction
import pandas as pd
import numpy as np
import joblib




# Page Configuration
st.set_page_config(
    page_title="Airline Price Prediction",
    layout="centered"
)

st.title("Airline Ticket Price Prediction")
st.markdown(
    "Predict  flight  ticket  prices  using  a  trained  XGBoost Regression  Model!"
)

st.divider()

# Load Model & Metadata
@st.cache_resource
def load_artifacts():
    
    try:
        model = joblib.load(os.path.join("Models", "XGBoost.pkl"))
        input_cols = joblib.load(os.path.join("Metadata", "input_columns.pkl"))
        unique_vals = joblib.load(os.path.join("Metadata", "unique_values.pkl"))
        return model, input_cols, unique_vals

    except FileNotFoundError as e:
        st.error(f"File not found: {e.filename}")
        return None, None, None

    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        return None, None, None


model, input_columns, unique_values = load_artifacts()

# Stop the app if loading fails
if model is None:
    st.stop()

# Sidebar Inputs
st.sidebar.header("Please Enter Flight Details! ")

airline = st.sidebar.selectbox(
    "Airline",
    sorted(unique_values["Airline"])
)

source = st.sidebar.selectbox(
    "Source",
    sorted(unique_values["Source"])
)

destination = st.sidebar.selectbox(
    "Destination",
    sorted(unique_values["Destination"])
)

journey_day = st.sidebar.selectbox(
    "Journey Day",
    sorted(unique_values["Journey_Day"])
)

journey_month = st.sidebar.selectbox(
    "Journey Month",
    sorted(unique_values["Journey_Month"])
)

total_stops = st.sidebar.selectbox(
    "Total Stops",
    unique_values["Total_Stops"]
)

duration_hours = st.sidebar.slider(
    "Flight Duration (hours)",
    min_value=0.5,
    max_value=30.0,
    value=5.0,
    step=0.5
)

duration = int(duration_hours * 60) 


dep_hour = st.sidebar.slider(
    "Departure Hour",
    min_value=0,
    max_value=23,
    value=10
)
# Build Input DataFrame
input_data = pd.DataFrame([{
    "Airline": airline,
    "Source": source,
    "Destination": destination,
    "Journey_Day": journey_day,
    "Journey_Month": journey_month,
    "Total_Stops": total_stops,
    "Duration": duration,
    "Dep_Hour": dep_hour
}])

# Feature Engineering
if "Is_Weekend" in input_columns:
    input_data["Is_Weekend"] = input_data["Journey_Day"].isin([5, 6]).astype(int)

# Ensure correct column order
input_data = input_data[input_columns]


# Prediction
if st.button("Predict Ticket Price", use_container_width=False):

    with st.spinner("Predicting price..."):

        # Prepare dictionary of input values
        values = {
            "Airline": airline,
            "Source": source,
            "Destination": destination,
            "Journey_Day": journey_day,
            "Journey_Month": journey_month,
            "Total_Stops": total_stops,
            "Duration": duration, 
            "Dep_Hour": dep_hour
        }

        # Call inference function
        log_price_pred = make_prediction(values)
        price_pred = np.expm1(log_price_pred)

    st.success("Prediction Complete")

    st.metric(
        label="Estimated Ticket Price",
        value=f"₹ {price_pred:,.0f}"
    )

# Footer
st.divider()
st.caption("Rights Reserved")
