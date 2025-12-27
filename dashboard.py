import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import glob
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="RailLife Predictive Maintenance",
    page_icon="🚄",
    layout="wide"
)

# --- Helper Functions ---
@st.cache_data
def load_data():
    """Loads the MetroPT-3 dataset from parquet files."""
    files = sorted(glob.glob("content/parquet_cache/*.parquet"))
    if not files:
        return pd.DataFrame()
    
    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            dfs.append(df)
        except Exception as e:
            st.warning(f"Could not load {f}: {e}")
            
    if not dfs:
        return pd.DataFrame()
        
    full_df = pd.concat(dfs, ignore_index=True)
    if "timestamp" in full_df.columns:
        full_df["timestamp"] = pd.to_datetime(full_df["timestamp"], utc=True)
        full_df = full_df.sort_values("timestamp")
        
    return full_df

@st.cache_data
def calculate_rul(df, failure_events):
    """Calculates Ground Truth RUL based on failure events."""
    if df.empty:
        return df
    df_rul = df.copy()
    events = sorted(failure_events, key=lambda x: x["start"])
    df_rul["RUL_hours"] = 0.0
    for e in reversed(events):
        mask = df_rul["timestamp"] < e["start"]
        time_diff = e["start"] - df_rul.loc[mask, "timestamp"]
        df_rul.loc[mask, "RUL_hours"] = time_diff.dt.total_seconds() / 3600.0
    return df_rul

# --- Sidebar ---
st.sidebar.title("RailLife Dashboard")
# Team details REMOVED as requested
page = st.sidebar.radio("Navigation", ["Final Results", "Data Explorer", "Visualizations", "Ground Truth RUL"])

# --- Constants ---
FAILURE_EVENTS = [
    {"id": 1, "start": pd.to_datetime("2020-04-18 00:00", utc=True), "end": pd.to_datetime("2020-04-18 23:59", utc=True), "fault": "Air leak"},
    {"id": 2, "start": pd.to_datetime("2020-05-29 23:30", utc=True), "end": pd.to_datetime("2020-05-30 06:00", utc=True), "fault": "Air leak"},
    {"id": 3, "start": pd.to_datetime("2020-06-05 10:00", utc=True), "end": pd.to_datetime("2020-06-07 14:30", utc=True), "fault": "Air leak"},
    {"id": 4, "start": pd.to_datetime("2020-07-15 14:30", utc=True), "end": pd.to_datetime("2020-07-15 19:00", utc=True), "fault": "Air leak"},
]

# --- Main App ---

if page == "Final Results":
    # Layout matches the provided image
    
    # 1. Top Row: Model Comparison & Highlights
    col_top_left, col_top_right = st.columns([2, 1])
    
    with col_top_left:
        st.subheader("Final Model Comparison (MAE - Lower is Better)")
        model_data = {
            "Model": ["LSTM (Sequences)", "GBR (Events)", "Dynamic Ensemble", "Cox Regression", "Baseline (Mean)"],
            "MAE": [139.4, 171.4, 216.0, 292.7, 576.6],
            "Color": ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c", "#95a5a6"]
        }
        fig_models = px.bar(model_data, x="MAE", y="Model", orientation='h', text="MAE",
                            color="Model", color_discrete_sequence=model_data["Color"])
        fig_models.update_traces(texttemplate='%{text}h', textposition='outside')
        fig_models.update_layout(showlegend=False, height=300, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_models, use_container_width=True)

    with col_top_right:
        st.write("") # Spacer
        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border: 1px solid #dcdcdc;">
            <h4>📋 PROJECT HIGHLIGHTS</h4>
            <br>
            <b>1. BEST MODEL: LSTM</b>
            <ul>
                <li>MAE: 139.4 hours</li>
                <li>75.8% Improvement over baseline</li>
                <li>Captures temporal degradation patterns</li>
            </ul>
            <b>2. RUNNER UP: Gradient Boosting</b>
            <ul>
                <li>MAE: 171.4 hours</li>
                <li>Robust on event-based data</li>
            </ul>
            <b>3. DATA INSIGHTS</b>
            <ul>
                <li>76% of data is censored (handled)</li>
                <li>No data leakage (windowed features)</li>
                <li>RUL Range: 0 - 1848 hours</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # 2. Middle Row: RUL Prediction Timeline
    st.subheader("RUL Prediction Timeline (Gradient Boosting)")
    if os.path.exists("GBR.png"):
        st.image("GBR.png", use_column_width=True)
    else:
        st.info("GBR.png not found. This chart would show GBR Predictions vs Actual RUL.")

    # 3. Bottom Row: Error Distribution & Features
    col_bot_left, col_bot_right = st.columns(2)
    
    with col_bot_left:
        st.subheader("Error Distribution Comparison")
        # Placeholder or static image if error distribution is not available
        # It's not in the file list, so we might skip or put a placeholder
        st.info("Error Distribution Data not available for interactive plotting.")
        
    with col_bot_right:
        st.subheader("Top 10 Predictive Features (GBR)")
        feature_data = {
            "Feature": ["MPG_ewm_24", "COMP_ewm_24", "temp_trend_720h", "DV_pressure_roll_min_24", 
                        "Caudal_impulses", "H1_ewm_24", "LPS_ewm_24", "temp_max_recent_24h", 
                        "DV_pressure_roll_max_24", "motor_volatility_48h"],
            "Importance": [0.391, 0.146, 0.082, 0.066, 0.057, 0.024, 0.021, 0.018, 0.015, 0.014]
        }
        # Sort for display
        feature_df = pd.DataFrame(feature_data).sort_values("Importance", ascending=True)
        
        fig_feat = px.bar(feature_df, x="Importance", y="Feature", orientation='h', 
                          color="Importance", color_continuous_scale="Viridis")
        fig_feat.update_layout(showlegend=False, height=350, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_feat, use_container_width=True)

elif page == "Data Explorer":
    st.title("Data Explorer 📊")
    with st.spinner("Loading MetroPT-3 Dataset..."):
        df = load_data()
    if not df.empty:
        st.success(f"Successfully loaded {len(df):,} rows.")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Samples", len(df))
        col2.metric("Sensors", len(df.columns) - 2) 
        col3.metric("Date Range", f"{df['timestamp'].dt.date.min()} to {df['timestamp'].dt.date.max()}")
        st.dataframe(df.head(10))
        with st.expander("See Data Statistics"):
            st.write(df.describe())

elif page == "Visualizations":
    st.title("Sensor Visualizations 📈")
    with st.spinner("Loading data..."):
        df = load_data()
    if not df.empty:
        downsample_rate = st.sidebar.slider("Downsample Rate", 1, 1000, 100)
        sample_df = df.iloc[::downsample_rate, :].copy()
        st.write(f"Plotting 1 out of every {downsample_rate} points")
        sensor_cols = [c for c in df.columns if c not in ["timestamp", "Unnamed: 0", "RUL_hours"]]
        selected_sensors = st.multiselect("Select Sensors", sensor_cols, default=["Motor_current", "Oil_temperature"])
        if selected_sensors:
            fig = px.line(sample_df, x="timestamp", y=selected_sensors, title="Sensor Readings Over Time")
            for i, event in enumerate(FAILURE_EVENTS):
                fig.add_vrect(x0=event["start"], x1=event["end"], fillcolor="red", opacity=0.2, layer="below")
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")
        st.subheader("Correlation Analysis")
        if os.path.exists("corrheatmap.png"):
            st.image("corrheatmap.png", caption="Sensor Correlations", width=600)

elif page == "Ground Truth RUL":
    st.title("Ground Truth RUL Analysis ⏳")
    with st.spinner("Processing RUL..."):
        df = load_data()
        if not df.empty:
            df_rul = calculate_rul(df, FAILURE_EVENTS)
            sample_rul = df_rul.iloc[::100, :].copy()
            fig = px.line(sample_rul, x="timestamp", y="RUL_hours", title="Ground Truth RUL", 
                          labels={"RUL_hours": "Hours until Next Failure"})
            fig.update_traces(line_color="green")
            for event in FAILURE_EVENTS:
                fig.add_vline(x=event["start"], line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
