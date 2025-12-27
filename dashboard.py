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
    # Use absolute path or relative to current directory
    # Assuming running from repo root
    files = sorted(glob.glob("content/parquet_cache/*.parquet"))
    if not files:
        # Fallback to check absolute path if necessary or warn
        st.error("No parquet files found in 'content/parquet_cache/'. Please check the data directory.")
        return pd.DataFrame()
    
    dfs = []
    # Limit to first few files if too large? 1.5M rows is manageable for Streamlit but might be slow.
    # We'll load all since there are only 4 parts.
    for f in files:
        try:
            df = pd.read_parquet(f)
            dfs.append(df)
        except Exception as e:
            st.warning(f"Could not load {f}: {e}")
            
    if not dfs:
        return pd.DataFrame()
        
    full_df = pd.concat(dfs, ignore_index=True)
    
    # Ensure timestamp is datetime
    if "timestamp" in full_df.columns:
        full_df["timestamp"] = pd.to_datetime(full_df["timestamp"], utc=True)
        full_df = full_df.sort_values("timestamp")
        
    return full_df

@st.cache_data
def calculate_rul(df, failure_events):
    """Calculates Ground Truth RUL based on failure events."""
    if df.empty:
        return df
        
    # Create a copy to avoid modifying the cached dataframe
    df_rul = df.copy()
    
    # Sort events by time
    events = sorted(failure_events, key=lambda x: x["start"])
    
    # Initialize RUL with 0 (or large number)
    df_rul["RUL_hours"] = 0.0
    
    # We iterate events in REVERSE order to overwrite earlier RULs correctly
    # Strategy: For a timestamp T, RUL is (Next_Failure_Start - T).
    # If we go randomly, we might set RUL diff against a far future event.
    # By going Reverse, we set RUL against Event 4 for all T < Event 4.
    # Then we set RUL against Event 3 for all T < Event 3 (overwriting the Event 4 calc for those points).
    # This leaves the points between Event 3 and Event 4 having the Event 4 calc. Perfect.
    
    for e in reversed(events):
        mask = df_rul["timestamp"] < e["start"]
        # Calculate hours difference
        time_diff = e["start"] - df_rul.loc[mask, "timestamp"]
        df_rul.loc[mask, "RUL_hours"] = time_diff.dt.total_seconds() / 3600.0
        
    # Points after the last failure currently have 0.0 (initialized) or whatever the loop did?
    # The loop condition `timestamp < e['start']` means points > last event start don't get touched in the first iteration.
    # So they remain 0.0. Correct, assuming RUL is 0 or undefined after failure during maintenance.
    
    return df_rul

# --- Sidebar ---
st.sidebar.title("RailLife Dashboard")
page = st.sidebar.radio("Navigation", ["Overview", "Data Explorer", "Visualizations", "RUL Analysis"])

st.sidebar.info(
    """
    **Team 13**
    - James Cook
    - Datta Sai VVN
    - Heba Syed
    - Faaizah Ismail
    - Lasya Sahithi
    """
)

# --- Constants ---
# Failure events from README / Notebook
FAILURE_EVENTS = [
    {"id": 1, "start": pd.to_datetime("2020-04-18 00:00", utc=True), "end": pd.to_datetime("2020-04-18 23:59", utc=True), "fault": "Air leak"},
    {"id": 2, "start": pd.to_datetime("2020-05-29 23:30", utc=True), "end": pd.to_datetime("2020-05-30 06:00", utc=True), "fault": "Air leak"},
    {"id": 3, "start": pd.to_datetime("2020-06-05 10:00", utc=True), "end": pd.to_datetime("2020-06-07 14:30", utc=True), "fault": "Air leak"},
    {"id": 4, "start": pd.to_datetime("2020-07-15 14:30", utc=True), "end": pd.to_datetime("2020-07-15 19:00", utc=True), "fault": "Air leak"},
]

# --- Main App ---

if page == "Overview":
    st.title("RailLife: Predictive Maintenance System")
    st.markdown("### CS 418 Final Project - Team 13")
    
    st.header("Project Overview")
    st.write("""
    RailLife is an advanced machine learning system for predicting the **Remaining Useful Life (RUL)** of train air production units. 
    Traditional railway maintenance relies on fixed schedules, leading to inefficiencies and unexpected breakdowns. 
    This system utilizes sensor data from the MetroPT-3 dataset to enable **condition-based maintenance**.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Problem Statement")
        st.info("""
        - **High Cost**: Unexpected breakdowns cost >$50k per incident.
        - **Inefficiency**: Fixed-schedule maintenance wastes resources.
        - **Safety Risk**: Failures can occur between scheduled checks.
        """)
    with col2:
        st.subheader("Goal")
        st.success("""
        - Develop a predictive model.
        - Target Accuracy: **< 50 hours of error** in RUL prediction.
        - Enable proactive maintenance scheduling.
        """)
    
    st.subheader("Known Failure Events")
    events_df = pd.DataFrame(FAILURE_EVENTS)
    # Format for display
    display_df = events_df.copy()
    display_df["start"] = display_df["start"].dt.strftime("%Y-%m-%d %H:%M")
    display_df["end"] = display_df["end"].dt.strftime("%Y-%m-%d %H:%M")
    st.table(display_df)
    
    st.markdown("---")
    st.subheader("Model Architecture (LSTM)")
    if os.path.exists("LSTM.png"):
        st.image("LSTM.png", caption="LSTM Network Structure", width=400)
    else:
        st.warning("LSTM.png not found in directory.")

elif page == "Data Explorer":
    st.title("Data Explorer 📊")
    
    with st.spinner("Loading MetroPT-3 Dataset... (this may take a moment)"):
        df = load_data()
    
    if not df.empty:
        st.success(f"Successfully loaded {len(df):,} rows.")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Samples", len(df))
        col2.metric("Sensors", len(df.columns) - 2) # approx
        col3.metric("Date Range", f"{df['timestamp'].dt.date.min()} to {df['timestamp'].dt.date.max()}")
        
        st.subheader("Raw Data Preview")
        st.dataframe(df.head(10))
        
        with st.expander("See Data Statistics"):
            st.write(df.describe())
            
        with st.expander("See Column Types"):
            st.write(df.dtypes.astype(str))
            
    else:
        st.error("Failed to load data.")

elif page == "Visualizations":
    st.title("Sensor Visualizations 📈")
    
    with st.spinner("Loading data..."):
        df = load_data()
        
    if not df.empty:
        # Downsample for faster plotting
        downsample_rate = st.sidebar.slider("Downsample Rate (for performance)", 1, 1000, 100)
        sample_df = df.iloc[::downsample_rate, :].copy()
        
        st.write(f"Plotting 1 out of every {downsample_rate} points ({len(sample_df)} points shown)")
        
        # Sensor Selection
        # Filter out non-sensor cols like timestamp, Unnamed
        sensor_cols = [c for c in df.columns if c not in ["timestamp", "Unnamed: 0", "RUL_hours"]]
        selected_sensors = st.multiselect("Select Sensors", sensor_cols, default=["Motor_current", "Oil_temperature"])
        
        if selected_sensors:
            fig = px.line(sample_df, x="timestamp", y=selected_sensors, title="Sensor Readings Over Time")
            
            # Overlay failure events
            for i, event in enumerate(FAILURE_EVENTS):
                fig.add_vrect(
                    x0=event["start"], x1=event["end"],
                    fillcolor="red", opacity=0.2,
                    layer="below", line_width=0,
                    annotation_text=f"Fail {event['id']}", annotation_position="top left"
                )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select at least one sensor to visualize.")
            
        st.markdown("---")
        st.subheader("Correlation Analysis")
        
        if os.path.exists("corrheatmap.png"):
            st.image("corrheatmap.png", caption="Sensor Correlations", width=600)
        else:
            st.info("Static correlation map not found. Computing live correlation on sample...")
            if not sample_df.empty:
                corr = sample_df[sensor_cols].corr()
                fig_corr = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
                st.plotly_chart(fig_corr)

elif page == "RUL Analysis":
    st.title("Remaining Useful Life (RUL) Analysis ⏳")
    
    st.markdown("""
    This section visualizes the **Ground Truth RUL**. 
    Since the trained LSTM model file is missing from the repository, we calculate the theoretical RUL based on the known failure events.
    """)
    
    with st.spinner("Processing RUL..."):
        df = load_data()
        if not df.empty:
            df_rul = calculate_rul(df, FAILURE_EVENTS)
            
            downsample = 100
            st.info(f"Visualizing RUL (downsampled by {downsample}x)")
            
            sample_rul = df_rul.iloc[::downsample, :].copy()
            
            fig = px.line(sample_rul, x="timestamp", y="RUL_hours", title="Ground Truth RUL", 
                          labels={"RUL_hours": "Hours until Next Failure"})
            
            fig.update_traces(line_color="green")
            
            # Add failure markers
            for event in FAILURE_EVENTS:
                fig.add_vline(x=event["start"], line_dash="dash", line_color="red")
                fig.add_annotation(x=event["start"], y=2000, text=f"Failure {event['id']}", showarrow=False, ax=10)
                
            st.plotly_chart(fig, use_container_width=True)
            
            st.warning("**Note**: This is the TARGET variable the model aims to predict. Real-time inference would show a predicted line overlaying this ground truth.")
        else:
            st.error("No data to analyze.")
