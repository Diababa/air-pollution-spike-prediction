import streamlit as st
import pandas as pd
import plotly.express as px
import requests

API_URL = "http://api:8000/predict-spike"

st.set_page_config(
    page_title="Air Pollution Spike Prediction",
    page_icon="🌍",
    layout="wide"
)


@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/train.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


@st.cache_data
def load_shap_data():
    return pd.read_csv("data/processed/shap_feature_importance.csv")


df = load_data()

st.title("🌍 Air Pollution Spike Prediction")
st.markdown(
    """
    This dashboard predicts whether a **PM2.5 pollution spike** may occur within the next **48 hours**
    using historical pollution levels, weather conditions, and time-based features.
    """
)

st.divider()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Dataset size", f"{len(df):,} rows")
col2.metric("Average PM2.5", f"{df['pm25'].mean():.1f}")
col3.metric("Max PM2.5", f"{df['pm25'].max():.1f}")
col4.metric("Spike rate", f"{df['spike_48h'].mean() * 100:.1f}%")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Data Overview",
    "🚨 Spike Analysis",
    "🔮 Live Prediction",
    "🧠 Explainability"
])

with tab1:
    st.subheader("PM2.5 evolution over time")

    fig = px.line(
        df,
        x="datetime",
        y="pm25",
        title="Hourly PM2.5 concentration"
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown(
        """
        PM2.5 is one of the most harmful air pollutants.
        The goal is not only to analyze current pollution levels, but also to anticipate future high-risk events.
        """
    )

with tab2:
    st.subheader("Historical pollution spikes")

    spikes = df[df["spike_48h"] == 1]
    threshold = df["pm25"].quantile(0.90)

    fig = px.scatter(
        df,
        x="datetime",
        y="pm25",
        color="spike_48h",
        title="Detected pollution spike periods",
        labels={"spike_48h": "Spike in 48h"}
    )
    st.plotly_chart(fig, width="stretch")

    col_a, col_b = st.columns(2)

    with col_a:
        st.metric("Number of spike situations", f"{len(spikes):,}")

    with col_b:
        st.metric("Spike threshold", f"{threshold:.1f} PM2.5")

    st.info(
        "A spike is defined as a future PM2.5 value above the 90th percentile of historical observations."
    )

with tab3:
    st.subheader("Predict pollution risk for the next 48 hours")

    st.markdown(
        """
        Select a historical observation below.
        The model will use pollution history, weather conditions and temporal features
        to estimate the probability of a pollution spike in the next 48 hours.
        """
    )

    selected_index = st.slider(
        "Select an observation",
        min_value=0,
        max_value=len(df) - 1,
        value=len(df) - 1
    )

    sample = df.iloc[selected_index]

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("### Selected context")
        st.write(f"**Datetime:** {sample['datetime']}")
        st.write(f"**Current PM2.5:** {sample['pm25']:.2f}")
        st.write(f"**Temperature:** {sample['temperature']:.2f} °C")
        st.write(f"**Humidity:** {sample['humidity']:.2f}%")
        st.write(f"**Wind speed:** {sample['wind_speed']:.2f}")
        st.write(f"**Wind direction:** {sample['wind_direction']:.2f}")

    payload = {}

    for col in df.columns:
        if col not in ["datetime", "target_pm25_48h", "spike_48h"]:
            payload[col] = float(sample[col])

    if st.button("🔮 Predict 48h spike risk", width="stretch"):
        try:
            response = requests.post(API_URL, json=payload)
            result = response.json()

            probability = result["spike_probability"]
            prediction = result["spike_prediction"]

            with col_right:
                st.markdown("### Prediction result")

                st.metric(
                    label="Spike probability",
                    value=f"{probability * 100:.1f}%"
                )

                st.progress(min(probability, 1.0))

                if prediction == 1:
                    st.error("🚨 High pollution spike risk detected")
                    st.markdown(
                        """
                        The model predicts a likely pollution spike within the next 48 hours.
                        This situation may require preventive monitoring or alerting.
                        """
                    )
                else:
                    st.success("✅ Low pollution spike risk")
                    st.markdown(
                        """
                        The model does not detect a high-risk pollution spike within the next 48 hours.
                        """
                    )

        except Exception as e:
            st.error("API connection failed. Make sure FastAPI is running.")
            st.write(e)

with tab4:
    st.subheader("🧠 Model Explainability")

    st.markdown(
        """
        This section explains which variables have the strongest influence on the pollution spike prediction model.
        The values are computed using SHAP, a standard explainability method for machine learning models.
        """
    )

    shap_df = load_shap_data()

    fig3 = px.bar(
        shap_df.head(15),
        x="importance",
        y="feature",
        orientation="h",
        title="Top 15 Features Influencing Spike Predictions",
        text="importance"
    )

    fig3.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside"
    )

    fig3.update_layout(
        yaxis=dict(autorange="reversed"),
        height=600
    )

    st.plotly_chart(fig3, width="stretch")

    st.success(
        """
        Main interpretation:
        seasonal variables, weather conditions and recent PM2.5 history are the strongest drivers of spike predictions.
        """
    )

st.divider()

st.markdown(
    """
    ### Project pipeline

    **Data ingestion → validation → feature engineering → XGBoost modeling → FastAPI serving → Streamlit dashboard**

    This project demonstrates an end-to-end machine learning system for environmental risk prediction.
    """
)