import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set page configuration
st.set_page_config(page_title="Energysage Dashboard", layout="wide")

# Add a title
st.markdown(
    """
    <div style="text-align: center;">
        <h1 style="color: #4CAF50;">Energysage - Predictive Maintenance Dashboard</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar navigation
st.sidebar.title("🔧 Dashboard Options")
page = st.sidebar.radio("Navigate", ["Upload Data", "Predictions", "Visualizations", "Settings"])

# Initialize session state for data persistence
if "df" not in st.session_state:
    st.session_state.df = None
if "failure_column" not in st.session_state:
    st.session_state.failure_column = None
if "timestamp_column" not in st.session_state:
    st.session_state.timestamp_column = None

# Upload Data Section
if page == "Upload Data":
    st.subheader("📂 Upload Sensor Data")
    uploaded_file = st.file_uploader("Upload Sensor CSV Data", type=["csv"])
    if uploaded_file:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.subheader("📊 Sensor Data Preview")
        st.write(st.session_state.df.head())

        # Dynamically map column names
        if "failure" in st.session_state.df.columns:
            st.session_state.failure_column = "failure"
        elif "failed" in st.session_state.df.columns:
            st.session_state.failure_column = "failed"

        if "timestamp" in st.session_state.df.columns:
            st.session_state.timestamp_column = "timestamp"

        if st.session_state.failure_column and st.session_state.timestamp_column:
            st.success("✅ Data uploaded successfully! Navigate to other sections for analysis.")
        else:
            st.warning("The dataset must contain both 'timestamp' and 'failure' or 'failed' columns.")

# Predictions Section
elif page == "Predictions":
    st.subheader("⚙️ Predictions")
    if st.session_state.df is not None and st.session_state.failure_column:
        # Prepare data
        X = st.session_state.df.select_dtypes(include=["number"]).drop(st.session_state.failure_column, axis=1)
        y = st.session_state.df[st.session_state.failure_column]

        # Split and train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predict
        predictions = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]

        st.subheader("⚠️ Predicted Failure Probabilities")
        predictions_df = pd.DataFrame({"Predicted Failure (1=Yes)": predictions, "Probability": prob})
        st.write(predictions_df.style.applymap(lambda x: "background-color: red" if x > 0.8 else "", subset=["Probability"]))
        st.caption("This table shows the predicted failure probabilities for each data point.")

        # Download predictions
        csv = predictions_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Predictions as CSV", data=csv, file_name="predictions.csv", mime="text/csv")

        st.success("✅ Predictions complete. You can now configure alerts or visualize risks.")

        # Configure Alerts Section
        st.subheader("🔔 Configure Alerts")
        st.write("Set thresholds for failure probabilities to trigger alerts.")
        alert_threshold = st.slider("Alert Threshold (Probability)", min_value=0.0, max_value=1.0, value=0.8, step=0.01)
        st.write(f"Alerts will be triggered for probabilities greater than {alert_threshold:.2f}.")

        # Visualize Risks Section
        st.subheader("📊 Visualize Risks")
        high_risk = predictions_df[predictions_df["Probability"] > alert_threshold]
        st.write(f"Number of high-risk predictions: {len(high_risk)}")
        st.write(high_risk)
    else:
        st.warning("Please upload data with a 'failure' or 'failed' column in the 'Upload Data' section.")

# Visualizations Section
elif page == "Visualizations":
    st.subheader("📊 Visualizations")
    if st.session_state.df is not None:
        # Failure distribution chart
        st.subheader("📈 Failure Distribution")
        if st.session_state.failure_column:
            failure_counts = st.session_state.df[st.session_state.failure_column].value_counts()
            fig, ax = plt.subplots(figsize=(8, 6))  # Adjusted size
            ax.bar(failure_counts.index, failure_counts.values, color=["green", "red"])
            ax.set_title("Failure Distribution")
            ax.set_xlabel("Failure (0=No, 1=Yes)")
            ax.set_ylabel("Count")
            st.pyplot(fig)
            st.caption("This chart shows the distribution of failure events in the dataset.")
        else:
            st.warning("The dataset does not contain a 'failure' or 'failed' column.")

        # Historical trends
        st.subheader("📈 Historical Failure Trends")
        if st.session_state.timestamp_column and st.session_state.failure_column:
            st.session_state.df[st.session_state.timestamp_column] = pd.to_datetime(st.session_state.df[st.session_state.timestamp_column])
            failure_trend = st.session_state.df.groupby(st.session_state.timestamp_column)[st.session_state.failure_column].sum()
            fig, ax = plt.subplots(figsize=(8, 6))  # Adjusted size
            ax.plot(failure_trend.index, failure_trend.values, marker="o")
            ax.set_title("Failure Trends Over Time")
            ax.set_xlabel("Time")
            ax.set_ylabel("Failure Count")
            st.pyplot(fig)
            st.caption("This graph shows how failure events have occurred over time.")
        else:
            st.warning("The dataset must contain both 'timestamp' and 'failure' or 'failed' columns for this visualization.")

        # Correlation heatmap
        st.subheader("📊 Correlation Heatmap")
        numeric_df = st.session_state.df.select_dtypes(include=["number"])
        if not numeric_df.empty:
            corr = numeric_df.corr()
            fig, ax = plt.subplots(figsize=(6, 4))  # Adjusted size
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
            st.caption("This heatmap shows the correlation between numeric features in the dataset.")
        else:
            st.warning("No numeric columns available for correlation heatmap.")
    else:
        st.warning("Please upload data in the 'Upload Data' section.")

# Settings Section
elif page == "Settings":
    st.subheader("⚙️ Settings")
    st.write(
        """
        This section allows you to configure the dashboard settings, such as:
        - **Theme**: Customize the appearance of the dashboard.
        - **Alert Thresholds**: Set thresholds for failure predictions.
        - **User Preferences**: Save user-specific settings for a personalized experience.
        """
    )

    # Placeholder for future settings functionality
    st.info("⚙️ Settings functionality is under development. Stay tuned for updates!")

# Footer
st.markdown(
    """
    <hr>
    <p style="text-align: center; color: gray;">© 2025 Energysage. All rights reserved.</p>
    """,
    unsafe_allow_html=True,
)