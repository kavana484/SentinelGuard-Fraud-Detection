import streamlit as st
import pandas as pd
import numpy as np
import joblib 

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# Website title
st.title("💳 Credit Card Fraud Detection System")

# Subtitle
st.write(
    "Welcome to our Machine Learning based "
    "Credit Card Fraud Detection System."
)

# Description
st.info(
    "This application uses a Random Forest "
    "Classifier to identify potentially fraudulent transactions."
)

# Sidebar
st.sidebar.title("Navigation")

menu = st.sidebar.selectbox(
    "Select Page",
    ["Home", "Fraud Detection", "About Project"]
)

# Home page
if menu == "Home":

    st.header("Welcome!")

    st.write(
        "This system helps identify whether "
        "a credit card transaction is legitimate or fraudulent."
    )

    st.success("Our Machine Learning Model is ready!")

    st.image(
        "https://images.unsplash.com/photo-1556742049-0cfed4f6a45d?w=1000",
        caption="Secure Digital Transactions"
    )

    # Fraud Detection Page
elif menu == "Fraud Detection":

    st.header("🔍 Fraud Detection")

    st.write("Upload your credit card transaction CSV file.")

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        data = pd.read_csv(uploaded_file)

        st.success("Dataset uploaded successfully!")

        st.write("Dataset Preview")

        st.dataframe(data.head())

        st.write("Dataset Shape:", data.shape)
# Load Machine Learning Model
model = joblib.load("fraud_detection_model (1).pkl")

st.write("Model type:", type(model))

if isinstance(model, dict):
    st.write("Model keys:", list(model.keys()))
    st.stop()
    # Prepare data for prediction
    X = data.drop(columns=["Class"], errors="ignore")
    # Make predictions
        predictions = model.predict(X)
# Add predictions to dataset
        data["Prediction"] = predictions
# Display results
        st.subheader("Fraud Detection Results")

        total = len(predictions)
        fraud = int((predictions == 1).sum())
        normal = int((predictions == 0).sum())

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Transactions", total)
        col2.metric("Normal Transactions", normal)
        col3.metric("Fraudulent Transactions", fraud)
# Display prediction table
        st.dataframe(data.head(20))

# Download results
        csv = data.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Prediction Results",
            csv,
            "fraud_predictions.csv",
            "text/csv"
        )

# About Project Page
elif menu == "About Project":

    st.header("About Our Project")

    st.write("""
    Credit Card Fraud Detection is a Machine Learning project.

    Algorithm Used: Random Forest Classifier

    Programming Language: Python

    Libraries Used:
    - Pandas
    - NumPy
    - Scikit-learn
    - Streamlit

    Objective:
    To identify fraudulent credit card transactions.
    """)
    
if menu == "Fraud Detection":

    st.header("🔍 Credit Card Fraud Detection")

    st.write("Upload your credit card transaction CSV file.")

    uploaded_file = st.file_uploader(
        "Choose CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        data = pd.read_csv(uploaded_file)

        st.success("Dataset uploaded successfully!")

        st.subheader("Dataset Preview")

        st.dataframe(data.head())

        st.write("Dataset Shape:", data.shape)

        if st.button("Predict Fraud"):

            try:
                # Check dataset columns
                if list(data.columns) != list(X.columns):
                    st.error(
                        "CSV columns do not match the model features."
                    )

                else:
                    # Make predictions
                    predictions = model.predict(data)

                    # Add prediction results
                    results = data.copy()

                    results["Prediction"] = predictions

                    results["Result"] = results["Prediction"].map({
                        0: "Legitimate",
                        1: "Fraudulent"
                    })

                    # Display results
                    st.subheader("Prediction Results")

                    st.dataframe(results)

                    # Count transactions
                    fraud_count = (predictions == 1).sum()

                    legitimate_count = (predictions == 0).sum()

                    col1, col2 = st.columns(2)

                    col1.metric(
                        "Legitimate Transactions",
                        int(legitimate_count)
                    )

                    col2.metric(
                        "Fraudulent Transactions",
                        int(fraud_count)
                    )

                    # Download results
                    csv = results.to_csv(index=False)

                    st.download_button(
                        "Download Results",
                        csv,
                        "fraud_detection_results.csv",
                        "text/csv"
                    )

            except Exception as e:
                st.error(f"Prediction Error: {e}")
                
