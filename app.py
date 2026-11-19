import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="SentinelGuard - Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

# --------------------------------
# HEADER
# --------------------------------

st.title("🛡️ SentinelGuard")
st.subheader("Credit Card Fraud Detection System")

st.write(
    "An intelligent Machine Learning application "
    "to identify fraudulent credit card transactions."
)

st.divider()

# --------------------------------
# LOAD MACHINE LEARNING MODEL
# --------------------------------

@st.cache_resource
def load_model():

    model_path = "fraud_detection_model (1).pkl"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    return joblib.load(model_path)


def find_predictor(obj):

    if hasattr(obj, "predict"):
        return obj

    if isinstance(obj, dict):

        preferred_keys = [
            "model",
            "classifier",
            "best_model",
            "rf_model",
            "random_forest",
            "pipeline"
        ]

        for key in preferred_keys:
            if key in obj:
                result = find_predictor(obj[key])
                if result is not None:
                    return result

        for value in obj.values():
            result = find_predictor(value)
            if result is not None:
                return result

    return None


# --------------------------------
# SIDEBAR NAVIGATION
# --------------------------------

st.sidebar.title("Navigation")

menu = st.sidebar.radio(
    "Select Page",
    [
        "Home",
        "Fraud Detection",
        "About Project"
    ]
)

# --------------------------------
# HOME PAGE
# --------------------------------

if menu == "Home":

    st.header("Welcome to SentinelGuard")

    st.write("""
    SentinelGuard is a Credit Card Fraud Detection
    application developed using Machine Learning.

    It helps identify suspicious transactions
    and distinguish fraudulent transactions
    from legitimate transactions.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.info("🤖 Machine Learning Based Detection")

    with col2:
        st.success("📊 Transaction Analysis")

    st.subheader("Project Features")

    st.write("""
    - Upload credit card transaction datasets.
    - Analyze transaction information.
    - Predict fraudulent transactions.
    - Display fraud detection statistics.
    - Download prediction results.
    """)

    st.divider()

    st.write("Developed using Python and Streamlit.")


# --------------------------------
# FRAUD DETECTION PAGE
# --------------------------------

elif menu == "Fraud Detection":

    st.header("Credit Card Fraud Detection")

    st.write(
        "Upload your credit card transaction CSV file "
        "to identify fraudulent transactions."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV Dataset",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            # Read dataset
            data = pd.read_csv(uploaded_file)

            st.success("Dataset uploaded successfully!")

            # Dataset information
            st.subheader("Dataset Preview")

            st.dataframe(data.head(10))

            col1, col2 = st.columns(2)

            col1.metric(
                "Total Rows",
                data.shape[0]
            )

            col2.metric(
                "Total Columns",
                data.shape[1]
            )

            st.divider()

            # Load model
            st.subheader("Machine Learning Model")

            saved_model = load_model()

            model = find_predictor(saved_model)

            if model is None:

                st.error(
                    "No prediction model found in the saved file."
                )

                if isinstance(saved_model, dict):
                    st.write(
                        "Available model keys:",
                        list(saved_model.keys())
                    )

                st.stop()

            st.success("Machine Learning model loaded successfully!")

            # Prepare dataset
            st.subheader("Preparing Dataset")

            X = data.drop(
                columns=["Class", "Prediction", "Result"],
                errors="ignore"
            )

            # Check expected model features
            expected_features = getattr(
                model,
                "feature_names_in_",
                None
            )

            if expected_features is not None:

                missing_columns = [
                    col for col in expected_features
                    if col not in X.columns
                ]

                if missing_columns:

                    st.error(
                        "Missing required model features: "
                        + ", ".join(missing_columns)
                    )

                    st.stop()

                X = X[list(expected_features)]

            # Check for numeric data
            if not all(
                pd.api.types.is_numeric_dtype(X[col])
                for col in X.columns
            ):

                st.error(
                    "Dataset contains non-numeric feature columns."
                )

                st.stop()

            # Make predictions
            st.subheader("Fraud Detection Results")

            if st.button("Predict Fraud"):

                predictions = model.predict(X)
        # MODEL PERFORMANCE
        st.subheader("Model Performance")

        if "Class" in data.columns:

            y_true = data["Class"].astype(int)

            accuracy = accuracy_score(y_true, predictions)
            precision = precision_score(
                y_true, predictions, zero_division=0
            )
            recall = recall_score(
                y_true, predictions, zero_division=0
            )
            f1 = f1_score(
                y_true, predictions, zero_division=0
            )

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Accuracy", f"{accuracy * 100:.2f}%")
            col2.metric("Precision", f"{precision * 100:.2f}%")
            col3.metric("Recall", f"{recall * 100:.2f}%")
            col4.metric("F1 Score", f"{f1 * 100:.2f}%")

            # Confusion Matrix

            st.subheader("Confusion Matrix")

            cm = confusion_matrix(y_true, predictions)

            cm_df = pd.DataFrame(
                cm,
                index=["Actual Legitimate", "Actual Fraud"],
                columns=["Predicted Legitimate", "Predicted Fraud"]
            )

            st.dataframe(cm_df)

        else:
            st.info(
                "Model performance requires a CSV file containing "
                "the actual Class column."
            )

                results = data.copy()

                results["Prediction"] = predictions

                results["Result"] = results[
                    "Prediction"
                ].map({
                    0: "Legitimate",
                    1: "Fraudulent"
                })

                st.success("Prediction completed successfully!")

                # Calculate statistics
                total = len(predictions)

                fraud = int(
                    (predictions == 1).sum()
                )

                normal = int(
                    (predictions == 0).sum()
                )

                # Display metrics
                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Total Transactions",
                    total
                )

                col2.metric(
                    "Normal Transactions",
                    normal
                )

                col3.metric(
                    "Fraudulent Transactions",
                    fraud
                )

                # Fraud percentage
                if total > 0:

                    fraud_percentage = (
                        fraud / total
                    ) * 100

                    st.write(
                        f"Fraud Percentage: "
                        f"{fraud_percentage:.2f}%"
                    )

                st.divider()

                # Display results
                st.subheader("Prediction Table")

                st.dataframe(results.head(100))

                # Download results
                csv = results.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="Download Prediction Results",
                    data=csv,
                    file_name="fraud_predictions.csv",
                    mime="text/csv"
                )

        except Exception as e:

            st.error(
                f"An error occurred: {e}"
            )

    else:

        st.info(
            "Please upload a CSV dataset to continue."
        )


# --------------------------------
# ABOUT PROJECT PAGE
# --------------------------------

elif menu == "About Project":

    st.header("About Our Project")

    st.write("""
    SentinelGuard is a Credit Card Fraud Detection
    application developed using Machine Learning.

    The project identifies potentially fraudulent
    transactions by analyzing transaction data.
    """)

    st.subheader("Technology Used")

    st.write("""
    - Python
    - Pandas
    - NumPy
    - Scikit-learn
    - Random Forest Classifier
    - Streamlit
    - Joblib
    """)

    st.subheader("Algorithm Used")

    st.write("Random Forest Classifier")

    st.subheader("Project Objective")

    st.write("""
    The main objective is to detect fraudulent
    credit card transactions and help improve
    transaction security.
    """)

    st.success("Thank you for using SentinelGuard!")
