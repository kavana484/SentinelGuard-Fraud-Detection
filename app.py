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

# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="SentinelGuard - Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

# -----------------------------------
# LOAD MODEL
# -----------------------------------

MODEL_PATH = "fraud_detection_model (1).pkl"


@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    loaded_model = joblib.load(MODEL_PATH)

    if isinstance(loaded_model, dict):

        for key in [
            "model",
            "classifier",
            "best_model",
            "rf_model",
            "random_forest",
            "pipeline"
        ]:

            if key in loaded_model:
                candidate = loaded_model[key]

                if hasattr(candidate, "predict"):
                    return candidate

    if hasattr(loaded_model, "predict"):
        return loaded_model

    raise ValueError("No prediction model found in PKL file.")


try:
    model = load_model()
    model_loaded = True

except Exception as e:
    model_loaded = False
    model_error = str(e)


# -----------------------------------
# SIDEBAR
# -----------------------------------

st.sidebar.title("🛡️ SentinelGuard")

st.sidebar.write("Credit Card Fraud Detection")

menu = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Fraud Detection",
        "About Project"
    ]
)


# -----------------------------------
# HOME PAGE
# -----------------------------------

if menu == "Home":

    st.title("🛡️ SentinelGuard")

    st.subheader(
        "Credit Card Fraud Detection System"
    )

    st.write(
        """
        SentinelGuard is a Machine Learning based
        application designed to identify fraudulent
        credit card transactions.
        """
    )

    st.info(
        "Upload transaction data to identify potential fraud."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Technology", "Machine Learning")

    col2.metric("Algorithm", "Random Forest")

    col3.metric("Application", "Streamlit")


# -----------------------------------
# FRAUD DETECTION PAGE
# -----------------------------------

elif menu == "Fraud Detection":

    st.title("Credit Card Fraud Detection")

    st.write(
        "Upload your transaction CSV file below."
    )

    if not model_loaded:

        st.error(
            f"Model loading failed: {model_error}"
        )

        st.stop()

    uploaded_file = st.file_uploader(
        "Upload CSV Dataset",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            data = pd.read_csv(uploaded_file)

            st.success("Dataset uploaded successfully!")

            # DATASET INFORMATION

            st.subheader("Dataset Overview")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Total Rows",
                data.shape[0]
            )

            col2.metric(
                "Total Columns",
                data.shape[1]
            )

            col3.metric(
                "Missing Values",
                int(data.isnull().sum().sum())
            )

            # DATASET PREVIEW

            st.subheader("Dataset Preview")

            st.dataframe(
                data.head(10),
                use_container_width=True
            )

            # CHECK MODEL FEATURES

            expected_features = getattr(
                model,
                "feature_names_in_",
                None
            )

            # REMOVE TARGET COLUMNS

            target_columns = [
                "Class",
                "Prediction",
                "Result"
            ]

            X = data.drop(
                columns=[
                    col for col in target_columns
                    if col in data.columns
                ],
                errors="ignore"
            )

            # SELECT REQUIRED FEATURES

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

            else:

                X = X.select_dtypes(
                    include=[np.number]
                )

                expected_count = getattr(
                    model,
                    "n_features_in_",
                    None
                )

                if (
                    expected_count is not None
                    and X.shape[1] != expected_count
                ):

                    st.error(
                        f"Model expects {expected_count} "
                        f"features, but dataset has "
                        f"{X.shape[1]} numeric features."
                    )

                    st.stop()

                st.warning(
                    "Feature order is based on the uploaded "
                    "CSV. Ensure it matches model training order."
                )

            # CHECK NUMERIC DATA

            if not all(
                pd.api.types.is_numeric_dtype(X[col])
                for col in X.columns
            ):

                st.error(
                    "Dataset contains non-numeric feature columns."
                )

                st.stop()

            if X.isnull().sum().sum() > 0:

                st.error(
                    "Dataset contains missing values. "
                    "Please clean the dataset first."
                )

                st.stop()

            # MAKE PREDICTIONS

            st.subheader("Fraud Detection Results")

            if st.button("Predict Fraud"):

                predictions = model.predict(X)

                results = data.copy()

                results["Prediction"] = predictions

                results["Result"] = [
                    "Fraudulent" if p == 1
                    else "Legitimate"
                    for p in predictions
                ]

                # PREDICTION SUMMARY

                fraud_count = int(
                    np.sum(predictions == 1)
                )

                legitimate_count = int(
                    np.sum(predictions == 0)
                )

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Total Transactions",
                    len(predictions)
                )

                col2.metric(
                    "Fraudulent Transactions",
                    fraud_count
                )

                col3.metric(
                    "Legitimate Transactions",
                    legitimate_count
                )

                # RESULTS TABLE

                st.subheader("Prediction Details")

                st.dataframe(
                    results,
                    use_container_width=True
                )

                # MODEL PERFORMANCE

                if "Class" in data.columns:

                    st.subheader("Model Performance")

                    y_true = data["Class"].astype(int)

                    accuracy = accuracy_score(
                        y_true,
                        predictions
                    )

                    precision = precision_score(
                        y_true,
                        predictions,
                        zero_division=0
                    )

                    recall = recall_score(
                        y_true,
                        predictions,
                        zero_division=0
                    )

                    f1 = f1_score(
                        y_true,
                        predictions,
                        zero_division=0
                    )

                    col1, col2, col3, col4 = st.columns(4)

                    col1.metric(
                        "Accuracy",
                        f"{accuracy * 100:.2f}%"
                    )

                    col2.metric(
                        "Precision",
                        f"{precision * 100:.2f}%"
                    )

                    col3.metric(
                        "Recall",
                        f"{recall * 100:.2f}%"
                    )

                    col4.metric(
                        "F1 Score",
                        f"{f1 * 100:.2f}%"
                    )

                    # CONFUSION MATRIX

                    st.subheader("Confusion Matrix")

                    cm = confusion_matrix(
                        y_true,
                        predictions,
                        labels=[0, 1]
                    )

                    cm_df = pd.DataFrame(
                        cm,
                        index=[
                            "Actual Legitimate",
                            "Actual Fraud"
                        ],
                        columns=[
                            "Predicted Legitimate",
                            "Predicted Fraud"
                        ]
                    )

                    st.dataframe(
                        cm_df,
                        use_container_width=True
                    )

                else:

                    st.info(
                        "Upload a dataset containing the "
                        "'Class' column to calculate "
                        "model performance metrics."
                    )

                # DOWNLOAD RESULTS

                csv = results.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="Download Prediction Results",
                    data=csv,
                    file_name="fraud_detection_results.csv",
                    mime="text/csv"
                )

                st.success(
                    "Fraud detection completed successfully!"
                )

        except Exception as e:

            st.error(
                f"An error occurred: {e}"
            )


# -----------------------------------
# ABOUT PROJECT
# -----------------------------------

elif menu == "About Project":

    st.title("About SentinelGuard")

    st.write(
        """
        SentinelGuard is a Credit Card Fraud Detection
        project developed using Machine Learning.

        Project Details:

        - Project Name: SentinelGuard
        - Domain: Machine Learning
        - Algorithm: Random Forest Classifier
        - Programming Language: Python
        - Framework: Streamlit
        - Dataset: Credit Card Transactions

        Objective:

        To identify fraudulent credit card transactions
        and distinguish them from legitimate transactions.
        """
    )

    st.success(
        "SentinelGuard - Intelligent Fraud Detection System"
    )
