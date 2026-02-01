from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

st.set_page_config(page_title="Traffic Violations - Model Evaluation", layout="wide")

st.markdown(
    """
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 0rem;
    padding-left: 1rem;
    padding-right: 1rem;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("Traffic Violations ML Model Evaluation")

st.markdown(
    """
    This application allows you to evaluate multiple trained machine learning models
    on traffic violation data and visualize their performance.
    """
)

ARTIFACTS_DIR = Path("models/artifacts")
DATASETS_DIR = Path("datasets")
METRIC_EXPLANATIONS = {
    "Accuracy": "Overall proportion of correct predictions across all classes.",
    "Precision (Macro)": "How often the model is correct when it predicts a class (averaged equally across classes).",
    "Recall (Macro)": "How well the model identifies all instances of each class (averaged equally).",
    "F1 Score (Macro)": "Balanced measure combining Precision and Recall across all classes.",
    "AUC (OvR)": "Ability of the model to rank the correct class higher than others (One-vs-Rest).",
    "MCC": "Balanced correlation metric considering all prediction outcomes; robust for multi-class problems.",
}

tab_evaluate, tab_compare = st.tabs(["🧪 Evaluate a Model", "📊 Model Comparison"])

with tab_evaluate:
    st.header("Evaluate a Trained Model")

    @st.cache_resource
    def load_label_encoder():
        return joblib.load(ARTIFACTS_DIR / "label_encoder.pkl")

    @st.cache_data
    def load_model_list():
        return sorted(
            [p.stem for p in ARTIFACTS_DIR.glob("*.pkl") if p.stem != "label_encoder"]
        )

    label_encoder = load_label_encoder()
    model_names = load_model_list()

    st.subheader("Available Models")

    if not model_names:
        st.error("No trained models found in artifacts directory.")
        st.stop()

    presentable_model_names = [name.replace("_", " ").title() for name in model_names]
    selected_model_name = st.selectbox(
        "Select a trained model", presentable_model_names
    )
    selected_model = selected_model_name.replace(" ", "_").lower()

    @st.cache_resource
    def load_model(model_name: str):
        model_path = ARTIFACTS_DIR / f"{model_name}.pkl"
        return joblib.load(model_path)

    model_pipeline = load_model(selected_model)

    st.subheader("Upload Test Dataset")

    uploaded_file = st.file_uploader("Upload test CSV file", type=["csv"])

    @st.cache_data
    def load_test_data(file) -> pd.DataFrame:
        return pd.read_csv(file)

    @st.cache_data
    def load_sample_test_data() -> pd.DataFrame:
        sample_path = DATASETS_DIR / "test.csv"
        return pd.read_csv(sample_path)

    st.download_button(
        "Download Sample Test Dataset (CSV)",
        data=load_sample_test_data().to_csv(index=False),
        file_name="sample_test_data.csv",
        mime="text/csv",
        help="Downloads the 20% sample of the original test dataset used during model training.",
    )

    if uploaded_file is None:
        st.warning("Please upload a CSV file to proceed.")
        st.stop()

    test_df = load_test_data(uploaded_file)

    st.write("Preview of uploaded data:")
    st.dataframe(test_df.head())

    TARGET_COLUMN = "Violation_Type"

    if TARGET_COLUMN not in test_df.columns:
        st.error(f"Target column '{TARGET_COLUMN}' not found in uploaded CSV.")
        st.stop()

    # Validate that the uploaded data has the expected feature columns
    sample_df = load_sample_test_data()
    expected_features = [col for col in sample_df.columns if col != TARGET_COLUMN]

    missing_features = [col for col in expected_features if col not in test_df.columns]
    if missing_features:
        st.error(
            "The uploaded CSV is missing required feature columns used by the model:\n"
            + ", ".join(missing_features)
        )
        st.stop()

    unexpected_features = [
        col
        for col in test_df.columns
        if col not in expected_features and col != TARGET_COLUMN
    ]
    if unexpected_features:
        st.warning(
            "The uploaded CSV contains unexpected columns that will be ignored by the model:\n"
            + ", ".join(unexpected_features)
        )
        test_df = test_df.drop(columns=unexpected_features)
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    y_test_enc = label_encoder.transform(y_test)

    st.subheader("Model Evaluation")

    with st.spinner("Running model predictions..."):
        y_pred = model_pipeline.predict(X_test)
        y_prob = model_pipeline.predict_proba(X_test)

    st.success("Predictions completed.")
    st.write("Predicted classes and probabilities:")
    pred_df = pd.DataFrame(y_prob, columns=label_encoder.classes_)
    pred_df["Predicted_Class"] = label_encoder.inverse_transform(y_pred)
    st.dataframe(pred_df.head())

    accuracy = accuracy_score(y_test_enc, y_pred)
    precision = precision_score(y_test_enc, y_pred, average="macro")
    recall = recall_score(y_test_enc, y_pred, average="macro")
    f1 = f1_score(y_test_enc, y_pred, average="macro")
    auc = roc_auc_score(y_test_enc, y_prob, multi_class="ovr")
    mcc = matthews_corrcoef(y_test_enc, y_pred)

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Accuracy", f"{accuracy * 100:.2f}%", help=METRIC_EXPLANATIONS["Accuracy"]
    )

    col2.metric(
        "Precision (Macro)",
        f"{precision * 100:.2f}%",
        help=METRIC_EXPLANATIONS["Precision (Macro)"],
    )

    col3.metric(
        "Recall (Macro)",
        f"{recall * 100:.2f}%",
        help=METRIC_EXPLANATIONS["Recall (Macro)"],
    )

    col4, col5, col6 = st.columns(3)

    col4.metric(
        "F1 Score (Macro)",
        f"{f1 * 100:.2f}%",
        help=METRIC_EXPLANATIONS["F1 Score (Macro)"],
    )

    col5.metric("AUC (OvR)", f"{auc * 100:.2f}%", help=METRIC_EXPLANATIONS["AUC (OvR)"])

    col6.metric("MCC", f"{mcc * 100:.2f}%", help=METRIC_EXPLANATIONS["MCC"])

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test_enc, y_pred)
    cm_df = pd.DataFrame(
        cm, index=label_encoder.classes_, columns=label_encoder.classes_
    )

    col_cm, _ = st.columns([2, 1])  # CM gets 2/3 width
    with col_cm:
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

    st.subheader("Per-Class Performance")

    report = classification_report(
        y_test_enc, y_pred, target_names=label_encoder.classes_, output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

with tab_compare:
    st.header("Model Performance Comparison")

    st.info(
        """
    **Reproducibility Note**

    The results shown here are based on a specific training and evaluation run.
    The same metrics are documented in the GitHub README.md file.

    Model performance may vary depending on:
    - Hardware and compute environment
    - Library versions
    - Random seeds
    - Dataset splits and preprocessing choices

    All comparisons are provided for relative evaluation purposes.
    """
    )

    comparison_path = ARTIFACTS_DIR / "model_comparison.csv"

    if not comparison_path.exists():
        st.error("Model comparison file not found.")
        st.stop()

    comparison_df = pd.read_csv(comparison_path)

    st.markdown(
        """
        This table compares all trained models using the same preprocessing
        pipeline and test dataset.
        """
    )

    st.dataframe(comparison_df)
    st.caption(
        "Higher values are better for all metrics except training and evaluation time."
    )

    st.subheader("Model Performance Visualization")

    metric_to_plot = st.selectbox(
        "Select metric to visualize",
        options=[
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "AUC",
            "MCC",
            "Training Time (s)",
            "Evaluation Time (s)",
        ],
        help="Choose a metric to compare models visually",
    )

    plot_df = comparison_df[["Model", metric_to_plot]].sort_values(
        by=metric_to_plot, ascending=False
    )

    st.bar_chart(plot_df.set_index("Model"))

    st.subheader("Detailed Model Analysis")

    selected_analysis_model = st.selectbox(
        "Select model for detailed analysis",
        options=comparison_df["Model"].tolist(),
        help="View confusion matrix and classification report for a specific model",
    )

    cm_path = (
        ARTIFACTS_DIR
        / "confusion_matrices"
        / f"confusion_matrix_{selected_analysis_model.lower().replace(' ', '_')}.csv"
        if selected_analysis_model
        else None
    )

    report_path = (
        ARTIFACTS_DIR
        / "classification_reports"
        / f"classification_report_{selected_analysis_model.lower().replace(' ', '_')}.csv"
        if selected_analysis_model
        else None
    )

    if cm_path and cm_path.exists():
        col_cm, _ = st.columns([2, 1])  # CM gets 2/3 width

        with col_cm:
            fig, ax = plt.subplots(figsize=(9, 5))
            sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)

    else:
        st.warning("Confusion matrix not found for this model.")

    if report_path and report_path.exists():
        st.markdown("### Classification Report (Per-Class Metrics)")
        report_df = pd.read_csv(report_path, index_col=0)
        st.dataframe(report_df, use_container_width=True)
    else:
        st.warning("Classification report not found for this model.")
