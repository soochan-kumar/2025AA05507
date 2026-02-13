import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from model.train_models import train_all_models

st.set_page_config(page_title="ML Classification Dashboard", layout="wide")

st.title("Diabetes Classification Model Comparison")

st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Training Models...")
    results = train_all_models(df)

    model_choice = st.selectbox(
        "Select Model",
        list(results.keys())
    )

    metrics = results[model_choice]

    st.subheader("Evaluation Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
    col2.metric("AUC", f"{metrics['AUC']:.4f}")
    col3.metric("Precision", f"{metrics['Precision']:.4f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Recall", f"{metrics['Recall']:.4f}")
    col5.metric("F1 Score", f"{metrics['F1']:.4f}")
    col6.metric("MCC", f"{metrics['MCC']:.4f}")

    st.subheader("Confusion Matrix")

    fig, ax = plt.subplots()
    sns.heatmap(metrics["Confusion Matrix"],
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=ax)
    st.pyplot(fig)

    st.subheader("Classification Report")
    st.text(metrics["Classification Report"])

    st.subheader("Observation About Model Performance")

    observation = f"""
    • Accuracy: {metrics['Accuracy']:.4f}  
    • AUC: {metrics['AUC']:.4f}  
    • Precision: {metrics['Precision']:.4f}  
    • Recall: {metrics['Recall']:.4f}  
    • F1 Score: {metrics['F1']:.4f}  
    • MCC: {metrics['MCC']:.4f}  

    This model shows balanced performance depending on dataset characteristics.
    """

    st.write(observation)

else:
    st.info("Please upload a CSV file to continue.")
