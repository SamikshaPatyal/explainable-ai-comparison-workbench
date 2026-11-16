
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance

st.set_page_config(
    page_title="Explainable AI Comparison Workbench",
    page_icon="🤖",
    layout="wide"
)

st.title("Explainable AI Comparison Workbench")
st.write("### Do Explanation Methods Agree?")
st.write(
    "Explore machine learning predictions and compare "
    "different feature explanation methods."
)

# Dataset selection
dataset_name = st.sidebar.selectbox(
    "Select Dataset",
    ["Iris", "Breast Cancer"]
)

if dataset_name == "Iris":
    dataset = load_iris()
else:
    dataset = load_breast_cancer()

X = pd.DataFrame(
    dataset.data,
    columns=dataset.feature_names
)

y = dataset.target

st.subheader("1. Dataset Overview")
st.write("Selected Dataset:", dataset_name)
st.write("Number of Records:", X.shape[0])
st.write("Number of Features:", X.shape[1])
st.dataframe(X.head())

# Train and test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train machine learning model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Model evaluation
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

st.subheader("2. Machine Learning Model Results")

col1, col2 = st.columns(2)

col1.metric("Model Accuracy", f"{accuracy * 100:.2f}%")
col2.metric("Number of Features", X.shape[1])

# Select a test record
st.subheader("3. Individual Prediction")

record_number = st.slider(
    "Select a test record",
    min_value=0,
    max_value=len(X_test) - 1,
    value=0
)

selected_record = X_test.iloc[[record_number]]
prediction = model.predict(selected_record)[0]

class_name = dataset.target_names[prediction]

st.write("Predicted Class:", str(class_name))
st.write("Actual Class:", str(dataset.target_names[y_test[record_number]]))

# Permutation Feature Importance
st.subheader("4. Permutation Feature Importance")

importance = permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=5,
    random_state=42
)

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance.importances_mean
}).sort_values("Importance", ascending=False)

fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(
    importance_df["Feature"].head(10)[::-1],
    importance_df["Importance"].head(10)[::-1]
)
ax.set_title("Top 10 Important Features")
ax.set_xlabel("Importance")
st.pyplot(fig)
plt.close(fig)

# SHAP Explanation
st.subheader("5. SHAP Explanation")

try:
    explainer = shap.Explainer(model, X_train)
    explanation = explainer(selected_record)

    shap_values = np.asarray(explanation.values)

    if shap_values.ndim == 3:
        feature_values = shap_values[0, :, prediction]
    else:
        feature_values = shap_values[0]

    shap_df = pd.DataFrame({
        "Feature": X.columns,
        "SHAP Value": feature_values
    })

    shap_df["Absolute Value"] = shap_df["SHAP Value"].abs()
    shap_df = shap_df.sort_values(
        "Absolute Value",
        ascending=False
    )

    st.write("Feature contributions for the selected prediction:")
    st.dataframe(
        shap_df[["Feature", "SHAP Value"]].head(10)
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(
        shap_df["Feature"].head(10)[::-1],
        shap_df["SHAP Value"].head(10)[::-1]
    )
    ax.set_title("SHAP Feature Contributions")
    ax.set_xlabel("SHAP Value")
    st.pyplot(fig)
    plt.close(fig)

except Exception as error:
    st.warning("SHAP explanation could not be generated.")
    st.write(str(error))

# Comparison
st.subheader("6. Explanation Comparison")

st.write(
    "Permutation Feature Importance shows global feature importance, "
    "while SHAP shows feature contributions to an individual prediction."
)

st.write(
    "Compare the important features in both results to understand "
    "where the explanation methods show similarities or differences."
)

st.success("Analysis completed successfully!")
