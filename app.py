
import streamlit as st
import joblib
import os
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Iris Flower Predictor", page_icon="🌸")

@st.cache_resource
def load_model():
    model_path = "iris_model.pkl"

    if not os.path.exists(model_path):
        iris = load_iris()
        model = RandomForestClassifier(random_state=42)
        model.fit(iris.data, iris.target)
        joblib.dump(model, model_path)

    return joblib.load(model_path)

model = load_model()
iris = load_iris()

st.title("🌸 Iris Flower Prediction")
st.write("Enter flower measurements to predict its species.")

sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 0.2)

if st.button("Predict Flower"):
    prediction = model.predict([[
        sepal_length, sepal_width,
        petal_length, petal_width
    ]])[0]
    st.success(f"Predicted flower: {iris.target_names[prediction]}")
