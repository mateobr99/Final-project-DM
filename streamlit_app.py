import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import gzip
import pickle

def load_model():
    filename = "best_model.pkl.gz"
    with gzip.open(filename, 'rb') as f:
        model = pickle.load(f)
    return model
    
# Cargar el modelo
model = load_model_one()

st.title("Predicción de Temperatura")


humedad = st.slider("Humedad (%)", min_value=0, max_value=100, value=50)
viento = st.slider("Viento (km/h)", min_value=0.0, max_value=50.0, value=10.0)
precipitaciones = st.slider("Precipitación (mm)", min_value=0.0, max_value=100.0, value=5.0)


ciudad = st.text_input("Ciudad (opcional)", "Ciudad Desconocida")


input_data = np.array([[humedad, viento, precipitaciones]])


if st.button("Predecir Temperatura"):
    prediccion = model.predict(input_data)
    st.success(f"Temperatura estimada en {ciudad}: {prediccion[0][0]:.2f}°C")
