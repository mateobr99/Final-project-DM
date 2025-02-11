import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Cargar el modelo correctamente
@st.cache_resource  # Optimiza la carga en Streamlit
def load_model():
    filename = "best_model.pkl.gz"  # Asegúrate de que el archivo correcto esté aquí
    model = load_model(filename)
    return model

model = load_model()

st.title("Predicción de Temperatura")

# Entrada manual de datos
humedad = st.slider("Humedad (%)", min_value=0, max_value=100, value=50)
viento = st.slider("Viento (km/h)", min_value=0.0, max_value=50.0, value=10.0)
precipitaciones = st.slider("Precipitación (mm)", min_value=0.0, max_value=100.0, value=5.0)

# Permitir ingresar la ciudad manualmente (opcional)
ciudad = st.text_input("Ciudad (opcional)", "Ciudad Desconocida")

# Convertir entrada a array con la misma forma esperada por el modelo
input_data = np.array([[humedad, viento, precipitaciones]])

# Botón para predecir
if st.button("Predecir Temperatura"):
    prediccion = model.predict(input_data)
    st.success(f"Temperatura estimada en {ciudad}: {prediccion[0][0]:.2f}°C")
