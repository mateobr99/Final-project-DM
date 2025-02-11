import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import gzip
import pickle

st.title("Análisis Exploratorio del Clima")

st.subheader("Matriz de correlación")
st.image("correlation_matrix.png")

st.subheader("Distribución de Humedad")
st.image("humidity_distribution.png")

st.subheader("Outliers en variables numéricas")
st.image("boxplot_Temperature_C.png")
st.image("boxplot_Humidity_pct.png")
st.image("boxplot_Wind_Speed_kmh.png")

def load_model():
    filename = "best_model (3).pkl.gz"
    with gzip.open(filename, 'rb') as f:
        model = pickle.load(f)
    return model
    
# Cargar el modelo
model = load_model()

st.title("Predicción de Temperatura")

cities_order = ['San Diego', 'Philadelphia', 'San Antonio', 'San Jose', 'New York',
                'Houston', 'Dallas', 'Chicago', 'Los Angeles', 'Phoenix']

# Mostrar un select box para elegir la ciudad
selected_city = st.selectbox("Selecciona la Ciudad", cities_order)

humedad = st.slider("Humedad (%)", min_value=0, max_value=100, value=50)
viento = st.slider("Viento (km/h)", min_value=0.0, max_value=50.0, value=10.0)
precipitaciones = st.slider("Precipitación (mm)", min_value=0.0, max_value=100.0, value=5.0)

# Crear el vector one-hot para la ciudad seleccionada
selected_city_vector = [1 if city == selected_city else 0 for city in cities_order]
st.write("Vector one-hot:", selected_city_vector)

# Formar el vector de entrada completo (asegúrate de respetar el orden usado en el entrenamiento)
input_data = np.array([[humedad, viento, precipitaciones] + selected_city_vector])

if st.button("Predecir Temperatura"):
    prediccion = model.predict(input_data)
    st.success(f"Temperatura estimada: {prediccion[0][0]:.2f}°C")
