import streamlit as st
import numpy as np
import tensorflow as tf
import gzip
import pickle

# Títulos y gráficos exploratorios
st.title("Análisis Exploratorio del Clima")

st.subheader("Matriz de correlación")
st.image("correlation_matrix.png")

st.subheader("Distribución de Humedad")
st.image("humidity_distribution.png")

st.subheader("Outliers en variables numéricas")
st.image("boxplot_Temperature_C.png")
st.image("boxplot_Humidity_pct.png")
st.image("boxplot_Wind_Speed_kmh.png")

# Función para cargar el modelo guardado (evitando conflictos con tf.keras.models.load_model)
def load_best_model():
    filename = "best_model (3).pkl.gz"
    with gzip.open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

# Cargar el modelo
model = load_best_model()

st.title("Predicción de Temperatura")

# Lista de ciudades (asegúrate de que esta lista coincida con la utilizada en el entrenamiento)
cities_order = ['San Diego', 'Philadelphia', 'San Antonio', 'San Jose', 'New York',
                'Houston', 'Dallas', 'Chicago', 'Los Angeles', 'Phoenix']

# Select para elegir la ciudad
selected_city = st.selectbox("Selecciona la Ciudad", cities_order)

# Sliders para las variables numéricas
humedad = st.slider("Humedad (%)", min_value=0, max_value=100, value=50)
viento = st.slider("Viento (km/h)", min_value=0.0, max_value=50.0, value=10.0)
precipitaciones = st.slider("Precipitación (mm)", min_value=0.0, max_value=100.0, value=5.0)

# Generar el vector one-hot para la ciudad seleccionada
selected_city_vector = [1 if city == selected_city else 0 for city in cities_order]

# Formar el vector de entrada completo: 
# se concatenan los 3 valores numéricos con el vector one-hot (dimensión total = 3 + len(cities_order))
input_data = np.array([[humedad, viento, precipitaciones] + selected_city_vector])

# Mostrar la forma y contenido del vector para confirmar
st.write("Forma de input_data:", input_data.shape)
st.write("input_data:", input_data)

# Botón para predecir
if st.button("Predecir Temperatura"):
    try:
        prediccion = model.predict(input_data)
        st.success(f"Temperatura estimada: {prediccion[0][0]:.2f}°C")
    except Exception as e:
        st.error(f"Error en la predicción: {str(e)}")
