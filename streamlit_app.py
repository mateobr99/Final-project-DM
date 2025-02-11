import streamlit as st
import numpy as np
import tensorflow as tf
import gzip
import pickle
import pandas as pd

# Cargar el modelo
def load_best_model():
    filename = "best_model (3).pkl.gz"
    with gzip.open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

model = load_best_model()

st.title("Predicción de Temperatura")

# Lista de ciudades (debe coincidir con las del modelo)
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

# Crear DataFrame para visualizar mejor el one-hot encoding
df_one_hot = pd.DataFrame({'Ciudad': cities_order, 'One-Hot Encoding': selected_city_vector})

# Mostrar tabla con nombres de ciudades y su vector one-hot
st.subheader("Vector One-Hot de la Ciudad Seleccionada")
st.table(df_one_hot)

# Formar el vector de entrada completo
input_data = np.array([[humedad, viento, precipitaciones] + selected_city_vector])

# Mostrar la tabla con los valores de entrada
st.subheader("Datos de Entrada al Modelo")
df_input = pd.DataFrame(input_data, columns=['Humedad', 'Viento', 'Precipitación'] + cities_order)
st.table(df_input)

# Botón para predecir
if st.button("Predecir Temperatura"):
    try:
        prediccion = model.predict(input_data)
        st.success(f"Temperatura estimada: {prediccion[0][0]:.2f}°C")
    except Exception as e:
        st.error(f"Error en la predicción: {str(e)}")
