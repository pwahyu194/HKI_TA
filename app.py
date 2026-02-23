import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Memory optimization functions

def set_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)  # Memory growth must be set at program startup

set_memory_growth()

# Streamlit app
st.title('LSTM Training with Optimization')

# User inputs
sequence_length = st.sidebar.slider('Sequence Length', 10, 100, 30)
num_features = st.sidebar.slider('Number of Features', 1, 10, 5)

# Dummy Data Generation
@st.cache_data
def generate_data(samples):
    x = np.random.rand(samples, sequence_length, num_features)
    y = np.random.rand(samples, 1)
    return x, y

X, y = generate_data(1000)

# Model Definition
model = keras.Sequential([
    keras.layers.LSTM(50, activation='relu', input_shape=(sequence_length, num_features)),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Training
if st.button('Train Model'):
    history = model.fit(X, y, epochs=10, verbose=1)
    st.success('Model trained successfully!')
    st.line_chart(history.history['loss'])
