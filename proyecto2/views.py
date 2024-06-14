# Se importan las librerias para el template y los renders
from django.shortcuts import render
from tensorflow.keras.models import Sequential


# Librerias de la Red Neuronal
import numpy as np
import pandas as pd

# Librerias para gráficas
import matplotlib.pyplot as plt
import seaborn as sns

# Libreria para dividir los datos
from sklearn.model_selection import train_test_split

# Libreria para métricas
from sklearn.metrics import confusion_matrix, classification_report

# Framework Tensorflow
import tensorflow as tf

from tensorflow.keras.layers import Dense


# -----------------------------------


def main(request):
    # *** Plantilla ***
    return render(request, "index.html", context={})


def prediccion(request):
    # Datos de entrenamiento
    datos_entrenamiento = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])

    # Datos de salida esperados
    datos_salida = np.array([[0], [1], [1], [0]])

    # Definición del modelo
    model = Sequential()
    model.add(Dense(4, input_dim=2, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    # Compilación del modelo
    model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])

    # Entrenamiento del modelo
    model.fit(datos_entrenamiento, datos_salida, epochs=200)

    # Evaluación del modelo con los datos de entrenamiento
    scores = model.evaluate(datos_entrenamiento, datos_salida)

    # Precisión del modelo
    precision = scores[1] * 100

    # Predicciones
    predicciones = model.predict(datos_entrenamiento).round()

    # Contexto para pasar a la plantilla
    context = {"precision": precision, "predicciones": predicciones}

    # Renderizar la plantilla con el contexto
    return render(request, "index.html", context=context)

