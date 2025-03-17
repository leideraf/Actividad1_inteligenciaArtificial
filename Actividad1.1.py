import tensorflow as tf
import numpy as np

# Generar datos de entrenamiento (X e Y)
X_train = np.array([0, 1, 2, 3, 4, 5], dtype=float)
Y_train = np.array([1, 3, 5, 7, 9, 11], dtype=float)  # Y = 2X + 1

# Definir el modelo (una sola neurona)
modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])  # Capa con 1 neurona
])

# Compilar el modelo
modelo.compile(optimizer="sgd", loss="mean_squared_error")

# Entrenar el modelo
print("Entrenando el modelo...")
modelo.fit(X_train, Y_train, epochs=500, verbose=0)  # Entrenar sin mostrar progreso

# Realizar una predicción
X_nuevo = np.array([10], dtype=float)
Y_pred = modelo.predict(X_nuevo)

print(f"\nPredicción para X={X_nuevo[0]}: {Y_pred[0][0]:.2f}")  # Debería ser ≈ 21
