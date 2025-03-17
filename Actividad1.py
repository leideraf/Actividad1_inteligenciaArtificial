# Importar las librerías necesarias
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Generar datos de ejemplo
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])  # Variable independiente
y = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])  # Variable dependiente

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo de regresión lineal
modelo = LinearRegression()

# Entrenar el modelo con los datos de entrenamiento
modelo.fit(X_train, y_train)

# Realizar predicciones sobre el conjunto de prueba
y_pred = modelo.predict(X_test)

# Mostrar la ecuación de la recta
print(f"Ecuación del modelo: y = {modelo.coef_[0]:.2f}x + {modelo.intercept_:.2f}")

# Graficar los datos y la regresión lineal
plt.scatter(X, y, color="blue", label="Datos reales")
plt.plot(X, modelo.predict(X), color="red", linestyle="--", label="Regresión lineal")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.title("Regresión Lineal con Scikit-Learn")
plt.show()


