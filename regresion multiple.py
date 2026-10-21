import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


np.random.seed(42)
n = 300

publicidad = np.random.uniform(100, 1000, n)
precio = np.random.uniform(10, 100, n)
clima = np.random.uniform(0, 40, n)

ventas = (
    50
    + 0.8 * publicidad
    - 2.5 * precio
    + 3 * clima
    + np.random.normal(0, 50, n)
)

df = pd.DataFrame({
    "publicidad": publicidad,
    "precio": precio,
    "clima": clima,
    "ventas": ventas
})


X = df[["publicidad", "precio", "clima"]]
y = df["ventas"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


modelo = LinearRegression()
modelo.fit(X_train, y_train)

predicciones = modelo.predict(X_test)

r2 = r2_score(y_test, predicciones)
mae = mean_absolute_error(y_test, predicciones)
rmse = np.sqrt(mean_squared_error(y_test, predicciones))

print("Coeficientes de cada predictor:")
for variable, coeficiente in zip(X.columns, modelo.coef_):
    print(f" {variable}: {coeficiente:.4f}")

print(f"\nIntercepto: {modelo.intercept_:.4f}")
print(f"R²: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")


nuevo_caso = pd.DataFrame({
    "publicidad": [500],
    "precio": [40],
    "clima": [25]
})

prediccion_nueva = modelo.predict(nuevo_caso)
print(f"\nPredicción de ventas para el nuevo caso: {prediccion_nueva[0]:.2f}")


fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(
    df["publicidad"],
    df["precio"],
    df["ventas"],
    c=df["clima"],
    cmap="viridis",
    s=20
)


ax.set_xlabel("Publicidad")
ax.set_ylabel("Precio")
ax.set_zlabel("Ventas")
ax.set_title("Regresión múltiple: predictores y variable de respuesta")

plt.tight_layout()
plt.show()