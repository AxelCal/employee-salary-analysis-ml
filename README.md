#  Employee Salary Prediction API

Este proyecto implementa un sistema de Inteligencia Artificial capaz de predecir el salario de un empleado basado en su **edad** y **departamento**. El proyecto abarca desde el entrenamiento del modelo (Machine Learning Pipeline) hasta su exposición mediante una **API REST**.

##  Decisiones Técnicas

- **Modelo:** Regresión Lineal (`Scikit-Learn`). Se eligió por su rapidez y, sobre todo, por su alta **explicabilidad** a través de coeficientes.
- **Preprocesamiento:** Se aplicó **One-Hot Encoding** para convertir variables categóricas (departamentos) en numéricas, permitiendo que el modelo matemático las procese.
- **API:** Desarrollada con `FastAPI` (Python), seleccionada por su validación automática de tipos de datos y su generación automática de documentación interactiva.

##  Resultados del Modelo (Análisis Técnico)

Tras el último entrenamiento realizado, el modelo arrojó las siguientes métricas:

- **MAE (Error Medio Absoluto):** 310.27 (En promedio, las predicciones tienen una variación de $310).
- **R2 Score:** 0.36 (Un punto de partida sólido que explica el 36% de la varianza de los datos).

### 🔹 Importancia de Variables (Coeficientes)
Los coeficientes nos indican cómo afecta cada variable al salario final:
- **Edad:** **+58.38** (Por cada año de edad, el salario estimado sube $58.38).
- **Ventas:** **+190.36** (Es el departamento con mayor impacto positivo en el sueldo base).
- **Marketing:** **-118.02** (Impacto relativo respecto a la media del dataset).

## 🛠️ Instalación y Uso

### 1. Preparar el entorno
Asegúrate de tener Python instalado y las librerías necesarias:
```bash
pip install fastapi uvicorn pandas scikit-learn joblib
