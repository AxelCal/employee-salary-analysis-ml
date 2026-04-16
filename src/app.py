from fastapi import FastAPI
import pandas as pd
import joblib
import os

app = FastAPI()

# --- CONFIGURACIÓN DE RUTAS SEGURAS ---
ruta_actual = os.path.dirname(os.path.abspath(__file__))
ruta_modelo = os.path.join(ruta_actual, "..", "data", "modelo_salarios.pkl")

print(f"📡 API intentando cargar modelo desde: {ruta_modelo}")

if not os.path.exists(ruta_modelo):
    raise FileNotFoundError(f" La API no encontró el archivo .pkl en: {ruta_modelo}")

# Carga del modelo en RAM
modelo = joblib.load(ruta_modelo)

@app.get("/")
def home():
    return {
        "mensaje": "¡Bienvenido a la API de Predicción de Salarios! 🚀",
        "instrucciones": "Prueba el endpoint /predict o ve a /docs para el simulador interactivo."
    }

# ==============================
# El Endpoint Mágico (Mejorado)
# ==============================
@app.get("/predict")
def predict(edad: int, departamento: str):
    #  1. Validación de Entrada (Pensamiento de producción)
    # Definimos lo que el modelo conoce según tu entrenamiento
    deptos_validos = {
        "IT": "IT",
        "MARKETING": "Marketing",
        "VENTAS": "Ventas",
        "MKT": "Marketing" # Soporte para abreviatura
    }
    
    depto_upper = departamento.strip().upper()

    if edad <= 18 or edad > 80:
        return {"estatus": "error", "mensaje": "La edad debe estar entre 19 y 80 años."}

    if depto_upper not in deptos_validos:
        return {
            "estatus": "error", 
            "mensaje": f"Departamento '{departamento}' no reconocido. Usa: IT, Marketing o Ventas."
        }

    # 🥈 2. Estructura Robusta de Columnas
    # Usamos capitalize para que coincida con 'departamento_Marketing' o 'departamento_Ventas'
    depto_formateado = deptos_validos[depto_upper]
    
    # Recreamos las columnas EXACTAS que el modelo vio al entrenar
    columnas_entrenamiento = ["edad", "departamento_IT", "departamento_Marketing", "departamento_Ventas"]
    
    # Creamos un diccionario base con ceros para evitar que falten columnas
    datos_para_df = {col: 0 for col in columnas_entrenamiento}
    
    # Llenamos los valores recibidos
    datos_para_df["edad"] = edad
    col_activa = f"departamento_{depto_formateado}"
    
    if col_activa in datos_para_df:
        datos_para_df[col_activa] = 1

    # 🥉 3. Conversión y Predicción
    # Convertimos a DataFrame asegurando que el orden de columnas sea idéntico al entrenamiento
    df_input = pd.DataFrame([datos_para_df])[columnas_entrenamiento]

    try:
        pred = modelo.predict(df_input)
        
        # 6. Respuesta Profesional
        return {
            "estatus": "exitoso",
            "datos_recibidos": {
                "edad": edad,
                "departamento": depto_formateado
            },
            "salario_estimado": round(float(pred[0]), 2),
            "moneda": "USD"
        }
    except Exception as e:
        return {"estatus": "error", "mensaje": f"Error en la predicción: {str(e)}"}