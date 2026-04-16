import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import os

# --- 1. FUNCIONES DE APOYO ---

def cargar_datos():
    ruta_script = os.path.dirname(os.path.abspath(__file__))
    ruta_csv = os.path.join(ruta_script, "..", "data", "empleados.csv")
    print(f" Intentando abrir: {ruta_csv}")
    return pd.read_csv(ruta_csv)

def preparar_datos(df):
    df = pd.get_dummies(df, columns=["departamento"])
    X = df.drop(columns=["nombre", "salario"])
    y = df["salario"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def entrenar_modelo(X_train, y_train):
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    print(" Modelo entrenado correctamente")
    return modelo

def evaluar_y_reportar(modelo, X_test, y_test, X_train):
    # Predicciones
    pred = modelo.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    
    # 🥉 3. Mostrar importancia de variables en consola
    print(f"\n Evaluación del modelo:")
    print(f"MAE: {mae:.2f}")
    print(f"R2: {r2:.2f}")
    
    print("\n Análisis de Coeficientes (Importancia):")
    for col, coef in zip(X_train.columns, modelo.coef_):
        print(f"🔹 {col}: {coef:.2f}")

    #  4. Guardar resultados técnicos en un archivo TXT
    directorio_src = os.path.dirname(os.path.abspath(__file__))
    ruta_reporte = os.path.join(directorio_src, "..", "data", "resultados_modelo.txt")
    
    with open(ruta_reporte, "w", encoding="utf-8") as f:
        f.write("REPORTE DE ENTRENAMIENTO\n")
        f.write("-" * 25 + "\n")
        f.write(f"Métrica MAE: {mae:.2f}\n")
        f.write(f"Métrica R2: {r2:.2f}\n")
        f.write("\nImportancia de variables (Coeficientes):\n")
        for col, coef in zip(X_train.columns, modelo.coef_):
            f.write(f"{col}: {coef:.2f}\n")

    print(f"\n Reporte técnico guardado en: {ruta_reporte}")
    return mae, r2

def guardar_modelo(modelo):
    directorio_src = os.path.dirname(os.path.abspath(__file__))
    carpeta_data = os.path.join(directorio_src, "..", "data")
    
    if not os.path.exists(carpeta_data):
        os.makedirs(carpeta_data)
    
    ruta_final = os.path.join(carpeta_data, "modelo_salarios.pkl")
    joblib.dump(modelo, ruta_final)
    
    print("-" * 30)
    print(f"¡ÉXITO! Modelo guardado en: {ruta_final}")
    print("-" * 30)

def cargar_modelo():
    directorio_src = os.path.dirname(os.path.abspath(__file__))
    ruta_archivo = os.path.join(directorio_src, "..", "data", "modelo_salarios.pkl")
    if not os.path.exists(ruta_archivo):
        raise FileNotFoundError(f"No existe el archivo: {ruta_archivo}")
    return joblib.load(ruta_archivo)

def predecir_salario(modelo):
    print("\n--- SIMULADOR DE PREDICCIÓN ---")
    try:
        edad = int(input("Introduce la Edad: "))
        depto = input("Departamento (IT, Ventas, Marketing): ").strip().capitalize()

        nuevo = {
            "edad": edad,
            "departamento_IT": 0,
            "departamento_Marketing": 0,
            "departamento_Ventas": 0
        }

        col_buscada = f"departamento_{depto}"
        if col_buscada in nuevo:
            nuevo[col_buscada] = 1
        else:
            print(f" El departamento '{depto}' no se reconoce, se usará base 0.")

        nuevo_df = pd.DataFrame([nuevo])
        pred = modelo.predict(nuevo_df)
        print(f" Salario estimado para {depto}: ${pred[0]:.2f}")
    except ValueError:
        print(" Error: Por favor introduce un número válido para la edad.")

# --- 2. MOTOR PRINCIPAL ---

def main():
    # 1. Cargar y preparar
    df = cargar_datos()
    X_train, X_test, y_train, y_test = preparar_datos(df)

    # 2. Entrenar y Evaluar (con los nuevos reportes)
    modelo_entrenado = entrenar_modelo(X_train, y_train)
    evaluar_y_reportar(modelo_entrenado, X_test, y_test, X_train)

    # 3. Guardar
    guardar_modelo(modelo_entrenado)

    # 4. Prueba de carga y predicción local
    modelo_cargado = cargar_modelo()
    predecir_salario(modelo_cargado)

if __name__ == "__main__":
    main()