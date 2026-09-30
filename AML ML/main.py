import os
import logging
from datetime import datetime

import pandas as pd
import xgboost as xgb
import pyodbc
from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fraude_api")

app = FastAPI(title="API de Detección de Fraude Transaccional")

# --- Configuración ---
# Umbral parametrizable: con scale_pos_weight=27.58 las probabilidades del modelo
# quedan descalibradas hacia arriba, así que 0.5 no tiene ningún significado especial.
# El umbral real se elegiría con la curva precision-recall según el costo de negocio
# (falso negativo = monto de la transacción perdido; falso positivo = fricción con el cliente).
UMBRAL_FRAUDE = float(os.getenv("UMBRAL_FRAUDE", "0.5"))

# Conexión a SQL Server vía variable de entorno, no hardcodeada.
SQL_CONN_STRING = os.getenv(
    "SQL_CONN_STRING",
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=MARCOLAPTOP\\SQLEXPRESS;"
    "DATABASE=FraudeDB;"
    "Trusted_Connection=yes;",
)

MODEL_PATH = os.getenv("MODEL_PATH", "modelo_fraude_demo.json")

# Nota: este es un modelo demo entrenado solo con las 5 variables más influyentes
# según SHAP, usado para probar que el contrato de la API (FastAPI -> modelo ->
# SQL Server -> respuesta) funciona de punta a punta. El modelo de producción
# sería el XGBoost completo (506 features, PR-AUC 0.674) entrenado en el notebook;
# servirlo requiere además serializar el scaler y los LabelEncoders y mantener
# el orden exacto de columnas, que es el siguiente paso pendiente.
FEATURES = ["TransactionAmt", "C14", "C13", "C1", "card6_credit"]

model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)


class Transaccion(BaseModel):
    TransactionAmt: float
    C14: float
    C13: float
    C1: float
    card6_credit: int  # 1 si es tarjeta de crédito, 0 si es débito


def guardar_prediccion(t: Transaccion, proba: float, es_fraude: bool) -> bool:
    """Intenta guardar la predicción en SQL Server. Devuelve True/False
    según el resultado, y deja registro del error en logs en vez de solo
    imprimirlo (que se pierde en producción)."""
    try:
        conn = pyodbc.connect(SQL_CONN_STRING, timeout=5)
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO PrediccionesFraude
               (transaction_amt, c14, c13, c1, card6_credit, probabilidad_fraude, es_fraude)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            t.TransactionAmt, t.C14, t.C13, t.C1, t.card6_credit, proba, es_fraude,
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        # No tronamos la respuesta al cliente por un fallo de persistencia:
        # la predicción ya se calculó y es lo que le importa al caller.
        # Pero sí queda registrado el error, no solo impreso a stdout.
        logger.error(f"No se pudo guardar en SQL Server: {e}")
        return False


@app.get("/")
def home():
    return {"status": "API de detección de fraude activa"}


@app.get("/model-info")
def model_info():
    return {
        "features": FEATURES,
        "umbral_fraude": UMBRAL_FRAUDE,
        "modelo": MODEL_PATH,
        "nota": (
            "Modelo demo de 5 variables para validar el contrato de la API. "
            "El modelo completo (506 features, PR-AUC 0.674) se entrenó en el "
            "notebook pero aún no está servido aquí."
        ),
    }


@app.post("/predict")
def predecir(t: Transaccion):
    # Se usa un DataFrame con los mismos nombres de columna que en entrenamiento.
    # xgboost guarda los feature_names del fit; pasarle un np.array sin nombres
    # puede generar warnings o, según la versión, un mismatch silencioso.
    datos = pd.DataFrame([[t.TransactionAmt, t.C14, t.C13, t.C1, t.card6_credit]], columns=FEATURES)

    proba = float(model.predict_proba(datos)[:, 1][0])
    es_fraude = proba > UMBRAL_FRAUDE

    guardado_ok = guardar_prediccion(t, proba, es_fraude)

    return {
        "probabilidad_fraude": round(proba, 4),
        "es_fraude": es_fraude,
        "umbral_usado": UMBRAL_FRAUDE,
        "guardado_en_bd": guardado_ok,
        "timestamp": datetime.now().isoformat(),
    }