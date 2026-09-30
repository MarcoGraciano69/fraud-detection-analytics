"""
Genera predicciones variadas contra la API de fraude para poblar
PrediccionesFraude con datos realistas antes de conectar Power BI.

Requiere: pip install requests
Requiere que main.py (la FastAPI) ya esté corriendo en otra terminal:
    uvicorn main:app --reload
"""

import random
import time
import requests

API_URL = "http://127.0.0.1:8000/predict"
N_REGISTROS = 40  # ajusta si quieres más o menos

# Tres perfiles distintos para que el dashboard muestre variedad real,
# no solo ruido aleatorio uniforme.
PERFILES = [
    # (nombre, rango TransactionAmt, rango C14, rango C13, rango C1, prob card6_credit=1)
    ("bajo_riesgo",   (5, 60),      (1, 10),    (1, 15),    (1, 50),    0.3),
    ("medio",         (60, 400),    (5, 40),    (10, 80),   (50, 300),  0.5),
    ("alto_riesgo",   (400, 5000),  (30, 150),  (50, 300),  (200, 1500), 0.7),
]


def generar_transaccion():
    nombre, amt_r, c14_r, c13_r, c1_r, p_credito = random.choice(PERFILES)
    return {
        "TransactionAmt": round(random.uniform(*amt_r), 2),
        "C14": round(random.uniform(*c14_r), 2),
        "C13": round(random.uniform(*c13_r), 2),
        "C1": round(random.uniform(*c1_r), 2),
        "card6_credit": 1 if random.random() < p_credito else 0,
    }, nombre


def main():
    exitosos = 0
    fallidos = 0
    fraude_detectado = 0

    print(f"Enviando {N_REGISTROS} predicciones a {API_URL}...\n")

    for i in range(1, N_REGISTROS + 1):
        transaccion, perfil = generar_transaccion()
        try:
            resp = requests.post(API_URL, json=transaccion, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            exitosos += 1
            if data.get("es_fraude"):
                fraude_detectado += 1
            print(
                f"[{i:02d}/{N_REGISTROS}] perfil={perfil:<12} "
                f"monto={transaccion['TransactionAmt']:>8.2f}  "
                f"proba={data['probabilidad_fraude']:.4f}  "
                f"fraude={data['es_fraude']}"
            )
        except requests.exceptions.RequestException as e:
            fallidos += 1
            print(f"[{i:02d}/{N_REGISTROS}] ERROR: {e}")

        time.sleep(0.15)  # pequeña pausa, no satura la API ni la conexión a SQL

    print("\n--- Resumen ---")
    print(f"Exitosos: {exitosos}")
    print(f"Fallidos: {fallidos}")
    print(f"Marcados como fraude: {fraude_detectado} ({fraude_detectado / max(exitosos,1) * 100:.1f}%)")


if __name__ == "__main__":
    main()
