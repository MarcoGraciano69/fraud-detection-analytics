# Detección de Fraude en Transacciones Financieras

**Autor:** Marco Iván Rodríguez Graciano
**Ocupación:** Ingeniero Físico | Data Analyst & Backend Developer (.NET/Python) — experiencia en banca, riesgo crediticio y migración de sistemas core bancarios
**Contacto:** macovanrodriguezgraciano@gmail.com

> **Estado del proyecto:** pipeline de datos, modelado, interpretabilidad y API de inferencia completos y funcionales. Base de datos relacional operativa. Dashboard en Power BI conectado a la fuente de datos, visualizaciones aún pendientes. Ver la sección "Estado real por componente" para el detalle honesto de qué está terminado y qué no.

---

## Objetivo del proyecto

Construir un sistema de detección de fraude transaccional que demuestre, de forma honesta y documentada, el ciclo completo de trabajo de un Analista de Datos / Data Scientist en un entorno financiero: desde la limpieza y preparación de datos reales, pasando por análisis exploratorio orientado a negocio, modelado predictivo, interpretabilidad, y su exposición como servicio con persistencia y monitoreo.

Es un proyecto de aprendizaje personal, no un producto de producción — las decisiones de alcance a lo largo del README reflejan eso.

## Dataset

**IEEE-CIS Fraud Detection** (Kaggle, competencia patrocinada por Vesta Corporation)
- ~590,540 transacciones reales anonimizadas
- Dos tablas (`transaction` + `identity`) unidas por `TransactionID`
- Variable target: `isFraud` (binaria, fuertemente desbalanceada: 3.5% fraude)
- Los archivos CSV no se incluyen en este repositorio por su tamaño (1.35 GB) y porque los términos de la competencia de Kaggle no permiten su redistribución. Descárgalos directamente desde [Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection/data).

## Stack técnico

- **Modelado:** Python (Google Colab), pandas, numpy, scikit-learn (StandardScaler, LabelEncoder, train_test_split), XGBoost, SHAP
- **API:** Python, FastAPI, pyodbc
- **Base de datos:** SQL Server
- **BI:** Power BI Desktop (conectado vía SQL Server; conexión DirectQuery)

> Nota: el CV/portafolio original contemplaba la API en ASP.NET Core (C#). Se optó por FastAPI para evitar un salto de serialización del modelo entre lenguajes (vía ONNX u otro puente) y aprovechar que el modelo, el pipeline y la API viven en el mismo lenguaje. El resto de mi experiencia en .NET/C# está demostrado en otros proyectos de este portafolio.

## Metodología

### 1. Preparación de datos
- Merge de `train_transaction` + `train_identity` (left join, preservando las 590,540 filas)
- Optimización de memoria (downcast a `float32`/`int32`) para evitar saturación de RAM

### 2. Limpieza de nulos — basada en evidencia, no en reglas genéricas
El hallazgo más relevante de esta fase: **la ausencia de ciertos datos (no su valor) es en sí misma una señal predictiva de fraude.**

- Se comparó la tasa de fraude entre transacciones con y sin cada dato, filtrando por tamaño de muestra mínimo (≥1,000 filas por grupo) para descartar diferencias no confiables — un hallazgo inicial con una muestra de solo 12 filas fue correctamente identificado y descartado como ruido estadístico, no señal real.
- Resultado confiable: columnas como `D7`, `addr1`/`addr2` o la presencia de datos de identidad/dispositivo muestran tasas de fraude 3-5 veces mayores (o menores, según el caso) entre el grupo con dato y sin dato — consistente con la hipótesis de que la captura de ciertos datos se activa ante transacciones ya marcadas como sospechosas por sistemas previos.
- Se crearon 19 variables binarias `has_X` para capturar esta señal explícitamente, evitando depender de que la imputación posterior la preservara.
- Imputación diferenciada: valor centinela (-999 / `'missing'`) para columnas con señal confirmada, mediana para el resto sin evidencia de patrón. Columnas con más de 90% de nulos fueron eliminadas.
- **Nota de rigor conocida:** la imputación con mediana se calculó sobre el dataset completo antes del split train/test — leakage leve. El impacto esperado es marginal dado el tamaño de la muestra (590K filas), pero la forma correcta es encapsularlo en un `Pipeline` de scikit-learn ajustado solo sobre train. Queda como mejora pendiente.

### 3. Análisis Exploratorio de Datos (EDA de negocio)
Preguntas respondidas con storytelling y validación estadística (revisando tamaño de muestra antes de interpretar cualquier diferencia porcentual):

- **Monto de transacción**: patrón no lineal — mayor fraude en montos muy bajos (posibles transacciones de prueba) y en el rango 250-1000, menor en 50-250 y en montos muy altos (posiblemente sujetos a controles adicionales)
- **Dispositivo**: mobile (10.2%) > desktop (6.5%) > sin dato (2.1%)
- **Hora del día**: pico marcado en la hora 7 (10.6%, ~3x el promedio general)
- **Dominio de correo**: proveedores gratuitos/instantáneos (Outlook, Hotmail, Gmail) muestran mayor fraude que proveedores ligados a contratos verificados (AT&T, Verizon, SBCGlobal)
- **Producto y tipo de tarjeta**: variación considerable por categoría, con crédito mostrando ~2.7x más fraude que débito

### 4. Feature Engineering
- Extracción de `TransactionHour` a partir de `TransactionDT` (timedelta cíclico, sin fecha de referencia conocida)
- Codificación de variables categóricas: One-Hot Encoding para baja cardinalidad (≤5 categorías), Label Encoding para alta cardinalidad (`DeviceInfo`, `P_emaildomain`, `R_emaildomain`, entre otras), con manejo explícito de categorías no vistas en test

### 5. Train/Test Split
- Split realizado **antes** de codificar y escalar, para evitar leakage por ese lado
- División 80/20 con `stratify` sobre el target, preservando la proporción de fraude (3.5%) en ambos conjuntos
- Nota: el split es aleatorio, no temporal. En un escenario real de producción, dado que el fraude tiene deriva en el tiempo, se preferiría una validación con split temporal.

### 6. Modelado
- **Baseline:** Regresión Logística con `class_weight='balanced'` — PR-AUC 0.445
- **Modelo final:** XGBoost — PR-AUC 0.674, Recall 0.82
- Evaluación con métricas orientadas al desbalance de clases (PR-AUC, Recall, matriz de confusión, classification report) en lugar de Accuracy, dado que el fraude representa solo 3.5% de las transacciones

### 7. Interpretabilidad
- SHAP con `TreeExplainer` sobre el modelo XGBoost final
- Identificación de las 5 variables más influyentes según SHAP, usadas para construir un modelo reducido de demostración para la API (ver siguiente sección)

## API de inferencia (FastAPI)

- Endpoint `POST /predict`: recibe una transacción con las 5 variables top de SHAP (`TransactionAmt`, `C14`, `C13`, `C1`, `card6_credit`), devuelve probabilidad de fraude, clasificación binaria según umbral, y si la predicción se persistió correctamente en SQL Server
- Endpoint `GET /model-info`: expone qué modelo está sirviendo la API y por qué
- El modelo servido en la API es una versión reducida (5 features, PR-AUC ~0.454) entrenada específicamente para validar el contrato de la API de punta a punta (FastAPI → modelo → persistencia en SQL Server → respuesta) antes de invertir tiempo en servir el modelo completo de 506 features, que requiere además serializar el `scaler` y los `LabelEncoder` y mantener el orden exacto de columnas — ese es el siguiente paso pendiente, no un límite del modelo real.
- Umbral de clasificación parametrizable vía variable de entorno (no fijo en 0.5): con `scale_pos_weight` alto, las probabilidades del modelo quedan descalibradas, así que el umbral se elegiría en la práctica con la curva precision-recall según el costo de negocio (falso negativo = monto de la transacción perdido; falso positivo = fricción con el cliente)
- Conexión a SQL Server vía variable de entorno, no hardcodeada
- Manejo de errores en la persistencia: si el insert a SQL falla, queda registrado en logs y la respuesta al cliente lo indica explícitamente, sin tronar el endpoint

## Base de datos

- SQL Server, tabla `PrediccionesFraude` con cada predicción servida por la API: features de entrada, probabilidad, clasificación y timestamp
- Poblada tanto con pruebas manuales como con un script de generación de datos sintéticos (`generar_datos_prueba.py`) que simula transacciones con distintos perfiles de riesgo (bajo/medio/alto), para tener variedad representativa antes de construir visualizaciones

## Dashboard (Power BI)

- Conexión establecida de Power BI Desktop a SQL Server (DirectQuery) sobre la tabla `PrediccionesFraude`
- **Visualizaciones aún no construidas.** El diseño planeado combina dos tipos de métrica que no deben mezclarse sin aclaración: (1) métricas de evaluación del modelo, estáticas, calculadas una sola vez sobre el test set (PR-AUC, Recall); y (2) métricas operacionales en tiempo real de las predicciones que llegan por la API (volumen, distribución de probabilidad, % marcado como fraude, tendencia en el tiempo).

## Estado real por componente

| Componente | Estado |
|---|---|
| EDA y feature engineering | Completo |
| Modelo baseline (Regresión Logística) | Completo |
| Modelo final (XGBoost) | Completo |
| Interpretabilidad (SHAP) | Completo |
| API de inferencia (FastAPI) | Completo, sirviendo modelo demo de 5 features |
| Persistencia (SQL Server) | Completo |
| Dashboard (Power BI) | Conectado, visualizaciones pendientes |
| API sirviendo el modelo completo (506 features) | Pendiente |

## Decisiones de arquitectura

Este proyecto NO incluye Data Warehouse (Snowflake/similar) ni Databricks. Con ~590K filas provenientes de una única fuente estática, ninguna de las dos tecnologías resuelve un problema real de escala o integración — agregarlas sería sobre-ingeniería sin beneficio funcional.

## Próximos pasos

- [ ] Construir las visualizaciones del dashboard en Power BI
- [ ] Servir el modelo completo (506 features) en la API, serializando scaler y encoders
- [ ] Curva precision-recall para justificar el umbral de clasificación con datos, no solo con argumento cualitativo
- [ ] Encapsular la imputación de nulos en un `Pipeline` de scikit-learn ajustado solo sobre train
- [ ] Evaluar split temporal en lugar de aleatorio, dado que el fraude tiene deriva en el tiempo

## Cómo correrlo

1. Clona este repositorio
2. Descarga `train_transaction.csv` y `train_identity.csv` desde [Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection/data) y colócalos en `data/`
3. Abre el notebook en `notebooks/` en Google Colab o Jupyter y ejecuta las celdas en orden
4. Para la API: dentro de `api/`, instala dependencias (`pip install fastapi uvicorn xgboost pandas pyodbc requests`) y corre `uvicorn main:app --reload`
5. (Opcional) Para poblar la base con datos de prueba variados: `python generar_datos_prueba.py`, con la API corriendo en paralelo

---

*Este proyecto forma parte de mi portafolio como Data Analyst / Backend Developer en transición hacia roles de Data Science aplicada a finanzas y prevención de fraude (AML).*
