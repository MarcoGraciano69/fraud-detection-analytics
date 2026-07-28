# Detección de Fraude en Transacciones Financieras

**Autor:** Marco Iván Rodríguez Graciano
**Ocupación:** Ingeniero Físico | Data Analyst & Backend Developer (.NET/Python) — experiencia en banca, riesgo crediticio y migración de sistemas core bancarios
**Contacto:** macovanrodriguezgraciano@gmail.com

> ⚠️ **Estado del proyecto: en construcción.** Este README refleja el avance actual (limpieza, feature engineering y EDA completos). Las secciones de modelado, interpretabilidad, base de datos, API y BI se irán agregando conforme se completen.

---

## Objetivo del proyecto

Construir un sistema de detección de fraude transaccional que demuestre, de forma honesta y documentada, el ciclo completo de trabajo de un Analista de Datos / Data Scientist en un entorno financiero: desde la limpieza y preparación de datos reales, pasando por análisis exploratorio orientado a negocio, hasta el modelado predictivo y su posible exposición como servicio.

## Dataset

**IEEE-CIS Fraud Detection** (Kaggle, competencia patrocinada por Vesta Corporation)
- ~590,540 transacciones reales anonimizadas
- Dos tablas (`transaction` + `identity`) unidas por `TransactionID`
- Variable target: `isFraud` (binaria, fuertemente desbalanceada: 3.5% fraude)
- Los archivos CSV no se incluyen en este repositorio por su tamaño (1.35 GB); ver `data/README.md` para instrucciones de descarga desde Kaggle.

## Stack técnico (fase actual)

- Python (Google Colab)
- pandas, numpy
- scikit-learn (StandardScaler, LabelEncoder, train_test_split)

## Metodología y avance actual

### 1. Preparación de datos
- Merge de `train_transaction` + `train_identity` (left join, preservando las 590,540 filas)
- Optimización de memoria (float64→float32, int64→int32) para evitar saturación de RAM

### 2. Limpieza de nulos — basada en evidencia, no en reglas genéricas
El hallazgo más relevante de esta fase: **la ausencia de ciertos datos (no su valor) es en sí misma una señal predictiva de fraude.**

- Se comparó la tasa de fraude entre transacciones con y sin cada dato (ej. información de identidad/dispositivo), filtrando por tamaño de muestra mínimo (≥1,000 filas por grupo) para descartar diferencias no confiables (un hallazgo inicial con muestra de solo 12 filas fue correctamente descartado como ruido estadístico).
- Resultado confiable: transacciones con información de identidad capturada muestran tasas de fraude 3-5 veces mayores que las que no la tienen — consistente con la hipótesis de que esta captura se activa ante transacciones ya marcadas como sospechosas por sistemas previos.
- Se crearon 19 variables binarias `has_X` para capturar esta señal explícitamente.
- Imputación diferenciada: valor bandera (-999) para columnas numéricas con señal confirmada, `'missing'` para categóricas, mediana para el resto sin evidencia de patrón. Columnas con más de 90% de nulos fueron eliminadas.

### 3. Análisis Exploratorio de Datos (EDA de negocio)
Preguntas respondidas con storytelling y validación estadística (revisando tamaño de muestra antes de interpretar cualquier diferencia porcentual):

- **Monto de transacción**: patrón no lineal — mayor fraude en montos muy bajos (posibles transacciones de prueba) y en el rango 250-1000, menor en 50-250 y en montos muy altos (posiblemente sujetos a controles adicionales)
- **Dispositivo**: mobile (10.2%) > desktop (6.5%) > sin dato (2.1%)
- **Hora del día**: pico marcado en la hora 7 (10.6%, ~3x el promedio general)
- **Dominio de correo**: proveedores gratuitos/instantáneos (Outlook, Hotmail, Gmail) muestran mayor fraude que proveedores ligados a contratos verificados (AT&T, Verizon, SBCGlobal)
- **Producto y tipo de tarjeta**: variación considerable por categoría, con crédito mostrando ~2.7x más fraude que débito

### 4. Feature Engineering final
- Extracción de `TransactionHour` a partir de `TransactionDT` (timedelta cíclico, sin fecha de referencia conocida)
- Codificación de variables categóricas: One-Hot Encoding para baja cardinalidad (≤5 categorías), Label Encoding para alta cardinalidad (`DeviceInfo`, `P_emaildomain`, `R_emaildomain`, entre otras)

### 5. Train/Test Split
- División 80/20 con `stratify` sobre el target, preservando la proporción de fraude (3.5%) en ambos conjuntos

## Próximos pasos

- [ ] Manejo de desbalance de clases (class_weight / SMOTE)
- [ ] Modelado: baseline (Regresión Logística) → Random Forest / XGBoost
- [ ] Evaluación con métricas de negocio (Precision-Recall AUC, Recall, matriz de confusión — no Accuracy)
- [ ] Interpretabilidad con SHAP
- [ ] Base de datos relacional (SQL Server / PostgreSQL)
- [ ] Dashboard en Power BI
- [ ] API de inferencia en ASP.NET Core / C# (`POST /predict`)

## Decisiones de arquitectura

Este proyecto NO incluye Data Warehouse (Snowflake/similar) ni Databricks. Con ~590K filas provenientes de una única fuente estática, ninguna de las dos tecnologías resuelve un problema real de escala o integración — agregarlas sería sobre-ingeniería sin beneficio funcional. El detalle de esta decisión se documentará en `docs/decisions.md` conforme avance el proyecto.

## Cómo correrlo

1. Clona este repositorio
2. Descarga `train_transaction.csv` y `train_identity.csv` desde [Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection/data) (ver `data/README.md`)
3. Abre el notebook en `notebooks/` en Google Colab o Jupyter
4. Ejecuta las celdas en orden

---

*Este proyecto forma parte de mi portafolio como Data Analyst / Backend Developer en transición hacia roles de Data Science aplicada a finanzas y prevención de fraude (AML).*
