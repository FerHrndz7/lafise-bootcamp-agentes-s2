# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Herramientas gobernadas: funciones de Unity Catalog + AI Search
# MAGIC
# MAGIC **Objetivo:** darle al agente sus **herramientas** — lo que puede *hacer* además de "hablar". ⏱️ ~20 min.
# MAGIC
# MAGIC Al terminar tendrás tres herramientas listas para el agente:
# MAGIC
# MAGIC | Herramienta | Tipo | Para qué la usa el agente |
# MAGIC |---|---|---|
# MAGIC | `obtener_alerta` | Función de UC (SQL) | Leer la alerta + perfil KYC del cliente |
# MAGIC | `features_transaccionales` | Función de UC (SQL) | Calcular señales: efectivo, montos cerca del umbral, contrapartes de alto riesgo |
# MAGIC | `buscar_normativa_pld` | Retriever de **AI Search** | Recuperar y **citar** la política, tipología o precedente aplicable |
# MAGIC
# MAGIC ### ¿Por qué funciones de Unity Catalog?
# MAGIC - **Gobernadas:** solo quien tiene `EXECUTE` puede usarlas; quedan en el **linaje** y la **auditoría**.
# MAGIC - **Reutilizables:** la misma función la usará el **Supervisor Agent** de Agent Bricks (Sesión 3) y se puede
# MAGIC   exponer vía **MCP** gestionado (Sesión 5). Se escribe una vez, se gobierna en un solo lugar.
# MAGIC - **Deterministas:** el cálculo de señales lo hace SQL, no el LLM → el modelo **no inventa** los números.
# MAGIC - El **comentario** de la función y de cada parámetro es lo que el LLM lee para decidir cuándo usarla.

# COMMAND ----------

# MAGIC %pip install --quiet "databricks-langchain>=0.20.0" "langgraph>=1.2,<1.3" "unitycatalog-langchain[databricks]>=0.3.0" "mlflow[databricks]>=3.17" "databricks-sdk>=0.94.0"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import re

dbutils.widgets.text("catalog", "", "Catálogo asignado (REQUERIDO)")
dbutils.widgets.text("schema", "pld_aml", "Schema base")
dbutils.widgets.text("llm", "", "Endpoint LLM (opcional)")
user_email = spark.sql("SELECT current_user()").collect()[0][0]
user_prefix = re.sub(r"[^a-z0-9]", "_", user_email.split("@")[0].lower())
CATALOG = dbutils.widgets.get("catalog").strip()
assert CATALOG, "Escribe en el widget 'catalog' el mismo catálogo que usaste en 00_setup."
SCHEMA = f"{(dbutils.widgets.get('schema').strip() or 'pld_aml')}_{user_prefix}"
FQ = f"`{CATALOG}`.{SCHEMA}"
INDEX_NAME = f"{CATALOG}.{SCHEMA}.chunks_index"
spark.sql(f"USE CATALOG `{CATALOG}`")
spark.sql(f"USE SCHEMA {SCHEMA}")

from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
print(f"Trabajando en {CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %run ./utils/agente_utils

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 1 — Función `obtener_alerta` (ejemplo completo)
# MAGIC Una **función de tabla** SQL: recibe un `alerta_id` y devuelve la alerta + el perfil KYC del cliente.
# MAGIC Fíjate en los `COMMENT`: son la "documentación" que leerá el LLM.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {FQ}.obtener_alerta(
  alerta_id STRING COMMENT 'ID de la alerta de monitoreo, por ejemplo AT-001'
)
RETURNS TABLE (
  alerta_id STRING, fecha_alerta STRING, tipologia STRING, score_riesgo INT, monto_involucrado_usd DOUBLE,
  estado STRING, cliente_id STRING, nombre_cliente STRING, segmento STRING, pais STRING, es_pep BOOLEAN,
  nivel_riesgo STRING, cliente_desde STRING
)
COMMENT 'Devuelve la alerta de monitoreo transaccional y el perfil KYC del cliente (segmento, país, si es PEP, nivel de riesgo). Úsala primero para entender la alerta.'
RETURN
  SELECT a.alerta_id, a.fecha, a.tipologia, CAST(a.score_riesgo AS INT), a.monto_involucrado_usd, a.estado,
         c.cliente_id, c.nombre, c.segmento, c.pais, c.es_pep, c.nivel_riesgo, c.fecha_alta
  FROM {FQ}.alertas a JOIN {FQ}.clientes c ON a.cliente_id = c.cliente_id
  WHERE a.alerta_id = obtener_alerta.alerta_id
""")
display(spark.sql(f"SELECT * FROM {FQ}.obtener_alerta('AT-001')"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 2 — Función `features_transaccionales`
# MAGIC Calcula las **señales** de los 30 días previos a la alerta. Las reglas vienen de la política y del manual
# MAGIC de tipologías de la Sesión 1:
# MAGIC - `pct_efectivo` — proporción del monto movido en **efectivo** (0 a 1).
# MAGIC - `n_efectivo_cerca_umbral` — depósitos en efectivo **entre USD 9,000 y 9,999.99** (justo debajo del umbral
# MAGIC   de RTE de USD 10,000 → señal de **estructuración**).
# MAGIC - `n_contraparte_alto_riesgo` — transacciones con contraparte en jurisdicción de alto riesgo.

# COMMAND ----------

# Paso 2
# pct_efectivo = monto en efectivo / monto total (protegido contra división por cero con nullif).
# n_efectivo_cerca_umbral = conteo condicional (count_if) de efectivo en [9000, 10000).
spark.sql(f"""
CREATE OR REPLACE FUNCTION {FQ}.features_transaccionales(
  alerta_id STRING COMMENT 'ID de la alerta de monitoreo, por ejemplo AT-001'
)
RETURNS TABLE (
  alerta_id STRING, cliente_id STRING, ventana_dias INT, n_transacciones BIGINT, monto_total_usd DOUBLE,
  monto_efectivo_usd DOUBLE, pct_efectivo DOUBLE, n_efectivo_cerca_umbral BIGINT, dias_con_efectivo BIGINT,
  n_contraparte_alto_riesgo BIGINT, monto_contraparte_alto_riesgo_usd DOUBLE, txn_maxima_usd DOUBLE
)
COMMENT 'Calcula señales transaccionales del cliente en los 30 días previos a la alerta: montos, porcentaje en efectivo, depósitos en efectivo justo por debajo del umbral de USD 10,000 (posible estructuración) y exposición a contrapartes en jurisdicciones de alto riesgo.'
RETURN
  WITH a AS (
    SELECT alerta_id, cliente_id, CAST(fecha AS DATE) AS f FROM {FQ}.alertas
    WHERE alerta_id = features_transaccionales.alerta_id
  ),
  t AS (
    SELECT t.* FROM {FQ}.transacciones t JOIN a ON t.cliente_id = a.cliente_id
    WHERE CAST(t.fecha AS DATE) BETWEEN date_sub(a.f, 30) AND a.f
  )
  SELECT
    first(a.alerta_id), first(a.cliente_id), 30,
    count(t.txn_id),
    round(coalesce(sum(t.monto_usd), 0), 2),
    round(coalesce(sum(CASE WHEN t.canal = 'Efectivo' THEN t.monto_usd END), 0), 2),
    round(coalesce(sum(CASE WHEN t.canal = 'Efectivo' THEN t.monto_usd END), 0) / nullif(sum(t.monto_usd), 0), 3),
    count_if(t.canal = 'Efectivo' AND t.monto_usd >= 9000 AND t.monto_usd < 10000),
    count(DISTINCT CASE WHEN t.canal = 'Efectivo' THEN t.fecha END),
    count_if(t.pais_contraparte IN ('Islas Caimán')),
    round(coalesce(sum(CASE WHEN t.pais_contraparte IN ('Islas Caimán') THEN t.monto_usd END), 0), 2),
    round(coalesce(max(t.monto_usd), 0), 2)
  FROM a LEFT JOIN t ON t.cliente_id = a.cliente_id
""")
display(spark.sql(f"SELECT * FROM {FQ}.features_transaccionales('AT-001')"))

# COMMAND ----------

# Resultado esperado: AT-001 → n_transacciones=9, pct_efectivo=1.0, n_efectivo_cerca_umbral=9
f1 = spark.sql(f"SELECT * FROM {FQ}.features_transaccionales('AT-001')").collect()[0]
assert f1["n_efectivo_cerca_umbral"] >= 9, f"Esperaba ≥9 depósitos cerca del umbral, obtuve {f1['n_efectivo_cerca_umbral']}"
assert abs((f1["pct_efectivo"] or 0) - 1.0) < 0.01, f"AT-001 es 100% efectivo; obtuve {f1['pct_efectivo']}"
f3 = spark.sql(f"SELECT * FROM {FQ}.features_transaccionales('AT-003')").collect()[0]
assert f3["n_contraparte_alto_riesgo"] >= 1
print("✅ features_transaccionales calcula las señales correctas (AT-001: estructuración; AT-003: alto riesgo)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 3 — De función de UC a herramienta del agente
# MAGIC `herramientas_uc(...)` envuelve las funciones como herramientas de LangChain:
# MAGIC - Si el workspace tiene **serverless**, usa el `UCFunctionToolkit` oficial (`databricks-langchain` +
# MAGIC   Unity Catalog AI), que ejecuta las funciones en cómputo serverless.
# MAGIC - Si **no** (cómputo clásico únicamente), ejecuta la **misma función de UC** con `spark.sql` en tu clúster.
# MAGIC
# MAGIC Mira qué ve el LLM de cada herramienta: **nombre, descripción y argumentos** (salen de los `COMMENT`).
# MAGIC Usamos el nombre corto de la función (`obtener_alerta`) en ambos modos, para que las instrucciones del
# MAGIC agente lo nombren igual.

# COMMAND ----------

FUNCIONES = [f"{CATALOG}.{SCHEMA}.obtener_alerta", f"{CATALOG}.{SCHEMA}.features_transaccionales"]
herramientas, modo = herramientas_uc(w, FUNCIONES, probar=(FUNCIONES[0], {"alerta_id": "AT-001"}))
print(f"Modo de ejecución de funciones UC: {modo}\n")
for h in herramientas:
    print(f"🔧 {h.name}\n   {h.description[:150]}...\n   args: {list(h.args)}\n")

# COMMAND ----------

# Paso 3 — invocar la herramienta directamente (sin LLM) con sus argumentos.
herr_alerta = next(h for h in herramientas if h.name.endswith("obtener_alerta"))
salida = herr_alerta.invoke({"alerta_id": "AT-001"})
print(str(salida)[:600])

# COMMAND ----------

assert "AT-001" in str(salida) and "CT-001" in str(salida)
print("✅ La herramienta obtener_alerta funciona")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 4 — El retriever de AI Search como herramienta
# MAGIC `VectorSearchRetrieverTool` (de `databricks-langchain`) convierte tu índice de **AI Search** de la Sesión 1
# MAGIC en una herramienta. Usamos búsqueda **híbrida** (keyword + semántica), que en la Sesión 1 vimos que recupera
# MAGIC identificadores exactos como `K-2024-018`.
# MAGIC
# MAGIC > El producto se llama ahora **Databricks AI Search** (antes *Vector Search*); las clases y APIs del SDK
# MAGIC > conservan el nombre `vector_search`.

# COMMAND ----------

buscar = herramienta_ai_search(INDEX_NAME, num_results=3)
print(f"🔧 {buscar.name}\n   {buscar.description[:160]}...\n")

# Paso 4 — una consulta en lenguaje natural, como la haría el agente.
docs = buscar.invoke("depósitos en efectivo fraccionados por debajo del umbral de reporte")
# La herramienta devuelve los fragmentos como texto, con su metadata (source_doc, articulo) para citar.
print(str(docs)[:1500])

# COMMAND ----------

texto_docs = str(docs)
assert "10,000" in texto_docs or "structuración" in texto_docs, "El retriever no trajo la normativa esperada"
print("✅ El retriever recupera la normativa de estructuración / umbral de RTE")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 5 — Gobierno de las herramientas
# MAGIC Las funciones son objetos de UC: puedes ver su definición, su dueño y otorgar `EXECUTE` como a cualquier
# MAGIC tabla. Así el equipo de Cumplimiento controla **qué puede hacer** cada agente.

# COMMAND ----------

display(spark.sql(f"DESCRIBE FUNCTION EXTENDED {FQ}.features_transaccionales"))
# Ejemplo (no ejecutar en el lab): dar permiso a un grupo o al service principal de un agente
# spark.sql(f"GRANT EXECUTE ON FUNCTION {FQ}.features_transaccionales TO `analistas_pld`")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resumen
# MAGIC - Creaste **dos funciones de UC** gobernadas que calculan contexto y señales de forma determinista.
# MAGIC - Las convertiste en **herramientas** del agente (serverless o clásico, mismo resultado).
# MAGIC - Convertiste el índice de **AI Search** en una herramienta de retrieval **híbrido** y citable.
# MAGIC
# MAGIC ➡️ En `02_agente_de_alertas` el LLM decidirá **cuándo** usar cada herramienta para dictaminar una alerta.
