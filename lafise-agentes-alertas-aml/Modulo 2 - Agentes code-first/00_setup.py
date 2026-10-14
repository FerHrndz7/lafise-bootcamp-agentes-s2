# Databricks notebook source
# MAGIC %md
# MAGIC # Módulo 2 · Preparación — Agentes code-first sobre Databricks (Agente de Alertas AML)
# MAGIC
# MAGIC | | |
# MAGIC |---|---|
# MAGIC | **Duración** | ~105 min de práctica (esta preparación: ~10 min) |
# MAGIC | **Nivel** | Intermedio |
# MAGIC | **Lenguaje** | Python / SQL |
# MAGIC | **Cómputo** | Clúster **clásico** (DBR **17.3 LTS** recomendado, modo de acceso *Standard*) |
# MAGIC | **Catalog.Schema** | `<catálogo_asignado>.pld_aml_<tu_usuario>` (el mismo de la Sesión 1) |
# MAGIC
# MAGIC ## Qué construirás hoy 🕵️
# MAGIC En la Sesión 1 construimos la **base de conocimiento** (políticas, tipologías y precedentes en
# MAGIC **AI Search**, antes *Vector Search*). Hoy construimos el **Agente de Alertas** que la usa: recibe una alerta
# MAGIC del monitoreo transaccional, **reúne contexto con herramientas**, **razona** y **recomienda una
# MAGIC disposición** (cerrar / escalar / reportar / pedir información) **citando** el fundamento.
# MAGIC
# MAGIC Recorrido de la sesión:
# MAGIC 1. `00_setup` — dependencias, configuración y verificación de la base de la Sesión 1 (**estás aquí**).
# MAGIC 2. `01_herramientas_uc` — herramientas gobernadas: **funciones de Unity Catalog** + retriever de AI Search.
# MAGIC 3. `02_agente_de_alertas` — el bucle de razonamiento con **LangGraph** + reglas de decisión + MLflow Tracing.
# MAGIC 4. `03_supervisor_y_subagentes` — orquestación: un **supervisor** que delega en sub-agentes especialistas.
# MAGIC 5. `04_empaquetar_y_desplegar` — `ResponsesAgent` de MLflow, registro en UC y caminos de despliegue.
# MAGIC 6. `05_reto` — **tarea** para después de la sesión: añade la herramienta de **listas de sanciones** y respeta la regla dura.
# MAGIC
# MAGIC > ⚠️ **Datos sintéticos.** Todo es inventado (Faker + hechos regulatorios plausibles pero ficticios).
# MAGIC > Nunca uses PII real de clientes en un workspace de laboratorio.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sección 1 — Dependencias
# MAGIC Lo ideal es que el clúster ya tenga instaladas las librerías de `requirements.txt` (librerías de clúster).
# MAGIC Esta celda las asegura para **este** notebook (si ya están, termina rápido). Cada notebook del módulo
# MAGIC trae la misma celda al inicio.

# COMMAND ----------

# MAGIC %pip install --quiet "databricks-langchain>=0.20.0" "langgraph>=1.2,<1.3" "unitycatalog-langchain[databricks]>=0.3.0" "mlflow[databricks]>=3.17" "databricks-agents>=1.12" "databricks-sdk>=0.94.0" "faker>=25.0.0"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sección 2 — Configuración (aislamiento por participante)
# MAGIC
# MAGIC Escribe en el widget **catalog** el **mismo catálogo que usaste en la Sesión 1**. Tu schema lleva tu
# MAGIC usuario automáticamente (`pld_aml_<tu_usuario>`), así que nunca colisionas con otros participantes.
# MAGIC
# MAGIC El widget **llm** es opcional: si lo dejas vacío, elegimos el mejor modelo de chat con *tool calling*
# MAGIC disponible en el workspace.

# COMMAND ----------

import re

dbutils.widgets.text("catalog", "", "Catálogo asignado (REQUERIDO)")
dbutils.widgets.text("schema", "pld_aml", "Schema base")
dbutils.widgets.text("llm", "", "Endpoint LLM (opcional)")

user_email = spark.sql("SELECT current_user()").collect()[0][0]
user_prefix = re.sub(r"[^a-z0-9]", "_", user_email.split("@")[0].lower())
CATALOG = dbutils.widgets.get("catalog").strip()
assert CATALOG, ("Escribe en el widget 'catalog' el catálogo que te asignó el facilitador "
                 "(el mismo de la Sesión 1).")
SCHEMA = f"{(dbutils.widgets.get('schema').strip() or 'pld_aml')}_{user_prefix}"
INDEX_NAME = f"{CATALOG}.{SCHEMA}.chunks_index"

print(f"Usuario:  {user_email}")
print(f"Destino:  {CATALOG}.{SCHEMA}")
print(f"Índice:   {INDEX_NAME}  (AI Search, de la Sesión 1)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sección 3 — Verificación del entorno

# COMMAND ----------

import sys
from importlib.metadata import version
from databricks.sdk import WorkspaceClient

print(f"Python: {sys.version.split()[0]}")
assert sys.version_info >= (3, 10), "Se requiere Python 3.10+"
for p in ["databricks-sdk", "mlflow", "databricks-langchain", "langgraph", "unitycatalog-ai", "databricks-agents"]:
    try:
        print(f"  {p:22s} {version(p)}")
    except Exception:
        print(f"  {p:22s} ⚠️ no instalado")

try:
    print(f"DBR del clúster: {spark.conf.get('spark.databricks.clusterUsageTags.sparkVersion')}")
except Exception:
    print("DBR del clúster: no legible aquí (confirma 17.3 LTS en la configuración del clúster).")

w = WorkspaceClient()

# COMMAND ----------

# MAGIC %run ./utils/agente_utils

# COMMAND ----------

LLM = elegir_llm(w, dbutils.widgets.get("llm").strip())
print(f"Modelo de chat elegido: {LLM}")

from databricks_langchain import ChatDatabricks
r = ChatDatabricks(endpoint=LLM, temperature=0).invoke("Responde solo: listo")
print("Respuesta del modelo:", r.content)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sección 4 — Permisos
# MAGIC Hoy, además de tablas, crearás **funciones** (`CREATE FUNCTION`) y registrarás un **modelo**
# MAGIC (`CREATE MODEL`) en tu schema. Fallamos rápido si falta algún permiso.

# COMMAND ----------

try:
    spark.sql(f"USE CATALOG `{CATALOG}`")
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{CATALOG}`.{SCHEMA}")
    spark.sql(f"CREATE OR REPLACE FUNCTION `{CATALOG}`.{SCHEMA}._prueba_permiso() RETURNS STRING RETURN 'ok'")
    assert spark.sql(f"SELECT `{CATALOG}`.{SCHEMA}._prueba_permiso() AS r").collect()[0]["r"] == "ok"
    spark.sql(f"DROP FUNCTION IF EXISTS `{CATALOG}`.{SCHEMA}._prueba_permiso")
    print(f"✅ Puedes crear y ejecutar funciones en {CATALOG}.{SCHEMA}")
except Exception as e:
    raise PermissionError(
        f"No se pueden crear funciones en {CATALOG}.{SCHEMA}. Pide al administrador USE CATALOG + USE SCHEMA + "
        f"CREATE FUNCTION (y CREATE MODEL para el notebook 04). Error subyacente: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sección 5 — Base de conocimiento de la Sesión 1 + casos testigo
# MAGIC El bootstrap **reutiliza** lo que construiste en la Sesión 1 (tablas, `chunks`, índice de AI Search) y
# MAGIC solo regenera lo que falte. Además crea tres **alertas testigo** (`AT-001..003`) con un resultado esperado
# MAGIC conocido, y una **lista de sanciones** sintética para el reto.
# MAGIC
# MAGIC > Si tu índice de la Sesión 1 ya no existe, esta celda lo recrea y puede tardar **~10–15 min** (arranque del
# MAGIC > pipeline gestionado de AI Search). Mientras tanto puedes leer el notebook `01`.

# COMMAND ----------

# MAGIC %run ./data/bootstrap_base_conocimiento $catalog=$catalog $schema=$schema

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sección 6 — Resumen

# COMMAND ----------

print("=" * 64)
print("PREPARACIÓN COMPLETA ✅")
print("=" * 64)
print(f"Catalog.Schema : {CATALOG}.{SCHEMA}")
print(f"Índice         : {INDEX_NAME}")
print(f"Modelo de chat : {LLM}")
display(spark.sql(f"SELECT * FROM `{CATALOG}`.{SCHEMA}.alertas WHERE alerta_id LIKE 'AT-%' ORDER BY alerta_id"))
print("➡️  Continúa con 01_herramientas_uc")
