# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Orquestación: un supervisor y sus sub-agentes especialistas
# MAGIC
# MAGIC **Objetivo:** repartir el análisis de la alerta entre **especialistas** coordinados por un **supervisor**,
# MAGIC tal como trabaja un equipo de PLD. ⏱️ ~20 min.
# MAGIC
# MAGIC     [Supervisor de alertas] ── decide a quién consultar y redacta el dictamen
# MAGIC        ├─► consultar_analista_cliente      → sub-agente KYC          (herramienta: obtener_alerta)
# MAGIC        ├─► consultar_analista_transaccional → sub-agente transaccional (herramienta: features_transaccionales)
# MAGIC        └─► consultar_analista_normativo    → sub-agente normativo    (herramienta: buscar_normativa_pld)
# MAGIC
# MAGIC ### ¿Por qué sub-agentes?
# MAGIC - **Instrucciones más cortas y enfocadas** por especialista → menos confusión con muchas herramientas.
# MAGIC - **Responsabilidades separadas**: cada especialista se prueba, evalúa y mejora por separado.
# MAGIC - **Escala**: los 18 *Agentes Maestro* de LAFISE (KYC, Sanciones, Investigaciones…) pueden convertirse en
# MAGIC   especialistas reutilizables de distintos supervisores.
# MAGIC
# MAGIC Patrón usado: **"sub-agentes como herramientas"** — cada especialista es un agente completo envuelto en una
# MAGIC herramienta que el supervisor puede llamar. Es el patrón más simple y portable (funciona igual en LangGraph,
# MAGIC OpenAI Agents SDK o detrás de un endpoint desplegado).

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
INDEX_NAME = f"{CATALOG}.{SCHEMA}.chunks_index"
FUNCIONES = [f"{CATALOG}.{SCHEMA}.obtener_alerta", f"{CATALOG}.{SCHEMA}.features_transaccionales"]

from databricks.sdk import WorkspaceClient
w = WorkspaceClient()

# COMMAND ----------

# MAGIC %run ./utils/agente_utils

# COMMAND ----------

import time
import mlflow
from databricks_langchain import ChatDatabricks
from langchain_core.tools import tool

LLM = elegir_llm(w, dbutils.widgets.get("llm").strip())
llm = ChatDatabricks(endpoint=LLM, temperature=0)
mlflow.set_experiment(f"/Users/{user_email}/lafise_s2_agente_alertas")
mlflow.langchain.autolog()

herramientas_fn, modo = herramientas_uc(w, FUNCIONES, probar=(FUNCIONES[0], {"alerta_id": "AT-001"}))
herr = lambda sufijo: next(h for h in herramientas_fn if h.name.endswith(sufijo))
buscar = herramienta_ai_search(INDEX_NAME, num_results=3)
print(f"Modelo: {LLM} · funciones UC en modo {modo}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 1 — Los tres especialistas
# MAGIC Cada especialista es un agente completo (el mismo bucle del notebook 02, ya empaquetado en
# MAGIC `crear_agente(...)`), con **una sola herramienta** y unas instrucciones cortas. Devuelven **hallazgos**,
# MAGIC no disposiciones: decidir es trabajo del supervisor.

# COMMAND ----------

analista_cliente = crear_agente(llm, [herr("obtener_alerta")], """Eres el especialista KYC de la Unidad PLD.
Con obtener_alerta, resume en 3-5 viñetas la alerta y el perfil del cliente: tipología de la alerta, score, monto,
segmento, país, si es PEP y nivel de riesgo. Solo hechos; no recomiendes una disposición.""")

analista_transaccional = crear_agente(llm, [herr("features_transaccionales")], """Eres el especialista
transaccional de la Unidad PLD. Con features_transaccionales, interpreta las señales de los 30 días previos a la
alerta: % en efectivo, depósitos justo por debajo del umbral de USD 10,000, contrapartes de alto riesgo, montos.
Indica qué patrón sugieren (por ejemplo, estructuración). Cita las cifras exactas. No recomiendes una disposición.""")

analista_normativo = crear_agente(llm, [buscar], """Eres el especialista normativo de la Unidad PLD. Con
buscar_normativa_pld, encuentra el artículo de la política, la tipología y un precedente aplicables al patrón que
te describan. Devuelve cada fundamento con su cita exacta (documento + artículo/tipología/caso).""")
print("✅ Especialistas listos")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 2 — Envolver cada especialista como herramienta del supervisor
# MAGIC El **docstring** es la descripción que lee el supervisor para decidir a quién consultar: escríbelo como si
# MAGIC le explicaras a un colega qué hace cada especialista.

# COMMAND ----------

@tool
def consultar_analista_cliente(alerta_id: str) -> str:
    """Consulta al especialista KYC: devuelve los datos de la alerta y el perfil del cliente (PEP, riesgo, país)."""
    return invocar(analista_cliente, f"Analiza la alerta {alerta_id}.")[0]

# Paso 2 — mismo patrón que el especialista KYC.
@tool
def consultar_analista_transaccional(alerta_id: str) -> str:
    """Consulta al especialista transaccional: interpreta las señales de los 30 días previos a la alerta
    (efectivo, montos cerca del umbral de reporte, contrapartes de alto riesgo) y el patrón que sugieren."""
    return invocar(analista_transaccional, f"Analiza las transacciones de la alerta {alerta_id}.")[0]

@tool
def consultar_analista_normativo(descripcion_del_patron: str) -> str:
    """Consulta al especialista normativo: dado un patrón descrito en texto (p. ej. 'depósitos en efectivo
    fraccionados bajo el umbral'), devuelve la política, tipología y precedente aplicables con sus citas."""
    return invocar(analista_normativo, descripcion_del_patron)[0]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 3 — El supervisor
# MAGIC El supervisor no tiene acceso directo a los datos: **solo** puede consultar a los especialistas. Sus
# MAGIC instrucciones definen el **orden de trabajo** y reutilizan las mismas disposiciones, reglas y formato del
# MAGIC notebook 02.

# COMMAND ----------

INSTRUCCIONES_SUPERVISOR = """Eres el Supervisor de Alertas AML de la Unidad PLD. Coordinas a tres especialistas y
redactas el dictamen final para el analista humano.

Procedimiento:
1. Consulta al analista de cliente y al analista transaccional con el alerta_id.
2. Con lo que te digan, describe el patrón observado y consulta al analista normativo.
3. Integra los hallazgos y decide UNA disposición: CERRAR_SIN_ACCION, SOLICITAR_INFORMACION,
   ESCALAR_INVESTIGACION o REPORTAR_ROS (mismos criterios del Instructivo de Escalamiento).

Reglas duras: no inventes cifras; cita al menos un fundamento devuelto por el analista normativo; si el cliente es
PEP o hay contrapartes de alto riesgo, no cierres sin acción.

""" + FORMATO_DICTAMEN

# Paso 3 — el supervisor recibe como herramientas a los TRES especialistas.
supervisor = crear_agente(llm, [consultar_analista_cliente, consultar_analista_transaccional,
                                consultar_analista_normativo], INSTRUCCIONES_SUPERVISOR)
print("✅ Supervisor listo")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 4 — Dictaminar `AT-001` con el equipo de especialistas

# COMMAND ----------

t0 = time.time()
texto_sup, mensajes_sup = invocar(supervisor, "Analiza la alerta AT-001 y emite tu dictamen.")
t_sup = time.time() - t0
print("Especialistas consultados por el supervisor:")
consultados = resumen_traza(mensajes_sup)
dictamen_sup = extraer_dictamen(texto_sup)
print(json.dumps(dictamen_sup, ensure_ascii=False, indent=2) if dictamen_sup else texto_sup)

# COMMAND ----------

# Resultado esperado: el supervisor consulta a los 3 especialistas → REPORTAR_ROS (o ESCALAR_INVESTIGACION)
assert dictamen_sup and dictamen_sup.get("disposicion") in ("REPORTAR_ROS", "ESCALAR_INVESTIGACION"), dictamen_sup
assert {"consultar_analista_cliente", "consultar_analista_transaccional", "consultar_analista_normativo"} <= set(consultados), consultados
assert dictamen_sup.get("citas"), "El dictamen debe traer citas"
print(f"✅ Supervisor → {dictamen_sup['disposicion']} en {t_sup:.0f}s, consultando a {len(set(consultados))} especialistas")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 5 — ¿Agente único o supervisor? Compáralos
# MAGIC Corremos el agente único del notebook 02 sobre la misma alerta y comparamos tiempo y disposición.
# MAGIC Abre las dos trazas en MLflow: en la del supervisor verás las trazas de cada especialista **anidadas**.

# COMMAND ----------

agente_unico = crear_agente(llm, herramientas_fn + [buscar], INSTRUCCIONES_AGENTE_ALERTAS)
t0 = time.time()
texto_uni, _ = invocar(agente_unico, "Analiza la alerta AT-001 y emite tu dictamen.")
t_uni = time.time() - t0
d_uni = extraer_dictamen(texto_uni) or {}
display(spark.createDataFrame([
    ("Agente único (02)", d_uni.get("disposicion"), round(t_uni, 1), len(d_uni.get("citas", []))),
    ("Supervisor + 3 especialistas (03)", dictamen_sup.get("disposicion"), round(t_sup, 1), len(dictamen_sup.get("citas", []))),
], ["enfoque", "disposicion", "segundos", "citas"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ### ¿Cuándo usar cada uno?
# MAGIC | | Agente único | Supervisor + especialistas |
# MAGIC |---|---|---|
# MAGIC | Pocas herramientas (≤ 5), un dominio | ✅ más simple, más rápido y barato | — |
# MAGIC | Muchas herramientas / varios dominios | se confunde | ✅ cada especialista con su contexto |
# MAGIC | Reutilizar especialistas entre casos de uso | — | ✅ (p. ej. KYC o Sanciones para varios *Agentes Maestro*) |
# MAGIC | Costo y latencia | menor | mayor (más llamadas al LLM) |
# MAGIC
# MAGIC **Otras formas de orquestar en Databricks:**
# MAGIC - **Supervisor Agent de Agent Bricks** (low-code, Sesión 3): el mismo patrón configurado desde la UI,
# MAGIC   apuntando a funciones de UC, Knowledge Assistants, Genie Agents, MCP o endpoints.
# MAGIC - **Especialistas desplegados por separado** (cada uno con su endpoint) que el supervisor llama como herramientas.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resumen
# MAGIC - Construiste un **equipo de agentes**: tres especialistas con una herramienta cada uno + un supervisor.
# MAGIC - El supervisor delega, integra y dictamina con las mismas reglas y el mismo formato auditable.
# MAGIC - Comparaste **agente único vs. supervisor** en disposición, tiempo y citas.
# MAGIC
# MAGIC ➡️ En `04_empaquetar_y_desplegar` convertimos el agente en un **artefacto versionado** listo para producción.
