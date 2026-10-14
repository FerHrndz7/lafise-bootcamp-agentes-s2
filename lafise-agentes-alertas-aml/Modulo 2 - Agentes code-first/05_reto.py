# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Reto — El punto ciego de las listas de sanciones
# MAGIC
# MAGIC **Contexto:** la alerta `AT-003` parece moderada (USD 25,900, una transferencia a Islas Caimán). Pero el
# MAGIC cliente **coincide con la lista OFAC**. La política (Artículo 9) y el Instructivo (regla dura 4) son claros:
# MAGIC *un hit confirmado en lista de sanciones nunca se cierra sin acción: se bloquea y se escala de inmediato.*
# MAGIC
# MAGIC Nuestro agente **no tiene cómo saberlo**: ninguna de sus herramientas consulta listas de sanciones.
# MAGIC
# MAGIC **Tu reto (abierto, ⏱️ ~10 min o para casa):**
# MAGIC 1. Crea la función de UC `verificar_sanciones(nombre)` sobre la tabla `lista_sanciones`.
# MAGIC 2. Agrégala como herramienta y añade la **regla dura** a las instrucciones.
# MAGIC 3. Dictamina `AT-003`: el agente debe detectar el hit, **escalar** y citar el fundamento.
# MAGIC 4. *(Bonus)* Dictamina las tres alertas testigo y guarda los resultados en una tabla → la usaremos en la
# MAGIC    **Sesión 4** para evaluar el agente.

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

from databricks.sdk import WorkspaceClient
w = WorkspaceClient()

# COMMAND ----------

# MAGIC %run ./utils/agente_utils

# COMMAND ----------

import mlflow
from databricks_langchain import ChatDatabricks

LLM = elegir_llm(w, dbutils.widgets.get("llm").strip())
llm = ChatDatabricks(endpoint=LLM, temperature=0)
mlflow.set_experiment(f"/Users/{user_email}/lafise_s2_agente_alertas")
mlflow.langchain.autolog()
display(spark.table(f"{FQ}.lista_sanciones"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 1 — La función `verificar_sanciones`
# MAGIC Debe recibir un `nombre` y devolver las filas de `lista_sanciones` que coincidan **exacta o
# MAGIC aproximadamente** (los nombres en listas rara vez vienen escritos igual). Pistas: `levenshtein(a, b)`,
# MAGIC `lower(...)`, `LIKE`. No olvides los `COMMENT`: el LLM los lee.

# COMMAND ----------

# TODO: escribe el cuerpo de la función (RETURN SELECT ...). La firma y los COMMENT ya están.
spark.sql(f"""
CREATE OR REPLACE FUNCTION {FQ}.verificar_sanciones(
  nombre STRING COMMENT 'Nombre completo de la persona o empresa a verificar'
)
RETURNS TABLE (nombre_listado STRING, lista STRING, pais STRING, fecha_inclusion STRING, motivo STRING, distancia INT)
COMMENT 'Coteja un nombre contra las listas de sanciones (OFAC, ONU y listas locales). Devuelve las coincidencias exactas o aproximadas; una coincidencia obliga a bloquear la operación y escalar de inmediato.'
RETURN
  <FILL_IN>   -- SELECT ... FROM {FQ}.lista_sanciones WHERE <coincidencia exacta o aproximada con verificar_sanciones.nombre>
""")
display(spark.sql(f"SELECT * FROM {FQ}.verificar_sanciones('Ramiro Quintanilla Rojas')"))

# COMMAND ----------

assert spark.sql(f"SELECT * FROM {FQ}.verificar_sanciones('Ramiro Quintanila Rojas')").count() >= 1, \
    "Debe tolerar errores de escritura (Quintanila con una sola l)"
assert spark.sql(f"SELECT * FROM {FQ}.verificar_sanciones('Lucía Fernanda Castillo Mena')").count() == 0
print("✅ verificar_sanciones detecta coincidencias aproximadas y no da falsos positivos obvios")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 2 — Herramienta + regla dura
# MAGIC Parte de `INSTRUCCIONES_AGENTE_ALERTAS` (las del notebook 02) y agrégale (a) el paso de verificar
# MAGIC sanciones y (b) la regla dura.

# COMMAND ----------

FUNCIONES = [f"{CATALOG}.{SCHEMA}.{f}" for f in ("obtener_alerta", "features_transaccionales", "verificar_sanciones")]
herramientas_fn, modo = herramientas_uc(w, FUNCIONES, probar=(FUNCIONES[0], {"alerta_id": "AT-001"}))

# TODO: construye INSTRUCCIONES_RETO a partir de INSTRUCCIONES_AGENTE_ALERTAS:
#   (a) agrega al procedimiento un paso "usa verificar_sanciones con el nombre del cliente"
#   (b) agrega una regla dura: ante una coincidencia, nunca cerrar; escalar, bloquear y citar el Artículo 9
# Pista: print(INSTRUCCIONES_AGENTE_ALERTAS) y usa .replace(...)
INSTRUCCIONES_RETO = <FILL_IN>
agente_reto = crear_agente(llm, herramientas_fn + [herramienta_ai_search(INDEX_NAME, num_results=3)], INSTRUCCIONES_RETO)
print("✅ Agente con verificación de sanciones listo")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 3 — Dictaminar `AT-003`

# COMMAND ----------

texto, mensajes = invocar(agente_reto, "Analiza la alerta AT-003 y emite tu dictamen.")
usadas = resumen_traza(mensajes)
dictamen = extraer_dictamen(texto)
print(json.dumps(dictamen, ensure_ascii=False, indent=2) if dictamen else texto)

# COMMAND ----------

assert any(u.endswith("verificar_sanciones") for u in usadas), f"El agente no verificó sanciones: {usadas}"
assert dictamen and dictamen.get("disposicion") == "ESCALAR_INVESTIGACION", dictamen
assert re.search(r"sanci|OFAC", json.dumps(dictamen, ensure_ascii=False), re.I), "El dictamen debe mencionar el hit"
print("✅ Reto completo: el agente detectó el hit en OFAC y escaló (regla dura respetada)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bonus — Dictaminar las tres alertas testigo y guardar los resultados
# MAGIC Esta tabla es el primer **set de evaluación** del agente: en la Sesión 4 la compararemos contra la
# MAGIC disposición esperada con jueces de MLflow.

# COMMAND ----------

ESPERADO = {"AT-001": "REPORTAR_ROS|ESCALAR_INVESTIGACION", "AT-002": "CERRAR_SIN_ACCION|SOLICITAR_INFORMACION",
            "AT-003": "ESCALAR_INVESTIGACION"}
filas = []
for alerta_id, esperado in ESPERADO.items():
    t, _ = invocar(agente_reto, f"Analiza la alerta {alerta_id} y emite tu dictamen.")
    d = extraer_dictamen(t) or {}
    filas.append((alerta_id, esperado, d.get("disposicion"), d.get("disposicion") in esperado.split("|"),
                  json.dumps(d, ensure_ascii=False)))
df = spark.createDataFrame(filas, ["alerta_id", "disposicion_esperada", "disposicion_agente", "coincide", "dictamen_json"])
df.write.mode("overwrite").saveAsTable(f"{FQ}.dictamenes_agente")
display(df)
print(f"Aciertos: {sum(f[3] for f in filas)}/{len(filas)}  →  tabla {CATALOG}.{SCHEMA}.dictamenes_agente")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Enfoques alternativos
# MAGIC - **Especialista de sanciones** (patrón del notebook 03): un sub-agente con `verificar_sanciones` que el
# MAGIC   supervisor consulta siempre; es la base del *Agente Maestro #9 Sanciones*, reutilizable por otros supervisores.
# MAGIC - **Regla dura en código, no en el prompt**: después del agente, un paso determinista que, si
# MAGIC   `verificar_sanciones` devolvió filas, fuerza `ESCALAR_INVESTIGACION`. Para controles regulatorios
# MAGIC   críticos conviene **las dos cosas**: el prompt guía, el código garantiza.
# MAGIC - En producción el cotejo real de sanciones vendría de un servicio interno o de un proveedor, expuesto vía
# MAGIC   **MCP** gobernado por Unity Gateway (Sesión 5).
# MAGIC
# MAGIC ## Cierre de la Sesión 2
# MAGIC Construiste un agente **code-first** completo: herramientas gobernadas en UC, razonamiento con LangGraph,
# MAGIC orquestación con sub-agentes, trazas en MLflow y un artefacto versionado en UC.
# MAGIC **Sesión 3:** el mismo objetivo por el camino **low-code** con **Agent Bricks** (Knowledge Assistant,
# MAGIC Supervisor Agent) — y comparamos esfuerzo, control y resultado.
