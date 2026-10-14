# Databricks notebook source
# MAGIC %md
# MAGIC # Validación de entorno — Workshop de Agentes AML (Sesión 2: agentes code-first)
# MAGIC
# MAGIC Ejecuta este notebook en el **workspace de desarrollo de LAFISE**, con un **clúster clásico** adjunto
# MAGIC (DBR **17.3 LTS**, modo de acceso *Standard*), para confirmar que el entorno soporta el laboratorio de la
# MAGIC Sesión 2.
# MAGIC
# MAGIC Igual que en la Sesión 1, no basta con *ver* los productos en la UI: este notebook **crea y usa** cada pieza
# MAGIC (instala librerías, crea una función de UC, llama al modelo con herramientas, consulta AI Search, corre un
# MAGIC mini-agente y registra un modelo en UC) y reporta **PASA / FALLA / ALTERNATIVA** por requisito. Al final
# MAGIC limpia lo que creó.
# MAGIC
# MAGIC ## Instrucciones
# MAGIC 1. Adjunta un **clúster clásico** DBR 17.3 LTS (el mismo tipo que usarán los asistentes).
# MAGIC 2. En el widget **catalog** escribe el catálogo donde trabajarán los asistentes (el de la Sesión 1).
# MAGIC 3. **Run all** (~5–10 min; hasta ~20 min si debe crear un índice de prueba de AI Search).
# MAGIC 4. Comparte la **tabla resumen final** con el facilitador.
# MAGIC
# MAGIC > Si la **primera celda** (`%pip install`) falla, el clúster **no tiene salida a PyPI**: instalen las
# MAGIC > librerías de `requirements.txt` como **librerías de clúster** (o configuren un mirror de PyPI) y repitan.

# COMMAND ----------

# MAGIC %pip install --quiet "databricks-langchain>=0.20.0" "langgraph>=1.2,<1.3" "unitycatalog-langchain[databricks]>=0.3.0" "mlflow[databricks]>=3.17" "databricks-agents>=1.12" "databricks-sdk>=0.94.0"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "", "Catálogo (requerido)")
dbutils.widgets.text("schema", "validacion_s2", "Schema de prueba")

import json, re, time
resultados = {}   # requisito -> (estado, detalle)

def marcar(req, estado, detalle=""):
    resultados[req] = (estado, str(detalle)[:300])
    print(f"[{estado}] {req} — {detalle}")

CATALOG = dbutils.widgets.get("catalog").strip()
SCHEMA = dbutils.widgets.get("schema").strip() or "validacion_s2"
assert CATALOG, "Escribe el catálogo en el widget 'catalog' antes de ejecutar."
FQ = f"`{CATALOG}`.{SCHEMA}"
usuario = spark.sql("SELECT current_user()").collect()[0][0]
user_prefix = re.sub(r"[^a-z0-9]", "_", usuario.split("@")[0].lower())

from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
marcar("0. Librerías instaladas desde PyPI", "PASA", "databricks-langchain, langgraph, unitycatalog-ai, mlflow, databricks-agents")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Cómputo clásico y versión de runtime

# COMMAND ----------

try:
    dbr = spark.conf.get("spark.databricks.clusterUsageTags.sparkVersion")
except Exception:
    dbr = None
marcar("1. Clúster clásico / DBR", "INFO", dbr or "no legible aquí; confirma DBR 17.3 LTS en la configuración del clúster")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Unity Catalog — crear y ejecutar una función (herramienta del agente) en el clúster clásico

# COMMAND ----------

FN = f"{CATALOG}.{SCHEMA}.fn_validacion"
try:
    spark.sql(f"USE CATALOG `{CATALOG}`")
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {FQ}")
    spark.createDataFrame([("AT-001", 9, 85320.0)], ["alerta_id", "n_txn", "monto"]) \
        .write.mode("overwrite").saveAsTable(f"{FQ}.alertas_val")
    spark.sql(f"""CREATE OR REPLACE FUNCTION {FQ}.fn_validacion(alerta_id STRING COMMENT 'ID de la alerta')
                  RETURNS TABLE (alerta_id STRING, n_txn BIGINT, monto DOUBLE)
                  COMMENT 'Función de prueba de la validación de entorno'
                  RETURN SELECT alerta_id, n_txn, monto FROM {FQ}.alertas_val
                         WHERE alerta_id = fn_validacion.alerta_id""")
    fila = spark.sql(f"SELECT * FROM {FQ}.fn_validacion(:a)", args={"a": "AT-001"}).collect()
    assert fila and fila[0]["n_txn"] == 9
    marcar("2. UC: CREATE FUNCTION + ejecución en clúster clásico", "PASA", f"{FN} OK")
except Exception as e:
    marcar("2. UC: CREATE FUNCTION + ejecución en clúster clásico", "FALLA", e)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Funciones UC vía `DatabricksFunctionClient` (ejecución serverless)
# MAGIC El cliente oficial de Unity Catalog AI ejecuta las funciones en **cómputo serverless**. Si no está
# MAGIC disponible, el laboratorio usa automáticamente la **ejecución en el clúster clásico** (requisito 2), así que
# MAGIC aquí `ALTERNATIVA` **no bloquea** el workshop. Para producción (agente desplegado) sí se necesita.

# COMMAND ----------

t0 = time.time()
try:
    from unitycatalog.ai.core.databricks import DatabricksFunctionClient
    r = DatabricksFunctionClient().execute_function(FN, {"alerta_id": "AT-001"})
    if r.error:
        raise RuntimeError(r.error)
    marcar("3. Funciones UC en serverless (DatabricksFunctionClient)", "PASA", f"en {time.time()-t0:.0f}s")
except Exception as e:
    marcar("3. Funciones UC en serverless (DatabricksFunctionClient)", "ALTERNATIVA",
           f"No disponible ({time.time()-t0:.0f}s) → el lab usa spark.sql en el clúster clásico. Detalle: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Foundation Model APIs — modelo de chat con *tool calling*

# COMMAND ----------

from databricks_langchain import ChatDatabricks
from langchain_core.tools import tool

PREFERIDOS = ["databricks-claude-sonnet-4-6", "databricks-claude-sonnet-4-5", "databricks-claude-sonnet-4",
              "databricks-meta-llama-3-3-70b-instruct", "databricks-llama-4-maverick",
              "databricks-qwen3-next-80b-a3b-instruct", "databricks-gemini-3-5-flash"]
LLM = None
try:
    servidos = {e.name for e in w.serving_endpoints.list()}
    LLM = next((m for m in PREFERIDOS if m in servidos), None)
    if not LLM:
        raise RuntimeError(f"Ninguno de {PREFERIDOS} está servido en el workspace")

    @tool
    def obtener_alerta(alerta_id: str) -> str:
        """Devuelve los datos de una alerta de monitoreo."""
        return "{}"

    resp = ChatDatabricks(endpoint=LLM, temperature=0).bind_tools([obtener_alerta]).invoke(
        "Usa la herramienta para obtener la alerta AT-001.")
    assert resp.tool_calls and resp.tool_calls[0]["name"] == "obtener_alerta", f"sin tool_calls: {resp}"
    marcar("4. FMAPI: chat con tool calling", "PASA", f"{LLM} pidió la herramienta correctamente")
except Exception as e:
    marcar("4. FMAPI: chat con tool calling", "FALLA", e)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. AI Search (antes Vector Search) como herramienta del agente
# MAGIC Si existe el índice de la Sesión 1 del usuario actual (`<catálogo>.pld_aml_<usuario>.chunks_index`), lo
# MAGIC consultamos. Si no, creamos un índice pequeño de prueba (tarda ~10–15 min) y lo borramos al final.

# COMMAND ----------

from databricks.sdk.service.vectorsearch import (
    DeltaSyncVectorIndexSpecRequest, EmbeddingSourceColumn, PipelineType, VectorIndexType, EndpointType)
INDICE_S1 = f"{CATALOG}.pld_aml_{user_prefix}.chunks_index"
IDX, EP, creado_idx, creado_ep = INDICE_S1, "lafise_vs_endpoint", False, False
try:
    try:
        w.vector_search_indexes.get_index(INDICE_S1)
        print(f"Usando el índice de la Sesión 1: {INDICE_S1}")
    except Exception:
        IDX = f"{CATALOG}.{SCHEMA}.idx_validacion"
        servidos = {e.name for e in w.serving_endpoints.list()}
        EMB = next(c for c in ["databricks-qwen3-embedding-0-6b", "databricks-gte-large-en", "databricks-bge-large-en"]
                   if c in servidos)
        spark.createDataFrame([("c1", "politica_pld_lafise.pdf", "Artículo 5", "Efectivo ≥ USD 10,000 genera RTE a la UAF."),
                               ("c2", "manual_tipologias.pdf", "Tipología 1", "Estructuración: fraccionar bajo USD 10,000.")],
                              ["chunk_id", "source_doc", "articulo", "chunk_to_retrieve"]) \
            .write.mode("overwrite").saveAsTable(f"{FQ}.chunks_val")
        spark.sql(f"ALTER TABLE {FQ}.chunks_val SET TBLPROPERTIES (delta.enableChangeDataFeed=true)")
        if EP not in {e.name for e in w.vector_search_endpoints.list_endpoints()}:
            print(f"Creando endpoint {EP} (5–15 min)...")
            w.vector_search_endpoints.create_endpoint_and_wait(name=EP, endpoint_type=EndpointType.STANDARD)
            creado_ep = True
        w.vector_search_indexes.create_index(
            name=IDX, endpoint_name=EP, primary_key="chunk_id", index_type=VectorIndexType.DELTA_SYNC,
            delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
                source_table=f"{CATALOG}.{SCHEMA}.chunks_val", pipeline_type=PipelineType.TRIGGERED,
                embedding_source_columns=[EmbeddingSourceColumn(name="chunk_to_retrieve",
                                                                embedding_model_endpoint_name=EMB)]))
        creado_idx = True
        limite = time.time() + 30 * 60
        while time.time() < limite:
            try:
                if w.vector_search_indexes.get_index(IDX).status.ready:
                    break
            except Exception:
                pass
            time.sleep(30)

    from databricks_langchain import VectorSearchRetrieverTool
    rt = VectorSearchRetrieverTool(index_name=IDX, tool_name="buscar_normativa_pld", num_results=2,
                                   query_type="HYBRID", columns=["source_doc", "articulo", "chunk_to_retrieve"],
                                   tool_description="Busca normativa PLD")
    salida = str(rt.invoke("umbral de efectivo para reportar"))
    assert "10,000" in salida, salida[:300]
    marcar("5. AI Search: retriever híbrido como herramienta", "PASA", f"{IDX} responde")
except Exception as e:
    marcar("5. AI Search: retriever híbrido como herramienta", "FALLA", e)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Mini-agente LangGraph de punta a punta (clúster clásico)
# MAGIC El mismo patrón del laboratorio: modelo ⇄ herramienta (la función UC del requisito 2 ejecutada con
# MAGIC `spark.sql`), con MLflow Tracing activo.

# COMMAND ----------

try:
    import mlflow
    from langchain_core.messages import AIMessage, SystemMessage
    from langgraph.graph import END, MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode
    from pyspark.sql import SparkSession
    _spark = SparkSession.getActiveSession()           # capturada en el hilo principal (ToolNode usa hilos)

    @tool
    def fn_validacion(alerta_id: str) -> str:
        """Devuelve el número de transacciones y el monto de una alerta."""
        filas = _spark.sql(f"SELECT * FROM {FQ}.fn_validacion(:a)", args={"a": alerta_id}).collect()
        return json.dumps([f.asDict() for f in filas])

    mlflow.set_experiment(f"/Users/{usuario}/validacion_entorno_s2")
    mlflow.langchain.autolog()
    llm = ChatDatabricks(endpoint=LLM, temperature=0).bind_tools([fn_validacion])
    g = StateGraph(MessagesState)
    g.add_node("modelo", lambda s: {"messages": [llm.invoke([SystemMessage(content="Responde usando la herramienta.")] + s["messages"])]})
    g.add_node("herramientas", ToolNode([fn_validacion]))
    g.set_entry_point("modelo")
    g.add_conditional_edges("modelo", lambda s: "herramientas" if isinstance(s["messages"][-1], AIMessage) and s["messages"][-1].tool_calls else END,
                            {"herramientas": "herramientas", END: END})
    g.add_edge("herramientas", "modelo")
    out = g.compile().invoke({"messages": [{"role": "user", "content": "¿Cuántas transacciones tiene la alerta AT-001?"}]})
    final = out["messages"][-1].content
    final = final if isinstance(final, str) else json.dumps(final)
    assert "9" in final, final[:200]
    marcar("6. Mini-agente LangGraph + MLflow Tracing", "PASA", final[:120])
except Exception as e:
    marcar("6. Mini-agente LangGraph + MLflow Tracing", "FALLA", e)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Registrar un modelo en Unity Catalog (`CREATE MODEL`)
# MAGIC En el notebook 04 cada asistente registra su agente como modelo de UC.

# COMMAND ----------

MODELO = f"{CATALOG}.{SCHEMA}.modelo_validacion"
try:
    import mlflow
    mlflow.set_registry_uri("databricks-uc")

    class Eco(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input):
            return model_input

    import pandas as pd
    with mlflow.start_run(run_name="validacion_s2"):
        info = mlflow.pyfunc.log_model(name="m", python_model=Eco(), registered_model_name=MODELO,
                                       input_example=pd.DataFrame({"x": [1]}))
    marcar("7. Registro de modelo en UC", "PASA", f"{MODELO} v{info.registered_model_version}")
except Exception as e:
    marcar("7. Registro de modelo en UC", "FALLA", e)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Despliegue (opcional para el workshop)
# MAGIC El laboratorio **no** despliega por defecto. Estos chequeos indican si los caminos de despliegue estarán
# MAGIC disponibles para producción o para la demo opcional:
# MAGIC - **Model Serving** (`agents.deploy()`): requiere permiso para crear endpoints de serving.
# MAGIC - **Databricks Apps / Agent Runtime** (camino recomendado con Agent Bricks CLI).

# COMMAND ----------

try:
    n = sum(1 for _ in w.serving_endpoints.list())
    marcar("8a. Model Serving (opcional)", "INFO", f"API accesible ({n} endpoints visibles). Crear endpoints requiere el "
                                                   "entitlement correspondiente; se prueba solo si se usa agents.deploy().")
except Exception as e:
    marcar("8a. Model Serving (opcional)", "INFO", f"No accesible: {e}")
try:
    n = sum(1 for _ in w.apps.list())
    marcar("8b. Databricks Apps / Agent Runtime (opcional)", "INFO", f"Apps habilitado ({n} apps visibles)")
except Exception as e:
    marcar("8b. Databricks Apps / Agent Runtime (opcional)", "INFO",
           f"Apps no disponible ({e}). No afecta al lab; sí al despliegue recomendado con Agent Bricks CLI.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Limpieza

# COMMAND ----------

if creado_idx:
    try:
        w.vector_search_indexes.delete_index(index_name=IDX)
    except Exception:
        pass
if creado_ep:
    try:
        w.vector_search_endpoints.delete_endpoint(EP)
    except Exception:
        pass
try:
    w.registered_models.delete(MODELO)
except Exception:
    pass
try:
    spark.sql(f"DROP SCHEMA IF EXISTS {FQ} CASCADE")
except Exception as e:
    print("Limpieza de schema:", e)
print("Limpieza intentada.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resumen — comparte esta tabla con el facilitador

# COMMAND ----------

print("=" * 76)
print(f"VALIDACIÓN DE ENTORNO · SESIÓN 2 — catálogo {CATALOG}")
print("=" * 76)
for req, (estado, det) in resultados.items():
    print(f"  [{estado:11s}] {req}")
    if det:
        print(f"                {det}")
print("=" * 76)
falla = [r for r, (e, _) in resultados.items() if e == "FALLA"]
if falla:
    print("⚠️  Requisitos con FALLA (pedir al administrador que los habilite):")
    for r in falla:
        print("   -", r)
else:
    print("✅ Sin fallas. El entorno soporta el laboratorio de la Sesión 2.")
try:
    dbutils.notebook.exit(json.dumps({k: v[0] for k, v in resultados.items()}, ensure_ascii=False))
except Exception:
    pass
