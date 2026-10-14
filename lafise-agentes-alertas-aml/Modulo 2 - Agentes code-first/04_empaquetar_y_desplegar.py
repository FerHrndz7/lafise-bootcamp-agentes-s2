# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Del notebook a producción: empaquetar, registrar y desplegar
# MAGIC
# MAGIC **Objetivo:** convertir el agente en un **artefacto versionado y gobernado** en Unity Catalog, y conocer los
# MAGIC caminos para desplegarlo. ⏱️ ~15 min.
# MAGIC
# MAGIC ### Las tres capas de un agente desplegado en Databricks
# MAGIC | Capa | Qué hace | Hoy |
# MAGIC |---|---|---|
# MAGIC | **Framework** | El bucle del agente (modelo ⇄ herramientas) | LangGraph (notebooks 02–03) |
# MAGIC | **Interfaz / servidor** | Expone el agente con una API estándar | `ResponsesAgent` de MLflow (API tipo OpenAI Responses) |
# MAGIC | **Runtime** | Dónde corre en producción | **Agent Runtime** (Databricks Apps, recomendado) o **Model Serving** |
# MAGIC
# MAGIC En este notebook (todo en **cómputo clásico**):
# MAGIC 1. Escribimos el agente en un archivo `agente_alertas.py` con la interfaz `ResponsesAgent`.
# MAGIC 2. Lo probamos localmente.
# MAGIC 3. Lo registramos en **Unity Catalog** como modelo versionado, declarando sus **recursos** (modelo, índice,
# MAGIC    funciones) para que al desplegarse obtenga credenciales automáticamente.
# MAGIC 4. Vemos los caminos de despliegue (el despliegue en sí es **opcional** en el lab).

# COMMAND ----------

# MAGIC %pip install --quiet "databricks-langchain>=0.20.0" "langgraph>=1.2,<1.3" "unitycatalog-langchain[databricks]>=0.3.0" "mlflow[databricks]>=3.17" "databricks-agents>=1.12" "databricks-sdk>=0.94.0"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import re, os, json

dbutils.widgets.text("catalog", "", "Catálogo asignado (REQUERIDO)")
dbutils.widgets.text("schema", "pld_aml", "Schema base")
dbutils.widgets.text("llm", "", "Endpoint LLM (opcional)")
dbutils.widgets.dropdown("desplegar", "no", ["no", "si"], "¿Desplegar en Model Serving? (opcional)")
user_email = spark.sql("SELECT current_user()").collect()[0][0]
user_prefix = re.sub(r"[^a-z0-9]", "_", user_email.split("@")[0].lower())
CATALOG = dbutils.widgets.get("catalog").strip()
assert CATALOG, "Escribe en el widget 'catalog' el mismo catálogo que usaste en 00_setup."
SCHEMA = f"{(dbutils.widgets.get('schema').strip() or 'pld_aml')}_{user_prefix}"
INDEX_NAME = f"{CATALOG}.{SCHEMA}.chunks_index"
FUNCIONES = [f"{CATALOG}.{SCHEMA}.obtener_alerta", f"{CATALOG}.{SCHEMA}.features_transaccionales"]
UC_MODEL = f"{CATALOG}.{SCHEMA}.agente_alertas_aml"

from databricks.sdk import WorkspaceClient
w = WorkspaceClient()

# COMMAND ----------

# MAGIC %run ./utils/agente_utils

# COMMAND ----------

LLM = elegir_llm(w, dbutils.widgets.get("llm").strip())
# Carpeta personal (en tu Volumen de la Sesión 1) donde vive el código del agente.
spark.sql(f"CREATE VOLUME IF NOT EXISTS `{CATALOG}`.{SCHEMA}.artefactos")
AGENT_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/artefactos/agente"
os.makedirs(AGENT_DIR, exist_ok=True)
AGENT_FILE = f"{AGENT_DIR}/agente_alertas.py"

# La configuración NO va escrita en el código: se inyecta con ModelConfig (dev: variable de entorno;
# registrado: model_config). Así el mismo archivo sirve para todos los participantes y entornos.
CONFIG = {"llm_endpoint": LLM, "index_name": INDEX_NAME, "funciones": FUNCIONES,
          "instrucciones": INSTRUCCIONES_AGENTE_ALERTAS}
os.environ["AGENTE_ALERTAS_CONFIG"] = json.dumps(CONFIG, ensure_ascii=False)
print(f"Modelo: {LLM}\nArchivo del agente: {AGENT_FILE}\nModelo en UC: {UC_MODEL}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 1 — El agente como archivo, con la interfaz `ResponsesAgent`
# MAGIC `ResponsesAgent` es la interfaz de agentes de **MLflow 3**: entrada/salida compatibles con la API *Responses*
# MAGIC de OpenAI, *streaming*, y compatibilidad con AI Playground, evaluación (Sesión 4) y monitoreo.
# MAGIC
# MAGIC Lee el código: es el **mismo grafo del notebook 02**, envuelto en
# MAGIC una clase con `predict` / `predict_stream`. La última línea, `mlflow.models.set_model(...)`, le dice a MLflow
# MAGIC cuál es el agente (*models from code*).

# COMMAND ----------

# MAGIC %%writefile {AGENT_FILE}
# MAGIC import json
# MAGIC import os
# MAGIC from typing import Generator
# MAGIC
# MAGIC import mlflow
# MAGIC from databricks_langchain import ChatDatabricks, VectorSearchRetrieverTool
# MAGIC from langchain_core.messages import AIMessage, SystemMessage
# MAGIC from langgraph.graph import END, MessagesState, StateGraph
# MAGIC from langgraph.prebuilt import ToolNode
# MAGIC from mlflow.pyfunc import ResponsesAgent
# MAGIC from mlflow.types.responses import (
# MAGIC     ResponsesAgentRequest, ResponsesAgentResponse, ResponsesAgentStreamEvent,
# MAGIC     output_to_responses_items_stream, to_chat_completions_input,
# MAGIC )
# MAGIC
# MAGIC # Configuración: en desarrollo viene de una variable de entorno; registrado, de model_config.
# MAGIC _dev = os.environ.get("AGENTE_ALERTAS_CONFIG")
# MAGIC cfg = mlflow.models.ModelConfig(development_config=json.loads(_dev) if _dev else None)
# MAGIC
# MAGIC
# MAGIC def _herramientas_uc(funciones):
# MAGIC     """UCFunctionToolkit (serverless) si está disponible; si no, spark.sql en el clúster clásico (solo notebook)."""
# MAGIC     try:
# MAGIC         if os.environ.get("AGENTE_UC_MODO") == "spark":
# MAGIC             raise RuntimeError("modo clásico forzado")
# MAGIC         from databricks_langchain import UCFunctionToolkit
# MAGIC         from unitycatalog.ai.core.databricks import DatabricksFunctionClient
# MAGIC         cliente = DatabricksFunctionClient()
# MAGIC         prueba = cliente.execute_function(funciones[0], {"alerta_id": "AT-001"})
# MAGIC         if not prueba.error:
# MAGIC             tools = list(UCFunctionToolkit(function_names=funciones, client=cliente).tools)
# MAGIC             for t in tools:   # nombre corto (el toolkit usa 'catalogo__schema__fn', truncado a 64)
# MAGIC                 t.name = next(f.split(".")[-1] for f in funciones if t.name.endswith(f.split(".")[-1]))
# MAGIC             return tools
# MAGIC     except Exception:
# MAGIC         pass
# MAGIC     from databricks.sdk import WorkspaceClient
# MAGIC     from langchain_core.tools import StructuredTool
# MAGIC     from pydantic import Field, create_model
# MAGIC     from pyspark.sql import SparkSession
# MAGIC     spark = SparkSession.getActiveSession()
# MAGIC     if spark is None:
# MAGIC         raise RuntimeError("Las funciones UC requieren serverless (o un clúster con Spark activo).")
# MAGIC     w = WorkspaceClient()
# MAGIC     herramientas = []
# MAGIC     for fn in funciones:
# MAGIC         info = w.functions.get(fn)
# MAGIC         params = list(info.input_params.parameters) if info.input_params else []
# MAGIC         Args = create_model(f"Args_{info.name}", **{p.name: (str, Field(description=p.comment or p.name)) for p in params})
# MAGIC         sql = f"SELECT * FROM {fn}(" + ", ".join(f":{p.name}" for p in params) + ")"
# MAGIC         def _run(_sql=sql, **kw):
# MAGIC             return json.dumps([r.asDict() for r in spark.sql(_sql, args=kw).limit(50).collect()],
# MAGIC                               ensure_ascii=False, default=str)
# MAGIC         herramientas.append(StructuredTool.from_function(func=_run, name=info.name, args_schema=Args,
# MAGIC                                                          description=info.comment or info.name))
# MAGIC     return herramientas
# MAGIC
# MAGIC
# MAGIC def _construir_grafo():
# MAGIC     llm = ChatDatabricks(endpoint=cfg.get("llm_endpoint"), temperature=0)
# MAGIC     herramientas = _herramientas_uc(cfg.get("funciones")) + [VectorSearchRetrieverTool(
# MAGIC         index_name=cfg.get("index_name"), tool_name="buscar_normativa_pld", num_results=3, query_type="HYBRID",
# MAGIC         columns=["source_doc", "articulo", "seccion", "chunk_to_retrieve"],
# MAGIC         tool_description="Busca en la base de conocimiento PLD (política, tipologías, instructivo, precedentes).")]
# MAGIC     llm_h = llm.bind_tools(herramientas)
# MAGIC     instrucciones = cfg.get("instrucciones")
# MAGIC
# MAGIC     def modelo(state: MessagesState):
# MAGIC         return {"messages": [llm_h.invoke([SystemMessage(content=instrucciones)] + state["messages"])]}
# MAGIC
# MAGIC     def decidir(state: MessagesState):
# MAGIC         ultimo = state["messages"][-1]
# MAGIC         return "herramientas" if isinstance(ultimo, AIMessage) and ultimo.tool_calls else END
# MAGIC
# MAGIC     g = StateGraph(MessagesState)
# MAGIC     g.add_node("modelo", modelo)
# MAGIC     g.add_node("herramientas", ToolNode(herramientas))
# MAGIC     g.set_entry_point("modelo")
# MAGIC     g.add_conditional_edges("modelo", decidir, {"herramientas": "herramientas", END: END})
# MAGIC     g.add_edge("herramientas", "modelo")
# MAGIC     return g.compile()
# MAGIC
# MAGIC
# MAGIC class AgenteAlertas(ResponsesAgent):
# MAGIC     def __init__(self):
# MAGIC         self._grafo = None          # se construye en la primera petición
# MAGIC
# MAGIC     @property
# MAGIC     def grafo(self):
# MAGIC         if self._grafo is None:
# MAGIC             self._grafo = _construir_grafo()
# MAGIC         return self._grafo
# MAGIC
# MAGIC     def predict_stream(self, request: ResponsesAgentRequest) -> Generator[ResponsesAgentStreamEvent, None, None]:
# MAGIC         mensajes = to_chat_completions_input([m.model_dump() for m in request.input])
# MAGIC         for _, evento in self.grafo.stream({"messages": mensajes}, stream_mode=["updates"]):
# MAGIC             for nodo in evento.values():
# MAGIC                 if nodo.get("messages"):
# MAGIC                     yield from output_to_responses_items_stream(nodo["messages"])
# MAGIC
# MAGIC     def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
# MAGIC         items = [e.item for e in self.predict_stream(request) if e.type == "response.output_item.done"]
# MAGIC         return ResponsesAgentResponse(output=items, custom_outputs=request.custom_inputs)
# MAGIC
# MAGIC
# MAGIC mlflow.langchain.autolog()
# MAGIC mlflow.models.set_model(AgenteAlertas())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 2 — Probar el agente empaquetado (localmente, en tu clúster)
# MAGIC Importamos el archivo y le enviamos una petición con el formato **Responses** (`input` = lista de mensajes).

# COMMAND ----------

import importlib.util, mlflow
spec = importlib.util.spec_from_file_location("agente_alertas", AGENT_FILE)
modulo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(modulo)
agente_pkg = modulo.AgenteAlertas()

PETICION = {"input": [{"role": "user", "content": "Analiza la alerta AT-001 y emite tu dictamen."}]}

def texto_final(respuesta):
    """Texto del último mensaje del asistente en una respuesta ResponsesAgent (objeto o dict)."""
    salida = respuesta["output"] if isinstance(respuesta, dict) else [i.model_dump() if hasattr(i, "model_dump") else i for i in respuesta.output]
    mensajes = [i for i in salida if i.get("type") == "message"]
    return "".join(c.get("text", "") for c in mensajes[-1]["content"]) if mensajes else ""

resp = agente_pkg.predict(PETICION)
dictamen_local = extraer_dictamen(texto_final(resp))
print(json.dumps(dictamen_local, ensure_ascii=False, indent=2))

# COMMAND ----------

assert dictamen_local and dictamen_local.get("disposicion") in DISPOSICIONES
print(f"✅ Agente empaquetado responde: {dictamen_local['disposicion']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 3 — Registrar en Unity Catalog (con sus **recursos**)
# MAGIC `resources=[...]` declara **todo lo que el agente usa**: el modelo de chat, el índice de AI Search (y su
# MAGIC modelo de embeddings) y las funciones de UC. Al desplegarlo, Databricks le da credenciales de corta duración
# MAGIC para exactamente esos recursos — **sin** tokens ni secretos en el código.
# MAGIC
# MAGIC > Si olvidas un recurso, el agente desplegado falla con `PERMISSION_DENIED` al usarlo.

# COMMAND ----------

from importlib.metadata import version
from mlflow.models.resources import DatabricksFunction, DatabricksServingEndpoint, DatabricksVectorSearchIndex

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(f"/Users/{user_email}/lafise_s2_agente_alertas")
EMB = w.vector_search_indexes.get_index(INDEX_NAME).delta_sync_index_spec.embedding_source_columns[0].embedding_model_endpoint_name

# Paso 3 — un recurso por cada cosa que el agente toca.
RECURSOS = [
    DatabricksServingEndpoint(endpoint_name=LLM),
    DatabricksServingEndpoint(endpoint_name=EMB),
    DatabricksVectorSearchIndex(index_name=INDEX_NAME),
    *[DatabricksFunction(function_name=f) for f in FUNCIONES],
]

REQUISITOS = [f"mlflow=={version('mlflow')}", f"databricks-langchain=={version('databricks-langchain')}",
              f"langgraph=={version('langgraph')}", f"databricks-sdk=={version('databricks-sdk')}",
              f"unitycatalog-langchain[databricks]=={version('unitycatalog-langchain')}"]

with mlflow.start_run(run_name="agente_alertas_aml"):
    info = mlflow.pyfunc.log_model(
        name="agente_alertas", python_model=AGENT_FILE, model_config=CONFIG, resources=RECURSOS,
        input_example=PETICION, pip_requirements=REQUISITOS, registered_model_name=UC_MODEL)
print(f"✅ Registrado {UC_MODEL} versión {info.registered_model_version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 4 — Cargar la versión registrada y probarla
# MAGIC Esto es exactamente lo que hará el runtime de producción: cargar **la versión de UC** (código + config +
# MAGIC dependencias declaradas) y responder.

# COMMAND ----------

os.environ.pop("AGENTE_ALERTAS_CONFIG", None)       # ahora la config viene del modelo registrado
# Paso 4 — cargar por URI de UC (models:/<catálogo.schema.modelo>/<versión>) y predecir.
registrado = mlflow.pyfunc.load_model(f"models:/{UC_MODEL}/{info.registered_model_version}")
resp_reg = registrado.predict(PETICION)
dictamen_reg = extraer_dictamen(texto_final(resp_reg))
print(json.dumps(dictamen_reg, ensure_ascii=False, indent=2))

# COMMAND ----------

# Resultado esperado: el modelo registrado devuelve un dictamen válido para AT-001
assert dictamen_reg and dictamen_reg.get("disposicion") in DISPOSICIONES
print(f"✅ La versión {info.registered_model_version} registrada en UC responde: {dictamen_reg['disposicion']}")
display(spark.sql(f"DESCRIBE FUNCTION EXTENDED `{CATALOG}`.{SCHEMA}.obtener_alerta").limit(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 5 — Caminos de despliegue
# MAGIC
# MAGIC | Camino | Estado (oct-2026) | Cuándo |
# MAGIC |---|---|---|
# MAGIC | **Agent Bricks CLI → Agent Runtime** (Databricks Apps) | **Recomendado** para agentes nuevos · CLI en *Beta* | Producción: endpoint autenticado, identidad propia (service principal), memoria/sesiones gestionadas, ejecución durable, tracing, UI de chat |
# MAGIC | **Model Serving** con `agents.deploy()` | Soportado (ya no es el camino recomendado; sin fecha de retiro) | Desplegar directo desde un notebook este artefacto de UC; integra Review App y tablas de inferencia |
# MAGIC
# MAGIC Ambos son **servicios gestionados serverless** (no dependen del cómputo clásico de tu notebook).
# MAGIC
# MAGIC > ⚠️ **Limitación conocida (Model Serving):** si la cuenta tiene activada la vista previa **Foundation Model
# MAGIC > UC Permissions**, un agente personalizado en Model Serving con autenticación automática **no puede** llamar
# MAGIC > modelos de `system.ai`, y eso incluye el modelo de embeddings del índice de AI Search. Falla con
# MAGIC > `User is missing privileges: USE CATALOG on system`. Está documentado; las soluciones son desplegar en
# MAGIC > **Databricks Apps / Agent Runtime** (camino recomendado) o usar autenticación *on-behalf-of-user*.
# MAGIC
# MAGIC **Demo del facilitador — Agent Bricks CLI** (desde una terminal con la CLI de Databricks autenticada):
# MAGIC
# MAGIC     pip install databricks-agentbricks
# MAGIC     agentbricks init --framework langgraph --profile <perfil> agente-alertas-aml   # plantilla LangGraph
# MAGIC     cd agente-alertas-aml
# MAGIC     agentbricks tools add uc-function <catálogo>.<schema>.obtener_alerta           # ¡las mismas funciones de UC!
# MAGIC     agentbricks tools add uc-function <catálogo>.<schema>.features_transaccionales
# MAGIC     # instrucciones (system_prompt) + retriever de AI Search como herramienta local en agent/tools/
# MAGIC     agentbricks dev                                                                # probar local
# MAGIC     agentbricks deploy agente-alertas                                              # Agent Runtime (app agent-bricks-agente-alertas)
# MAGIC
# MAGIC Para un agente LangGraph que ya existe: `agentbricks init --framework langgraph --existing .`
# MAGIC
# MAGIC **Opcional en el lab — Model Serving:** cambia el widget **desplegar** a `si` y ejecuta la celda siguiente.
# MAGIC Tarda **~15–20 min** y crea un endpoint con tu usuario. Bórralo al terminar (ver la última celda).

# COMMAND ----------

ENDPOINT = f"agente-alertas-{user_prefix}"[:63].replace("_", "-")
if dbutils.widgets.get("desplegar") == "si":
    from databricks import agents
    despliegue = agents.deploy(UC_MODEL, info.registered_model_version, endpoint_name=ENDPOINT,
                               scale_to_zero=True, tags={"curso": "lafise-s2"})
    print(f"Desplegando en el endpoint {despliegue.endpoint_name} (puede tardar 15–20 min)")
else:
    print("ℹ️ Despliegue omitido (widget 'desplegar' = no). El agente queda registrado en UC y listo para desplegar.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Limpieza (solo si desplegaste)
# MAGIC Un endpoint de Model Serving consume mientras está activo. Para borrarlo:
# MAGIC
# MAGIC     from databricks import agents
# MAGIC     agents.delete_deployment(model_name=UC_MODEL, model_version=<versión>)
# MAGIC
# MAGIC (o desde **Serving** en la UI → tu endpoint → *Delete*).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resumen
# MAGIC - Empaquetaste el agente con la interfaz **`ResponsesAgent`** de MLflow 3 (*models from code* + `ModelConfig`).
# MAGIC - Lo registraste en **Unity Catalog** como modelo versionado, con sus **recursos** declarados.
# MAGIC - Cargaste la versión registrada y respondió igual que en desarrollo.
# MAGIC - Conociste los caminos de despliegue: **Agent Bricks CLI → Agent Runtime** (recomendado) y Model Serving.
# MAGIC
# MAGIC ➡️ `05_reto`: el agente tiene un punto ciego peligroso… las **listas de sanciones**.
