# Databricks notebook source
# MAGIC %md
# MAGIC # Utilidades compartidas — Agente de Alertas AML (Sesión 2)
# MAGIC
# MAGIC Los notebooks `02`–`05` lo cargan con `%run ../utils/agente_utils` (o `./utils/agente_utils`). Contiene
# MAGIC la "fontanería" para que los ejercicios se concentren en lo importante:
# MAGIC
# MAGIC - `elegir_llm(...)` — elige un modelo de chat con *tool calling* disponible en el workspace.
# MAGIC - `herramientas_uc(...)` — expone funciones de Unity Catalog como herramientas del agente, en
# MAGIC   **cómputo clásico o serverless** (ver nota abajo).
# MAGIC - `herramienta_ai_search(...)` — el índice de AI Search de la Sesión 1 como herramienta de retrieval.
# MAGIC - `crear_agente(...)` — el bucle de razonamiento (LangGraph) que construimos en el notebook `02`.
# MAGIC - `extraer_dictamen(...)` / `resumen_traza(...)` — leer la respuesta y ver qué herramientas usó.
# MAGIC
# MAGIC > **Funciones UC en cómputo clásico.** El cliente oficial de Unity Catalog AI (`DatabricksFunctionClient`)
# MAGIC > ejecuta las funciones en **cómputo serverless**. Si tu workspace no tiene serverless para notebooks,
# MAGIC > `herramientas_uc` cae automáticamente a ejecutar **la misma función de UC** con `spark.sql` en tu
# MAGIC > clúster clásico. La función sigue siendo un objeto gobernado de UC (permiso `EXECUTE`, linaje,
# MAGIC > auditoría); solo cambia *dónde* se ejecuta.

# COMMAND ----------

import json, re
from typing import Any

from databricks.sdk import WorkspaceClient
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import Field, create_model

DISPOSICIONES = ["CERRAR_SIN_ACCION", "ESCALAR_INVESTIGACION", "REPORTAR_ROS", "SOLICITAR_INFORMACION"]

# Formato de salida común a todos los agentes del módulo (lo auditamos y, en la Sesión 4, lo evaluamos).
FORMATO_DICTAMEN = """Responde en español con un único bloque JSON y nada más:
{"alerta_id": "...", "disposicion": "...", "nivel_confianza": "alta|media|baja",
 "senales_clave": ["..."], "justificacion": "...",
 "citas": [{"documento": "...", "referencia": "..."}]}"""

# Las instrucciones del agente único que construimos en el notebook 02 (versión de referencia).
INSTRUCCIONES_AGENTE_ALERTAS = """Eres el Agente de Alertas AML de la Unidad PLD de un banco centroamericano. Analizas UNA alerta del
monitoreo transaccional y recomiendas su disposición a un analista humano, que toma la decisión final.

Procedimiento:
1. Usa obtener_alerta para leer la alerta y el perfil KYC del cliente.
2. Usa features_transaccionales para obtener las señales de los 30 días previos.
3. Usa buscar_normativa_pld para encontrar el artículo de la política, la tipología aplicable y, si existe,
   un precedente similar. Haz al menos una búsqueda.
4. Decide la disposición.

Disposiciones posibles (elige exactamente una):
- CERRAR_SIN_ACCION: el análisis descarta el indicio y la actividad es consistente con el perfil del cliente.
- SOLICITAR_INFORMACION: faltan datos para decidir (por ejemplo, soporte del origen de los fondos).
- ESCALAR_INVESTIGACION: persisten indicios que requieren profundizar (pass-through, testaferros, alto riesgo).
- REPORTAR_ROS: los hechos configuran una tipología con evidencia clara; corresponde un ROS a la UAF.

Reglas duras:
- Nunca inventes cifras: usa solo las que devuelven las herramientas.
- Toda disposición debe citar al menos un fundamento (documento + artículo o tipología) recuperado con
  buscar_normativa_pld.
- Si el cliente es PEP o hay contrapartes en jurisdicciones de alto riesgo, no cierres sin acción.

""" + FORMATO_DICTAMEN

# Modelos con tool calling, en orden de preferencia. Se usa el primero que esté servido en el workspace.
LLM_PREFERIDOS = [
    "databricks-claude-sonnet-4-6", "databricks-claude-sonnet-4-5", "databricks-claude-sonnet-4",
    "databricks-meta-llama-3-3-70b-instruct", "databricks-llama-4-maverick",
    "databricks-qwen3-next-80b-a3b-instruct", "databricks-gemini-3-5-flash",
]

# COMMAND ----------

def elegir_llm(w: WorkspaceClient, preferido: str = "") -> str:
    """Devuelve el nombre del endpoint de chat a usar (el del widget si existe; si no, el primero disponible)."""
    servidos = {e.name for e in w.serving_endpoints.list()}
    if preferido and preferido in servidos:
        return preferido
    for cand in LLM_PREFERIDOS:
        if cand in servidos:
            return cand
    raise RuntimeError("No encontré un modelo de chat con tool calling servido. Revisa Foundation Model APIs "
                       "o escribe el nombre de un endpoint en el widget 'llm'.")


def _cliente_uc_serverless(probar_fn: str, params: dict):
    """Intenta el DatabricksFunctionClient (ejecuta en serverless). Devuelve el cliente o None."""
    try:
        from unitycatalog.ai.core.databricks import DatabricksFunctionClient
        cliente = DatabricksFunctionClient()
        r = cliente.execute_function(probar_fn, params)
        return cliente if not r.error else None
    except Exception:
        return None


def _herramienta_spark(w: WorkspaceClient, nombre_fn: str) -> StructuredTool:
    """Envuelve una función de UC como herramienta que se ejecuta con spark.sql en el clúster clásico."""
    info = w.functions.get(nombre_fn)
    params = list(info.input_params.parameters) if info.input_params else []
    campos = {p.name: (str, Field(description=p.comment or p.name)) for p in params}
    Args = create_model(f"Args_{info.name}", **campos)
    marcadores = ", ".join(f":{p.name}" for p in params)
    # La sesión se captura AQUÍ (hilo principal del notebook): ToolNode ejecuta las herramientas en hilos
    # auxiliares, donde SparkSession.getActiveSession() devuelve None.
    from pyspark.sql import SparkSession
    spark_ = SparkSession.getActiveSession()

    def _ejecutar(**kwargs):
        filas = spark_.sql(f"SELECT * FROM {nombre_fn}({marcadores})", args=kwargs).limit(50).collect()
        return json.dumps([r.asDict() for r in filas], ensure_ascii=False, default=str)

    return StructuredTool.from_function(func=_ejecutar, name=info.name, args_schema=Args,
                                        description=info.comment or info.name)


def herramientas_uc(w: WorkspaceClient, nombres_fn: list, probar: tuple, modo: str = "auto"):
    """Funciones de UC -> herramientas LangChain.

    modo='auto'  : usa UCFunctionToolkit (serverless) si funciona; si no, spark.sql en el clúster clásico.
    modo='spark' : fuerza la ejecución en el clúster clásico.
    `probar` = (nombre_fn, params) para comprobar si la vía serverless está disponible.
    Devuelve (lista_de_herramientas, modo_usado).
    La variable de entorno AGENTE_UC_MODO=spark fuerza el modo clásico (útil para probar ese camino).
    """
    import os
    modo = os.environ.get("AGENTE_UC_MODO", modo)
    if modo == "auto":
        cliente = _cliente_uc_serverless(*probar)
        if cliente is not None:
            from databricks_langchain import UCFunctionToolkit
            tools = list(UCFunctionToolkit(function_names=nombres_fn, client=cliente).tools)
            for t in tools:   # el toolkit nombra 'catalogo__schema__fn' (truncado a 64); usamos el nombre corto
                t.name = next(f.split(".")[-1] for f in nombres_fn if t.name.endswith(f.split(".")[-1]))
            return tools, "serverless"
    return [_herramienta_spark(w, f) for f in nombres_fn], "clasico (spark.sql)"


def herramienta_ai_search(index_name: str, num_results: int = 4):
    """El índice de AI Search (antes Vector Search) de la Sesión 1, como herramienta de retrieval híbrido."""
    from databricks_langchain import VectorSearchRetrieverTool
    return VectorSearchRetrieverTool(
        index_name=index_name,
        tool_name="buscar_normativa_pld",
        tool_description=("Busca en la base de conocimiento PLD: artículos de la política PLD/FT, tipologías de "
                          "lavado, instructivo de disposición de alertas y precedentes (narrativas de casos). "
                          "Úsala para fundamentar y CITAR la disposición."),
        columns=["source_doc", "articulo", "seccion", "chunk_to_retrieve"],
        num_results=num_results,
        query_type="HYBRID",
    )

# COMMAND ----------

def crear_agente(llm, herramientas: list, instrucciones: str):
    """Agente de tool calling (patrón ReAct) con LangGraph: modelo -> ¿pidió herramientas? -> herramientas -> modelo."""
    from langgraph.graph import END, MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode

    llm_con_herramientas = llm.bind_tools(herramientas)

    def llamar_modelo(state: MessagesState):
        mensajes = [SystemMessage(content=instrucciones)] + state["messages"]
        return {"messages": [llm_con_herramientas.invoke(mensajes)]}

    def decidir(state: MessagesState):
        ultimo = state["messages"][-1]
        return "herramientas" if isinstance(ultimo, AIMessage) and ultimo.tool_calls else END

    grafo = StateGraph(MessagesState)
    grafo.add_node("modelo", llamar_modelo)
    grafo.add_node("herramientas", ToolNode(herramientas))
    grafo.set_entry_point("modelo")
    grafo.add_conditional_edges("modelo", decidir, {"herramientas": "herramientas", END: END})
    grafo.add_edge("herramientas", "modelo")
    return grafo.compile()


def invocar(agente, pregunta: str, max_pasos: int = 25):
    """Ejecuta el agente con una pregunta y devuelve (texto_final, mensajes)."""
    out = agente.invoke({"messages": [{"role": "user", "content": pregunta}]},
                        config={"recursion_limit": max_pasos})
    msgs = out["messages"]
    return _texto(msgs[-1].content), msgs


def _texto(contenido: Any) -> str:
    """Algunos modelos devuelven el contenido como lista de bloques; lo normalizamos a texto."""
    if isinstance(contenido, str):
        return contenido
    if isinstance(contenido, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in contenido)
    return str(contenido)

# COMMAND ----------

def extraer_dictamen(texto: str):
    """Extrae el JSON del dictamen de la respuesta del agente. Devuelve dict o None."""
    if not texto:
        return None
    bloque = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", texto, re.S)
    candidato = bloque.group(1) if bloque else None
    if candidato is None:
        ini, fin = texto.find("{"), texto.rfind("}")
        candidato = texto[ini:fin + 1] if ini != -1 and fin > ini else None
    if candidato is None:
        return None
    try:
        return json.loads(candidato)
    except json.JSONDecodeError:
        return None


def resumen_traza(mensajes) -> list:
    """Imprime y devuelve la secuencia de herramientas que llamó el agente."""
    llamadas = []
    for m in mensajes:
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                llamadas.append(tc["name"])
                print(f"  🔧 {tc['name']}({json.dumps(tc['args'], ensure_ascii=False)[:120]})")
        elif isinstance(m, ToolMessage):
            print(f"     ↳ {_texto(m.content)[:140]}...")
    return llamadas

print("✅ agente_utils cargado: elegir_llm, herramientas_uc, herramienta_ai_search, crear_agente, invocar, "
      "extraer_dictamen, resumen_traza")
