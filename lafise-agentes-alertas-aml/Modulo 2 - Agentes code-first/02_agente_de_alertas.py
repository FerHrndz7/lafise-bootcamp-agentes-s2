# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · El Agente de Alertas: razonamiento + herramientas + reglas de decisión
# MAGIC
# MAGIC **Objetivo:** construir, con código, un agente que analiza una alerta, **decide qué herramientas usar**,
# MAGIC y emite un **dictamen** (disposición + justificación + citas). ⏱️ ~30 min.
# MAGIC
# MAGIC ### El bucle de un agente (patrón ReAct / *tool calling*)
# MAGIC
# MAGIC     usuario ──► [modelo] ──¿pidió herramientas?──sí──► [herramientas] ──┐
# MAGIC                    ▲                                                   │
# MAGIC                    └───────────────── resultados ◄─────────────────────┘
# MAGIC                    │
# MAGIC                    └──no──► respuesta final (dictamen)
# MAGIC
# MAGIC - El **modelo** (un LLM servido por Databricks) razona y decide si necesita una herramienta.
# MAGIC - Las **herramientas** (las del notebook 01) traen datos y normativa reales → el modelo no inventa.
# MAGIC - Lo construimos con **LangGraph**, un framework open source que Databricks soporta de punta a punta
# MAGIC   (es una de las dos plantillas del **Agent Bricks CLI**, junto con OpenAI Agents SDK).
# MAGIC - **MLflow Tracing** registra cada paso para que puedas auditar *por qué* el agente decidió lo que decidió.

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

# MAGIC %md
# MAGIC ## Paso 1 — Modelo + MLflow Tracing
# MAGIC `ChatDatabricks` conecta con un modelo servido en Databricks (Foundation Model APIs). Usamos
# MAGIC `temperature=0` para que el dictamen sea lo más reproducible posible.
# MAGIC
# MAGIC `mlflow.langchain.autolog()` activa el **tracing**: cada llamada al modelo y a cada herramienta queda
# MAGIC registrada (verás la traza debajo de la celda y en el experimento de MLflow).

# COMMAND ----------

import mlflow
from databricks_langchain import ChatDatabricks

LLM = elegir_llm(w, dbutils.widgets.get("llm").strip())
llm = ChatDatabricks(endpoint=LLM, temperature=0)

mlflow.set_experiment(f"/Users/{user_email}/lafise_s2_agente_alertas")
mlflow.langchain.autolog()
print(f"Modelo: {LLM}  ·  experimento MLflow: /Users/{user_email}/lafise_s2_agente_alertas")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 2 — Las herramientas del notebook 01

# COMMAND ----------

herramientas_fn, modo = herramientas_uc(w, FUNCIONES, probar=(FUNCIONES[0], {"alerta_id": "AT-001"}))
herramientas = herramientas_fn + [herramienta_ai_search(INDEX_NAME, num_results=3)]
print(f"Funciones UC en modo: {modo}")
print("Herramientas:", [h.name for h in herramientas])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 3 — Las instrucciones: aquí vive la **lógica de decisión**
# MAGIC El *system prompt* le dice al agente **cómo** trabajar (procedimiento), **qué** puede recomendar
# MAGIC (disposiciones, alineadas con el Instructivo de la Sesión 1), qué **nunca** debe hacer (reglas duras) y
# MAGIC **en qué formato** responder (un JSON que podamos auditar y evaluar).

# COMMAND ----------

# Paso 3
# Las disposiciones replican el Instructivo de Escalamiento (cerrar / escalar / reportar) + "pedir información".
# Las reglas duras se escriben como prohibiciones explícitas: los LLM las respetan mejor que las sugerencias.
INSTRUCCIONES = """Eres el Agente de Alertas AML de la Unidad PLD de un banco centroamericano. Analizas UNA alerta del
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

Responde en español con un único bloque JSON y nada más:
{"alerta_id": "...", "disposicion": "...", "nivel_confianza": "alta|media|baja",
 "senales_clave": ["..."], "justificacion": "...",
 "citas": [{"documento": "...", "referencia": "..."}]}"""

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 4 — El grafo del agente (LangGraph)
# MAGIC Dos **nodos** y una **arista condicional**:
# MAGIC - `modelo` — llama al LLM con las instrucciones + la conversación (el LLM ya "conoce" las herramientas
# MAGIC   gracias a `bind_tools`).
# MAGIC - `herramientas` — `ToolNode` ejecuta las herramientas que el LLM pidió.
# MAGIC - Arista condicional: si el último mensaje del modelo **pide herramientas** → `herramientas`; si no → `END`.
# MAGIC - Después de `herramientas` siempre volvemos a `modelo` (para que lea los resultados).

# COMMAND ----------

from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

llm_con_herramientas = llm.bind_tools(herramientas)

def llamar_modelo(state: MessagesState):
    mensajes = [SystemMessage(content=INSTRUCCIONES)] + state["messages"]
    return {"messages": [llm_con_herramientas.invoke(mensajes)]}

# Paso 4
# La decisión mira el ÚLTIMO mensaje: si el modelo devolvió tool_calls, toca ejecutar herramientas.
def decidir(state: MessagesState):
    ultimo = state["messages"][-1]
    return "herramientas" if isinstance(ultimo, AIMessage) and ultimo.tool_calls else END

grafo = StateGraph(MessagesState)
grafo.add_node("modelo", llamar_modelo)
grafo.add_node("herramientas", ToolNode(herramientas))
grafo.set_entry_point("modelo")
grafo.add_conditional_edges("modelo", decidir, {"herramientas": "herramientas", END: END})
grafo.add_edge("herramientas", "modelo")
agente = grafo.compile()
print("✅ Agente compilado")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 5 — Dictaminar la alerta `AT-001`
# MAGIC Recuerda el caso testigo: 9 depósitos en efectivo de USD 9,2xx–9,8xx en 5 días. Observa **qué herramientas**
# MAGIC decide usar el agente y en qué orden, y abre la **traza de MLflow** debajo de la celda.

# COMMAND ----------

texto, mensajes = invocar(agente, "Analiza la alerta AT-001 y emite tu dictamen.")
print("Herramientas usadas:")
usadas = resumen_traza(mensajes)
dictamen = extraer_dictamen(texto)
print("\nDictamen:")
print(json.dumps(dictamen, ensure_ascii=False, indent=2) if dictamen else texto)

# COMMAND ----------

# Resultado esperado: disposicion = REPORTAR_ROS (o ESCALAR_INVESTIGACION), citas con politica/tipologías.
assert dictamen is not None, "El agente no devolvió un JSON válido; revisa la sección de formato del prompt"
assert dictamen.get("disposicion") in DISPOSICIONES, f"Disposición fuera del catálogo: {dictamen.get('disposicion')}"
assert dictamen["disposicion"] in ("REPORTAR_ROS", "ESCALAR_INVESTIGACION"), \
    f"AT-001 es estructuración clara; obtuvimos {dictamen['disposicion']}"
assert dictamen.get("citas"), "El dictamen debe citar al menos un fundamento"
assert "buscar_normativa_pld" in usadas and len(set(usadas)) >= 2, f"Herramientas usadas: {usadas}"
print(f"✅ AT-001 → {dictamen['disposicion']} con {len(dictamen['citas'])} cita(s)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 6 — Un caso que **no** debería escalarse: `AT-002`
# MAGIC Cliente de riesgo bajo con movimientos pequeños y consistentes. Un buen agente también sabe **no**
# MAGIC sobre-reaccionar (cada falso positivo consume horas del equipo de PLD).

# COMMAND ----------

texto2, mensajes2 = invocar(agente, "Analiza la alerta AT-002 y emite tu dictamen.")
resumen_traza(mensajes2)
dictamen2 = extraer_dictamen(texto2)
print(json.dumps(dictamen2, ensure_ascii=False, indent=2) if dictamen2 else texto2)

# COMMAND ----------

# Resultado esperado: CERRAR_SIN_ACCION (o SOLICITAR_INFORMACION)
assert dictamen2 and dictamen2.get("disposicion") in DISPOSICIONES
if dictamen2["disposicion"] in ("CERRAR_SIN_ACCION", "SOLICITAR_INFORMACION"):
    print(f"✅ AT-002 → {dictamen2['disposicion']} (no sobre-reacciona)")
else:
    print(f"⚠️ AT-002 → {dictamen2['disposicion']}. Discútelo: ¿qué regla del prompt lo empujó a escalar?")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 7 — Auditar el razonamiento con MLflow Tracing
# MAGIC Abre la pestaña de **trazas** debajo de las celdas anteriores (o el experimento
# MAGIC `lafise_s2_agente_alertas`). Para cada dictamen verás:
# MAGIC - cada llamada al **modelo** (entrada, salida, tokens, latencia),
# MAGIC - cada **herramienta** llamada, con sus argumentos y resultado,
# MAGIC - el orden exacto de los pasos → evidencia para el auditor de *por qué* se recomendó esa disposición.
# MAGIC
# MAGIC En la **Sesión 4** usaremos estas trazas para **evaluar** el agente contra alucinaciones.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resumen
# MAGIC - Construiste el **bucle de razonamiento** de un agente con LangGraph (modelo ⇄ herramientas).
# MAGIC - Pusiste la **lógica de decisión** del área de PLD en las instrucciones: disposiciones + reglas duras + formato.
# MAGIC - El agente dictaminó una alerta **citando** la normativa, y cada paso quedó **trazado** en MLflow.
# MAGIC
# MAGIC ➡️ En `03_supervisor_y_subagentes` dividimos el trabajo entre **especialistas** coordinados por un supervisor.
