# Databricks notebook source
# MAGIC %md
# MAGIC # LAFISE — Bootcamp de Agentes AML · Sesión 2
# MAGIC ## Agentes code-first sobre Databricks: el Agente de Alertas que razona y recomienda
# MAGIC
# MAGIC En la **Sesión 1** construimos la **base de conocimiento** (políticas PLD, tipologías y precedentes en
# MAGIC **Databricks AI Search**, antes *Vector Search*). Hoy construimos, **con código**, el **Agente de Alertas**
# MAGIC que la usa: recibe una alerta del monitoreo transaccional, reúne contexto con **herramientas gobernadas en
# MAGIC Unity Catalog**, razona y **recomienda una disposición citando su fundamento**.
# MAGIC
# MAGIC > **Sobre los nombres.** Lo que en el plan del bootcamp llamamos *Mosaic AI Agent Framework* hoy forma parte de
# MAGIC > **Agent Bricks**, la plataforma de desarrollo de agentes de Databricks. Esta sesión cubre el camino
# MAGIC > **code-first** (agentes personalizados en Python con LangGraph + MLflow); la Sesión 3 cubre el camino
# MAGIC > **low-code** (Knowledge Assistant, Supervisor Agent).

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## A. Aviso
# MAGIC
# MAGIC <div style="border-left: 4px solid #f44336; background: #ffebee; padding: 14px 18px; border-radius: 4px; margin: 16px 0;">
# MAGIC <strong style="display:block; color:#c62828; margin-bottom:6px; font-size: 1.1em;">Solo laboratorio · datos sintéticos</strong>
# MAGIC <div style="color:#333;">
# MAGIC <ul>
# MAGIC <li>Ejecuta esta práctica únicamente en el <strong>workspace de laboratorio</strong> proporcionado.</li>
# MAGIC <li>Clientes, transacciones, alertas y listas de sanciones son <strong>sintéticos</strong>. <strong>Nunca</strong> uses PII real de clientes de LAFISE.</li>
# MAGIC <li>El agente <strong>recomienda</strong>; la decisión final siempre es del analista humano. Los umbrales y plazos son <strong>didácticos</strong>, no asesoría de cumplimiento.</li>
# MAGIC </ul>
# MAGIC </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## B. Recorrido de la sesión (2 h)
# MAGIC
# MAGIC <div style="max-width: 1200px; margin: 0 auto; font-family: sans-serif;">
# MAGIC <div style="background: #F9F7F4; border-radius: 10px; padding: 20px 24px; box-shadow: 0 2px 8px rgba(27,49,57,0.06);">
# MAGIC <style>.agenda-table td, .agenda-table th { font-size: 14pt !important; }</style>
# MAGIC <table class="agenda-table" style="width: 100%; border-collapse: collapse; line-height: 1.5;">
# MAGIC   <thead>
# MAGIC      <tr style="background: #1B5162; color: white;">
# MAGIC       <th style="padding: 10px 14px; text-align: center; border: 1px solid #EEEDE9; width: 60px;">#</th>
# MAGIC       <th style="padding: 10px 14px; text-align: center; border: 1px solid #EEEDE9; width: 90px;">Tiempo</th>
# MAGIC       <th style="padding: 10px 14px; text-align: left; border: 1px solid #EEEDE9;">Notebook</th>
# MAGIC      </tr>
# MAGIC   </thead>
# MAGIC   <tbody>
# MAGIC      <tr style="background: white;"><td style="padding: 8px 14px; border: 1px solid #EEEDE9; text-align:center; font-weight:700; color:#1B5162;">—</td><td style="padding: 8px 14px; border: 1px solid #EEEDE9; text-align:center;">15 min</td><td style="padding: 8px 14px; border: 1px solid #EEEDE9;">Introducción del facilitador: agentes code-first en Databricks</td></tr>
# MAGIC      <tr style="background: #FBFAF8;"><td style="padding: 8px 14px; border: 1px solid #EEEDE9; text-align:center; font-weight:700; color:#1B5162;">00</td><td style="padding: 8px 14px; border: 1px solid #EEEDE9; text-align:center;">10 min</td><td style="padding: 8px 14px; border: 1px solid #EEEDE9;">Preparación — dependencias + base de la Sesión 1 + casos testigo</td></tr>
# MAGIC      <tr style="background: white;"><td style="padding: 8px 14px; border: 1px solid #EEEDE9; text-align:center; font-weight:700; color:#1B5162;">01</td><td style="padding: 8px 14px; border: 1px solid #EEEDE9; text-align:center;">20 min</td><td style="padding: 8px 14px; border: 1px solid #EEEDE9;">Herramientas gobernadas: funciones de Unity Catalog + retriever de AI Search</td></tr>
# MAGIC      <tr style="background: #FBFAF8;"><td style="padding: 8px 14px; border: 1px solid #EEEDE9; text-align:center; font-weight:700; color:#1B5162;">02</td><td style="padding: 8px 14px; border: 1px solid #EEEDE9; text-align:center;">30 min</td><td style="padding: 8px 14px; border: 1px solid #EEEDE9;">El Agente de Alertas: LangGraph + lógica de decisión + MLflow Tracing</td></tr>
# MAGIC      <tr style="background: white;"><td style="padding: 8px 14px; border: 1px solid #EEEDE9; text-align:center; font-weight:700; color:#1B5162;">03</td><td style="padding: 8px 14px; border: 1px solid #EEEDE9; text-align:center;">20 min</td><td style="padding: 8px 14px; border: 1px solid #EEEDE9;">Supervisor + sub-agentes especialistas</td></tr>
# MAGIC      <tr style="background: #FBFAF8;"><td style="padding: 8px 14px; border: 1px solid #EEEDE9; text-align:center; font-weight:700; color:#1B5162;">04</td><td style="padding: 8px 14px; border: 1px solid #EEEDE9; text-align:center;">15 min</td><td style="padding: 8px 14px; border: 1px solid #EEEDE9;">Empaquetar (<code>ResponsesAgent</code>), registrar en UC y caminos de despliegue</td></tr>
# MAGIC      <tr style="background: white;"><td style="padding: 8px 14px; border: 1px solid #EEEDE9; text-align:center; font-weight:700; color:#1B5162;">05</td><td style="padding: 8px 14px; border: 1px solid #EEEDE9; text-align:center;">Tarea</td><td style="padding: 8px 14px; border: 1px solid #EEEDE9;">Reto — listas de sanciones y regla dura (tarea para después de la sesión)</td></tr>
# MAGIC   </tbody>
# MAGIC </table>
# MAGIC </div>
# MAGIC </div>
# MAGIC
# MAGIC Empieza en `Modulo 2 - Agentes code-first/00_setup`. Los notebooks `01`–`04` traen el código completo: los ejecutamos y
# MAGIC explicamos juntos. El `05_reto` es la tarea para después de la sesión (su solución está en `solutions/`).

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## C. Requisitos
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 14px 18px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="color:#333;">
# MAGIC
# MAGIC - Conecta un **clúster clásico** (DBR **17.3 LTS**, modo de acceso *Standard*) para ejecutar los notebooks.
# MAGIC - En el widget **catalog** escribe el **mismo catálogo de la Sesión 1**. Si ya no tienes el índice de la
# MAGIC   Sesión 1, `00_setup` lo reconstruye (≈10–15 min).
# MAGIC - Usa **Chrome o Firefox** (no Safari), sin modo Incógnito.
# MAGIC - Servicios gestionados que deben estar habilitados (independientes del cómputo del notebook):
# MAGIC   **Foundation Model APIs** (un modelo de chat con *tool calling*), **AI Search**, y permiso para crear
# MAGIC   **funciones** y **modelos** en Unity Catalog. Se validan con el notebook de validación de entorno.
# MAGIC
# MAGIC </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. Todos los derechos reservados.
