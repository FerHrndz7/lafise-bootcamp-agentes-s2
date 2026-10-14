# Databricks notebook source
# MAGIC %md
# MAGIC # Bootstrap — Base de conocimiento PLD + casos testigo (Sesión 2)
# MAGIC
# MAGIC Lo llama `00_setup`. Deja tu schema `pld_aml_<usuario>` listo para la Sesión 2, **reutilizando lo que ya
# MAGIC construiste en la Sesión 1** y regenerando solo lo que falte:
# MAGIC
# MAGIC | Pieza | Si ya existe (Sesión 1) | Si falta |
# MAGIC |---|---|---|
# MAGIC | `clientes`, `transacciones`, `alertas`, `narrativas_casos` | se reutilizan | se generan con Faker (mismo generador de la S1) |
# MAGIC | `chunks` | se reutiliza | se genera desde el corpus PLD (un chunk por artículo/sección, **ID determinístico**) |
# MAGIC | Índice `chunks_index` (AI Search) | se reutiliza | se crea (Delta Sync, embeddings gestionados) |
# MAGIC | **Casos testigo** `AT-001..003` + `lista_sanciones` | se refrescan (idempotente) | se crean |
# MAGIC
# MAGIC > ⚠️ **Todo es SINTÉTICO** (Faker + hechos regulatorios plausibles pero ficticios). Nunca PII real.

# COMMAND ----------

dbutils.widgets.text("catalog", "", "Catalog")
dbutils.widgets.text("schema", "pld_aml", "Schema base")
dbutils.widgets.text("num_clientes", "500", "Num clientes")

# COMMAND ----------

import re, json, random, time
from datetime import date, timedelta

CATALOG = dbutils.widgets.get("catalog").strip()
NUM_CLIENTES = int(dbutils.widgets.get("num_clientes"))
user_email = spark.sql("SELECT current_user()").collect()[0][0]
user_prefix = re.sub(r"[^a-z0-9]", "_", user_email.split("@")[0].lower())
assert CATALOG, "Falta el catálogo. Ejecuta este bootstrap desde 00_setup (o pásale el widget 'catalog')."
SCHEMA = f"{(dbutils.widgets.get('schema').strip() or 'pld_aml')}_{user_prefix}"
FQ = f"`{CATALOG}`.{SCHEMA}"

VS_ENDPOINT = "lafise_vs_endpoint"                  # mismo endpoint compartido de la Sesión 1
INDEX_NAME = f"{CATALOG}.{SCHEMA}.chunks_index"

spark.sql(f"USE CATALOG `{CATALOG}`")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {FQ}")
random.seed(42)

def existe_tabla(t):
    return spark.catalog.tableExists(f"{CATALOG}.{SCHEMA}.{t}")

print(f"Destino: {CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Corpus PLD (idéntico al de la Sesión 1, para que las citas coincidan)

# COMMAND ----------

POLITICA = (
    "politica_pld_lafise", "pdf",
    "Política de Prevención de Lavado de Dinero y Financiamiento al Terrorismo (PLD/FT) — Grupo LAFISE",
    [
        ("Artículo 1. Objeto y marco",
         ["La presente política establece el marco de prevención de lavado de dinero y financiamiento al "
          "terrorismo (PLD/FT) de la institución, alineado con las 40 Recomendaciones del GAFI y la normativa "
          "de la Unidad de Análisis Financiero (UAF). Su cumplimiento es obligatorio para todo colaborador."]),
        ("Artículo 2. Enfoque basado en riesgo",
         ["La institución adopta un enfoque basado en riesgo (EBR): asigna a cada cliente un nivel de riesgo "
          "(bajo, medio, alto) según su perfil, actividad económica, geografía y canal, y calibra la intensidad "
          "de la debida diligencia en función de ese nivel."]),
        ("Artículo 3. Debida diligencia del cliente (DDC)",
         ["Se aplica debida diligencia estándar a clientes de riesgo bajo o medio, y debida diligencia "
          "reforzada (EDD) a clientes de alto riesgo. La identificación del beneficiario final es obligatoria "
          "en todos los casos."]),
        ("Artículo 4. Personas Expuestas Políticamente (PEP)",
         ["Toda relación con una Persona Expuesta Políticamente (PEP) requiere debida diligencia reforzada y "
          "la aprobación expresa de la alta gerencia antes de abrir o mantener la cuenta. El monitoreo de las "
          "PEP es continuo."]),
        ("Artículo 5. Reporte de Transacciones en Efectivo (RTE)",
         ["Toda transacción en efectivo igual o superior a USD 10,000 —individual o acumulada por un mismo "
          "cliente durante un mes calendario— genera un Reporte de Transacción en Efectivo (RTE) que se remite "
          "a la UAF. El fraccionamiento deliberado para evadir este umbral constituye una señal de alerta."]),
        ("Artículo 6. Reporte de Operación Sospechosa (ROS)",
         ["Cuando el análisis confirma indicios de una operación sospechosa, el Oficial de Cumplimiento remite "
          "un Reporte de Operación Sospechosa (ROS) a la UAF dentro de un plazo máximo de 3 días hábiles "
          "contados desde la confirmación. El ROS es confidencial: está prohibido informar al cliente (tipping-off)."]),
        ("Artículo 7. Conservación de registros",
         ["Los registros de identificación de clientes, transacciones y análisis de alertas se conservan por un "
          "plazo mínimo de 5 años contados desde el cierre de la relación o la fecha de la operación."]),
        ("Artículo 8. Monitoreo transaccional",
         ["El monitoreo transaccional es automatizado y basado en reglas y tipologías. Genera alertas que el "
          "área de cumplimiento analiza y dispone según el Instructivo de Escalamiento y Disposición de Alertas."]),
        ("Artículo 9. Listas de sanciones",
         ["La institución coteja clientes y contrapartes contra listas de sanciones (OFAC, Consejo de Seguridad "
          "de la ONU y listas locales). Una coincidencia (hit) confirmada bloquea la operación y se escala de "
          "inmediato; nunca se cierra sin acción."]),
    ],
)

TIPOLOGIAS = (
    "manual_tipologias", "pdf",
    "Manual de Tipologías de Lavado de Activos — Cumplimiento LAFISE",
    [
        ("Tipología 1. Estructuración (pitufeo)",
         ["Descripción: fraccionar una suma grande en múltiples depósitos o transferencias menores al umbral de "
          "USD 10,000 para evadir el Reporte de Transacción en Efectivo.",
          "Señales de alerta: múltiples depósitos en efectivo por montos apenas por debajo de USD 10,000; uso de "
          "varias sucursales el mismo día; depósitos por terceros en la cuenta de una misma persona."]),
        ("Tipología 2. Movimiento rápido de fondos (pass-through)",
         ["Descripción: fondos que ingresan y salen de una cuenta en un lapso muy corto (menos de 48 a 72 horas), "
          "dejando un saldo cercano a cero, sin una razón económica aparente.",
          "Señales de alerta: la cuenta funciona como puente; entradas y salidas del mismo monto; beneficiario "
          "final distinto del titular."]),
        ("Tipología 3. Uso de terceros y testaferros",
         ["Descripción: uso de terceros (familiares, empleados, sociedades pantalla) para ocultar al verdadero "
          "beneficiario final de los fondos.",
          "Señales de alerta: cuentas de personas sin capacidad económica que reciben montos altos; poderes "
          "amplios a terceros; sociedades recién constituidas sin actividad real."]),
        ("Tipología 4. Operaciones inconsistentes con el perfil",
         ["Descripción: transacciones cuyo monto, frecuencia o naturaleza no corresponden con el perfil "
          "transaccional declarado ni con la actividad económica del cliente.",
          "Señales de alerta: un cliente de perfil bajo que de pronto mueve montos altos; giros al exterior sin "
          "relación con su negocio; cambios abruptos de comportamiento."]),
        ("Tipología 5. Geografías de alto riesgo",
         ["Descripción: operaciones con origen o destino en jurisdicciones de alto riesgo o no cooperantes según "
          "el GAFI, o en zonas de frontera asociadas a economías ilícitas.",
          "Señales de alerta: transferencias frecuentes hacia países de alto riesgo; uso de centros financieros "
          "extraterritoriales sin justificación; ruteo innecesario de fondos por múltiples jurisdicciones."]),
    ],
)

INSTRUCTIVO = (
    "instructivo_alertas", "pdf",
    "Instructivo de Escalamiento y Disposición de Alertas — Unidad PLD",
    [
        ("1. Alcance",
         ["Este instructivo define cómo el analista de la Unidad de Cumplimiento analiza y dispone las alertas "
          "generadas por el monitoreo transaccional."]),
        ("2. Plazo de análisis (SLA)",
         ["Cada alerta debe analizarse y disponerse en un plazo máximo de 15 días hábiles desde su generación. "
          "Las alertas de prioridad alta (score de riesgo elevado o hit en lista) se atienden en 24 horas."]),
        ("3. Disposiciones posibles",
         ["(a) Cerrar sin acción: cuando el análisis descarta el indicio y la operación es consistente con el "
          "perfil. Debe documentarse el fundamento.",
          "(b) Escalar a investigación: cuando persisten indicios que requieren profundización por la unidad de "
          "investigaciones.",
          "(c) Reportar: cuando se confirma la sospecha, se genera un ROS a la UAF dentro de 3 días hábiles."]),
        ("4. Reglas duras",
         ["Un hit confirmado en lista de sanciones (OFAC/ONU) nunca se cierra sin acción: se bloquea y se escala "
          "de inmediato. Toda disposición debe citar la política o la tipología que la fundamenta."]),
        ("5. Priorización",
         ["Las alertas se priorizan por score de riesgo, monto involucrado, y presencia de PEP o de geografías de "
          "alto riesgo."]),
    ],
)


DOCS = [POLITICA, TIPOLOGIAS, INSTRUCTIVO]

NARRATIVAS = [
    ("K-2024-018", "Estructuración", "Reportado",
     "El cliente realizó nueve depósitos en efectivo de entre USD 9,200 y USD 9,800 en tres sucursales "
     "distintas durante cinco días. El patrón es consistente con estructuración para evadir el umbral de RTE. "
     "Se confirmó la sospecha y se remitió ROS a la UAF. Fundamento: Tipología 1 y Artículo 5 de la política."),
    ("K-2024-042", "Pass-through", "Escalado",
     "Una cuenta empresarial recibió tres transferencias del exterior por USD 45,000 que salieron hacia una "
     "tercera cuenta en menos de 48 horas, dejando saldo cero. No hay soporte comercial. Movimiento rápido de "
     "fondos; se escaló a investigación para identificar al beneficiario final."),
    ("K-2023-233", "Inconsistente con perfil", "Cerrado",
     "Un cliente de banca de personas con perfil bajo recibió USD 30,000. Tras el análisis se documentó que "
     "correspondía a la venta de un inmueble con escritura verificable. Operación consistente; se cerró sin acción."),
    ("K-2024-107", "Geografía de alto riesgo", "Reportado",
     "Transferencias frecuentes hacia una jurisdicción de alto riesgo listada por el GAFI, sin relación con la "
     "actividad declarada del cliente. Se confirmó la sospecha y se reportó a la UAF."),
    ("K-2024-151", "Testaferros", "Escalado",
     "Depósitos altos en la cuenta de una persona sin capacidad económica aparente, con poder amplio otorgado a "
     "un tercero. Posible uso de testaferro; se escaló a investigación."),
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Tablas estructuradas (solo si faltan)

# COMMAND ----------

from pyspark.sql import Row

if not existe_tabla("alertas"):
    from faker import Faker
    fake = Faker("es_MX"); Faker.seed(42)
    SEGMENTOS = ["Banca de personas", "Banca de empresas", "Comercio exterior"]
    TIPOLOGIAS_CAT = ["Estructuración", "Pass-through", "Testaferros",
                      "Inconsistente con perfil", "Geografía de alto riesgo"]
    hoy = date.today()
    clientes = [Row(cliente_id=f"C{i:05d}", nombre=fake.name(), segmento=random.choice(SEGMENTOS),
                    pais=random.choice(["Nicaragua", "Costa Rica", "Honduras", "Panamá", "Guatemala"]),
                    es_pep=random.random() < 0.05,
                    nivel_riesgo=random.choice(["Alto", "Alto", "Medio", "Bajo", "Bajo", "Bajo"]),
                    fecha_alta=fake.date_between(start_date="-5y", end_date="today").isoformat())
                for i in range(NUM_CLIENTES)]
    spark.createDataFrame(clientes).write.mode("overwrite").saveAsTable(f"{FQ}.clientes")
    txns = []
    for i in range(NUM_CLIENTES * 8):
        monto = round(random.choice([random.uniform(50, 3000), random.uniform(9000, 9999),
                                     random.uniform(100, 50000)]), 2)
        txns.append(Row(txn_id=f"T{i:07d}", cliente_id=f"C{random.randint(0, NUM_CLIENTES-1):05d}",
                        fecha=(hoy - timedelta(days=random.randint(0, 180))).isoformat(), monto_usd=monto,
                        canal=random.choice(["Efectivo", "Transferencia", "ACH", "Cheque"]),
                        pais_contraparte=random.choice(["Nicaragua", "Panamá", "EE.UU.", "Islas Caimán", "Colombia"])))
    spark.createDataFrame(txns).write.mode("overwrite").saveAsTable(f"{FQ}.transacciones")
    alertas = [Row(alerta_id=f"A{i:05d}", cliente_id=f"C{random.randint(0, NUM_CLIENTES-1):05d}",
                   fecha=(hoy - timedelta(days=random.randint(0, 60))).isoformat(),
                   tipologia=random.choice(TIPOLOGIAS_CAT), score_riesgo=random.randint(30, 99),
                   monto_involucrado_usd=round(random.uniform(8000, 120000), 2),
                   estado=random.choice(["Abierta", "En análisis", "Cerrada", "Escalada", "Reportada"]))
               for i in range(120)]
    spark.createDataFrame(alertas).write.mode("overwrite").saveAsTable(f"{FQ}.alertas")
    print("✅ Tablas clientes/transacciones/alertas generadas (no venían de la Sesión 1)")
else:
    print("♻️  Reutilizando clientes/transacciones/alertas de la Sesión 1")

if not existe_tabla("narrativas_casos"):
    spark.createDataFrame([Row(caso_id=c[0], tipologia=c[1], disposicion=c[2], narrativa=c[3]) for c in NARRATIVAS]) \
        .write.mode("overwrite").saveAsTable(f"{FQ}.narrativas_casos")
    print("✅ narrativas_casos generada")
else:
    print("♻️  Reutilizando narrativas_casos de la Sesión 1")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Casos testigo (Sesión 2) + lista de sanciones
# MAGIC
# MAGIC Tres alertas **diseñadas a mano** con un comportamiento esperado conocido. Las usaremos para comprobar
# MAGIC que el agente razona bien (y, en la Sesión 4, como semilla del set de evaluación):
# MAGIC
# MAGIC | Alerta | Patrón | Disposición esperada |
# MAGIC |---|---|---|
# MAGIC | `AT-001` | 9 depósitos en efectivo de USD 9,2xx–9,8xx en 5 días (estructuración) | `REPORTAR_ROS` (o `ESCALAR_INVESTIGACION`) |
# MAGIC | `AT-002` | Cliente de riesgo bajo, movimientos pequeños y consistentes | `CERRAR_SIN_ACCION` (o `SOLICITAR_INFORMACION`) |
# MAGIC | `AT-003` | Movimiento moderado, pero el cliente **coincide con la lista de sanciones** | `ESCALAR_INVESTIGACION` — nunca cerrar (regla dura) |

# COMMAND ----------

hoy = date.today()
d = lambda n: (hoy - timedelta(days=n)).isoformat()

CLIENTES_T = [
    ("CT-001", "Marvin Antonio Zelaya Ríos", "Banca de personas", "Nicaragua", False, "Medio", "2021-03-15"),
    ("CT-002", "Lucía Fernanda Castillo Mena", "Banca de personas", "Costa Rica", False, "Bajo", "2016-08-02"),
    ("CT-003", "Ramiro Quintanilla Rojas", "Banca de empresas", "Honduras", False, "Medio", "2023-11-20"),
]
TXNS_T = (
    [(f"TT-001-{i:02d}", "CT-001", d(9 - i // 2), float(9200 + 70 * i), "Efectivo", "Nicaragua") for i in range(9)]
    + [("TT-001-90", "CT-001", d(40), 850.00, "Transferencia", "Nicaragua")]
    + [(f"TT-002-{i:02d}", "CT-002", d(5 + 9 * i), float(1200 + 35 * i), "ACH", "Costa Rica") for i in range(6)]
    + [("TT-003-01", "CT-003", d(12), 18500.00, "Transferencia", "Panamá"),
       ("TT-003-02", "CT-003", d(6), 7400.00, "Transferencia", "Islas Caimán")]
)
ALERTAS_T = [
    ("AT-001", "CT-001", d(2), "Estructuración", 91, 85320.00, "Abierta"),
    ("AT-002", "CT-002", d(2), "Inconsistente con perfil", 38, 3705.00, "Abierta"),
    ("AT-003", "CT-003", d(2), "Geografía de alto riesgo", 64, 25900.00, "Abierta"),
]

def refrescar(tabla, filas, cols, clave, prefijo):
    """Borra las filas testigo previas (por prefijo) y vuelve a insertarlas -> idempotente."""
    destino = spark.table(f"{FQ}.{tabla}")
    df = spark.createDataFrame(filas, cols)
    df = df.select([df[c].cast(destino.schema[c].dataType).alias(c) for c in destino.columns])
    spark.sql(f"DELETE FROM {FQ}.{tabla} WHERE {clave} LIKE '{prefijo}%'")
    df.write.mode("append").saveAsTable(f"{FQ}.{tabla}")

refrescar("clientes", CLIENTES_T,
          ["cliente_id", "nombre", "segmento", "pais", "es_pep", "nivel_riesgo", "fecha_alta"], "cliente_id", "CT-")
refrescar("transacciones", TXNS_T,
          ["txn_id", "cliente_id", "fecha", "monto_usd", "canal", "pais_contraparte"], "txn_id", "TT-")
refrescar("alertas", ALERTAS_T,
          ["alerta_id", "cliente_id", "fecha", "tipologia", "score_riesgo", "monto_involucrado_usd", "estado"],
          "alerta_id", "AT-")

SANCIONES = [
    ("Ramiro Quintanilla Rojas", "OFAC SDN", "Honduras", "2024-05-10", "Facilitador de redes de lavado"),
    ("Inversiones Costa Azul S.A.", "OFAC SDN", "Panamá", "2023-09-01", "Sociedad pantalla"),
    ("Héctor Iván Montalvo", "ONU CSNU", "Guatemala", "2022-02-14", "Financiamiento al terrorismo"),
    ("Grupo Comercial Tres Ríos", "Lista local UAF", "Nicaragua", "2025-01-20", "Contrabando"),
]
spark.createDataFrame(SANCIONES, ["nombre_listado", "lista", "pais", "fecha_inclusion", "motivo"]) \
    .write.mode("overwrite").saveAsTable(f"{FQ}.lista_sanciones")
print("✅ Casos testigo AT-001..003 + lista_sanciones listos")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Tabla `chunks` (solo si falta)
# MAGIC Un chunk por artículo/sección con su metadata de citación. El `chunk_id` es **determinístico**
# MAGIC (`sha2(source_doc | articulo | n)`): si el documento cambia, el índice actualiza solo lo que cambió.

# COMMAND ----------

from pyspark.sql import functions as F

if not existe_tabla("chunks"):
    filas = []
    for name, fmt, title, sections in DOCS:
        doc = f"{name}.{fmt}"
        for heading, paragraphs in sections:
            m = re.match(r"^(Artículo\s+\d+|Tipología\s+\d+|\d+)", heading)
            articulo = m.group(0) if m else heading[:40]
            texto = heading + "\n" + "\n".join(paragraphs)
            filas.append((doc, heading, articulo, texto, f"[{doc} · {heading}] {texto}"))
    for c in NARRATIVAS:
        txt = f"Caso {c[0]} — Tipología {c[1]} — Disposición: {c[2]}. {c[3]}"
        filas.append(("narrativas_casos", f"Caso {c[0]}", c[1], txt, f"[Precedente · {c[1]}] {txt}"))
    df = (spark.createDataFrame(filas, ["source_doc", "seccion", "articulo", "chunk_to_retrieve", "chunk_to_embed"])
          .withColumn("chunk_id", F.sha2(F.concat_ws("|", "source_doc", "seccion"), 256)))
    df.write.saveAsTable(f"{FQ}.chunks")
    spark.sql(f"ALTER TABLE {FQ}.chunks SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
    print(f"✅ chunks generada ({df.count()} filas)")
else:
    print("♻️  Reutilizando chunks de la Sesión 1")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Índice de AI Search (solo si falta)
# MAGIC Si ya tienes `chunks_index` de la Sesión 1 lo reutilizamos tal cual. Si no, lo creamos y esperamos a que
# MAGIC esté listo (la primera sincronización tarda ~5–15 min: es el arranque del pipeline gestionado, no los
# MAGIC embeddings).

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import (
    DeltaSyncVectorIndexSpecRequest, EmbeddingSourceColumn, PipelineType, VectorIndexType, EndpointType)
w = WorkspaceClient()

def indice_existe():
    try:
        w.vector_search_indexes.get_index(INDEX_NAME)
        return True
    except Exception:
        return False

if indice_existe():
    print(f"♻️  Reutilizando índice {INDEX_NAME}")
else:
    servidos = {e.name for e in w.serving_endpoints.list()}
    EMB = next(c for c in ["databricks-qwen3-embedding-0-6b", "databricks-gte-large-en", "databricks-bge-large-en"]
               if c in servidos)
    if VS_ENDPOINT not in {e.name for e in w.vector_search_endpoints.list_endpoints()}:
        print(f"Creando endpoint {VS_ENDPOINT} (5–15 min)...")
        w.vector_search_endpoints.create_endpoint_and_wait(name=VS_ENDPOINT, endpoint_type=EndpointType.STANDARD)
    w.vector_search_indexes.create_index(
        name=INDEX_NAME, endpoint_name=VS_ENDPOINT, primary_key="chunk_id",
        index_type=VectorIndexType.DELTA_SYNC,
        delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
            source_table=f"{CATALOG}.{SCHEMA}.chunks", pipeline_type=PipelineType.TRIGGERED,
            embedding_source_columns=[EmbeddingSourceColumn(name="chunk_to_embed",
                                                            embedding_model_endpoint_name=EMB)]))
    print(f"Índice creado con {EMB}; esperando a que esté listo...")
    deadline = time.time() + 30 * 60
    while time.time() < deadline:
        try:
            if getattr(w.vector_search_indexes.get_index(INDEX_NAME).status, "ready", False):
                break
        except Exception:
            pass                                   # consistencia eventual justo después de crear
        time.sleep(30)
    print("✅ Índice listo" if indice_existe() and w.vector_search_indexes.get_index(INDEX_NAME).status.ready
          else "⚠️ El índice aún no está listo; continúa y vuelve a probar en unos minutos.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Verificación

# COMMAND ----------

for t in ["clientes", "transacciones", "alertas", "narrativas_casos", "lista_sanciones", "chunks"]:
    print(f"  tabla {t:18s} {spark.table(f'{FQ}.{t}').count():>8,} filas")
print(f"  índice {INDEX_NAME}  ready={w.vector_search_indexes.get_index(INDEX_NAME).status.ready}")
