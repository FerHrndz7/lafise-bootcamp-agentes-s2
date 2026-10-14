# Bootcamp de Agentes AML · Sesión 2: Agentes code-first sobre Databricks

Contenido práctico de la Sesión 2 del bootcamp de Agentes AML para el equipo de PLD de Grupo LAFISE: construir con
código el **Agente de Alertas AML** (funciones de Unity Catalog como herramientas, retriever de AI Search, LangGraph,
MLflow Tracing, supervisor con sub-agentes y registro del agente en Unity Catalog).

> Todos los datos del laboratorio son **100% sintéticos**. Nunca uses PII real en un workspace de laboratorio.

## Contenido

| Ruta | Qué es |
|---|---|
| `lafise-agentes-alertas-aml/00 - Course Overview` | Agenda y requisitos de la sesión |
| `lafise-agentes-alertas-aml/Modulo 2 - Agentes code-first/` | Notebooks `00_setup` a `04` (código completo) y `05_reto` (tarea), con `solutions/`, `data/` y `utils/` |
| `lafise-agentes-alertas-aml/requirements.txt` | Librerías (para instalarlas como librerías de clúster si no hay salida a PyPI) |
| `validacion/validacion_entorno_lafise_s2.py` | Notebook de validación de entorno |
| `build/lafise-agentes-alertas-aml-s2.dbc` | El mismo árbol de notebooks en un solo archivo, para importar en un clic |

## Cómo cargarlo en Databricks

**Opción A (recomendada): Git folder.** No requiere descargar archivos.
1. En el workspace: **Workspace → Create → Git folder**.
2. URL: `https://github.com/FerHrndz7/lafise-bootcamp-agentes-s2`, proveedor **GitHub** (repositorio público, no
   requiere credenciales).
3. Abrir `lafise-agentes-alertas-aml/00 - Course Overview`.

**Opción B: descargar el ZIP.** En GitHub: **Code → Download ZIP**, descomprimir e importar la carpeta
`lafise-agentes-alertas-aml` al workspace.

**Opción C: archivo `.dbc`.** Descargar `build/lafise-agentes-alertas-aml-s2.dbc` y subirlo con **Workspace → Import → File**
(importa todo el árbol en un clic). Si el firewall bloquea archivos `.dbc`, usar la opción A o B.

## Prueba completa antes de la sesión (30 a 40 min)

Con una **cuenta de asistente (no administrador)**, en un clúster clásico **DBR 17.3 LTS, modo de acceso Standard**:

1. `Modulo 2 - Agentes code-first/00_setup`: widget `catalog` = el catálogo de la Sesión 1; dejar `schema` = `pld_aml`
   y `llm` vacío. Anotar el **modelo de chat** que imprime. Si el índice de la Sesión 1 ya no existe, lo recrea
   (10 a 15 min).
2. `01` → `02` → `03` → `04`, con el mismo `catalog` en cada uno.

Resultado esperado: todas las celdas de validación en ✅ (por ejemplo, `AT-001 → REPORTAR_ROS`).
`05_reto` es la tarea para después de la sesión (su solución está en `solutions/`).
