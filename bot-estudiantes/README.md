# 🎓 Bot Estudiantes — Asistente analítico académico

> Parte de la colección [**llamaindex-agents**](../README.md).

Agente conversacional (CLI) que responde preguntas sobre las notas de una sección,
usando **LlamaIndex** (agente ReAct + RAG) y **OpenAI**.

> Proyecto académico — demuestra el uso de **agentes con herramientas**, **RAG**
> y análisis de datos con **pandas** integrados con un LLM.

## ✨ Qué hace

- 📊 Calcula el **promedio general** por materia.
- 🏆 Genera el **ranking de los 3 mejores promedios**.
- 🔍 Realiza **búsquedas en lenguaje natural** sobre los estudiantes (RAG).

## 🧠 Arquitectura

| Componente | Rol |
|---|---|
| `ReActAgent` + `FunctionTool` | El LLM decide qué herramienta usar para responder |
| `VectorStoreIndex` (RAG) | Indexa el CSV para búsquedas semánticas |
| `pandas` | Cálculo de promedios y rankings |
| `OpenAI` (GPT-3.5) | Modelo de lenguaje |
| `python-dotenv` | Carga de la API key desde `.env` |

Datos de ejemplo en `data/estudiantes.csv` (9 estudiantes, 5 materias).

## 📂 Estructura

```
bot-estudiantes/
├── studentsAgent.py        # script principal
├── requirements.txt
├── data/
│   └── estudiantes.csv
└── .env                    # NO versionado  →  OPENAI_API_KEY=sk-...
```

## 🚀 Cómo ejecutar

```bash
# 1) Entorno virtual
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2) Dependencias
pip install -r requirements.txt

# 3) API key de OpenAI
echo "OPENAI_API_KEY=sk-tu-clave-aqui" > .env

# 4) Ejecutar
python studentsAgent.py
```

## 💬 Ejemplo de uso

```
Bot: 🎓 Asistente Académico - Sección A 5to Año
     Escribe 'ayuda' para opciones o 'salir' para terminar

Tú: ¿Cuál es el promedio general?
Bot: 📊 Promedio General: 85.20

Tú: ¿Quiénes son los 3 mejores?
Bot: 🏆 Top 3 Estudiantes: ...

Tú: salir
```

## ⚠️ Requisitos

- Python 3.10+
- Una `OPENAI_API_KEY` con crédito (usa GPT-3.5 y *embeddings* de OpenAI para el RAG).

## 📌 Notas

- Escrito contra la API de **LlamaIndex 0.11.x** (`ReActAgent.from_tools`).
  Si usás otra versión y la API cambió, ajustá las versiones en `requirements.txt`.
- Los datos son de **ejemplo** (ficticios) y representan datos sensibles:
  en un caso real habría que tratar la privacidad de los estudiantes.
