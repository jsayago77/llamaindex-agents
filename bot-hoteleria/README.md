# 🏨 Bot Hotelería — Asistente de consultas

> Parte de la colección [**llamaindex-agents**](../README.md).

Agente conversacional (CLI) que responde preguntas sobre las reservas de un hotel,
usando **LlamaIndex** (agente ReAct + RAG) y **OpenAI**.

> Proyecto académico — demuestra el uso de **agentes con herramientas**, **RAG**
> y análisis de datos con **pandas** integrados con un LLM.

## ✨ Qué hace

- ✅ Verifica **disponibilidad** de una habitación en una fecha.
- 🏆 Muestra las **5 habitaciones más reservadas**.
- 💰 Calcula los **ingresos** de un mes (solo reservas confirmadas).
- 🔍 Realiza **búsquedas en lenguaje natural** sobre los datos (RAG).

## 🧠 Arquitectura

| Componente | Rol |
|---|---|
| `ReActAgent` + `FunctionTool` | El LLM decide qué herramienta usar para responder |
| `VectorStoreIndex` (RAG) | Indexa los CSV para búsquedas semánticas |
| `pandas` | Limpieza, merge y análisis de los datos |
| `OpenAI` (GPT-4) | Modelo de lenguaje |
| `python-dotenv` | Carga de la API key desde `.env` |

Datos de ejemplo en `datasets/`: `reservas.csv` (47 reservas) y `habitaciones.csv` (9 habitaciones).

## 📂 Estructura

```
bot-hoteleria/
├── agent-hotel.py          # script principal
├── requirements.txt
├── datasets/
│   ├── reservas.csv
│   └── habitaciones.csv
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
python agent-hotel.py
```

## 💬 Ejemplo de uso

```
Tú: ¿Está disponible la habitación 101 el 2025-02-01?
Bot: La habitación 101 está disponible el 2025-02-01. ✅

Tú: ¿Cuáles son las 5 habitaciones más reservadas?
Bot: ...

Tú: salir
```

## ⚠️ Requisitos

- Python 3.10+
- Una `OPENAI_API_KEY` con crédito (usa GPT-4 y *embeddings* de OpenAI para el RAG).

## 📌 Notas

- El bot es de **solo consulta**: no crea ni modifica reservas.
- Escrito contra la API de **LlamaIndex 0.11.x** (`ReActAgent.from_tools`).
  Si usás otra versión y la API cambió, ajustá las versiones en `requirements.txt`.
- Los datos son de **ejemplo** (ficticios).
