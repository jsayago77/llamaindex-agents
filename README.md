# 🤖 Agentes con LlamaIndex

Colección de **agentes conversacionales** construidos con [LlamaIndex](https://www.llamaindex.ai/)
que combinan **herramientas** (acciones sobre datos tabulares) con **RAG** (recuperación
semántica) y un LLM de OpenAI.

> Proyecto de portfolio. La idea es mostrar **un mismo patrón de agente aplicado a dos
> dominios distintos**, en lugar de dos ejercicios sueltos.

## 🧠 El patrón (común a ambos bots)

```
Usuario ──▶ ReActAgent ──┬── FunctionTool ──▶ pandas  (datos tabulares)
                         ├── FunctionTool ──▶ VectorStoreIndex  (RAG sobre los CSV)
                         └── LLM (OpenAI)  ── decide qué herramienta usar y redacta
```

| Pieza | Rol |
|---|---|
| `ReActAgent` | Orquesta: razona y elige la herramienta adecuada |
| `FunctionTool` | Envuelve funciones Python (pandas) como acciones del agente |
| `VectorStoreIndex` + `SimpleDirectoryReader` | RAG: indexa los CSV para búsquedas en lenguaje natural |
| `OpenAI` (GPT-4 / GPT-3.5) | Modelo de lenguaje |
| `pandas` | Análisis de los datos |
| `python-dotenv` | Gestión de la `OPENAI_API_KEY` |

Cada bot es **independiente y ejecutable por separado** (su propio `README.md` y
`requirements.txt`).

## 📦 Bots

### 🏨 [bot-hoteleria](./bot-hoteleria)
Asistente de **consultas hoteleras**: verificación de disponibilidad, ingresos del mes y
habitaciones más reservadas. → [README](./bot-hoteleria/README.md)

### 🎓 [bot-estudiantes](./bot-estudiantes)
Asistente de **analítica académica**: promedios por materia y ranking de los mejores
estudiantes. → [README](./bot-estudiantes/README.md)

## 🚀 Cómo ejecutar

Ejemplo con `bot-hoteleria` (con `bot-estudiantes` es análogo):

```bash
cd bot-hoteleria
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
echo "OPENAI_API_KEY=sk-tu-clave-aqui" > .env
python agent-hotel.py
```

## ⚙️ Requisitos

- Python 3.10+
- Una `OPENAI_API_KEY` con crédito (se usan *embeddings* de OpenAI para el RAG).

## 📌 Notas

- Escritos contra la API de **LlamaIndex 0.11.x** (`ReActAgent.from_tools`).
  Si usás otra versión y la API cambió, ajustá las versiones en `requirements.txt`.
- Ambos agentes son de **solo lectura**: consultan datos, no los modifican.
- Los datos (`datasets/`, `data/`) son de **ejemplo** (ficticios).
