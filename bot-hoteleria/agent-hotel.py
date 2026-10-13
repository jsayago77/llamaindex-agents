"""
Asistente de consultas para un hotel (bot "Hector").

Agente conversacional construido con LlamaIndex (ReActAgent) que combina:
  - Herramientas (FunctionTool) sobre datos tabulares (pandas).
  - Recuperacion aumentada (RAG) sobre los CSV indexados (VectorStoreIndex).
  - Un LLM de OpenAI.

Proyecto academico. Ver README.md para instrucciones de uso.
"""
from pathlib import Path
from datetime import datetime
import sys

import pandas as pd
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI

load_dotenv()  # carga OPENAI_API_KEY desde el archivo .env

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "datasets"


def load_csv_data():
    """Carga los CSV en pandas y los indexa para las busquedas semanticas."""
    try:
        reservas = pd.read_csv(DATA_DIR / "reservas.csv")
        habitaciones = pd.read_csv(DATA_DIR / "habitaciones.csv")

        # Combinamos las reservas con los datos de cada habitacion
        df = pd.merge(reservas, habitaciones, on="Habitacion", how="left")
        df["Fecha"] = pd.to_datetime(df["Fecha"])

        # Indexamos los documentos de la carpeta para RAG
        documents = SimpleDirectoryReader(str(DATA_DIR)).load_data()
        vector_index = VectorStoreIndex.from_documents(documents)

        return df, vector_index.as_query_engine()
    except Exception as e:
        print(f"Error cargando datos: {e}")
        sys.exit(1)


# Datos cargados en memoria al iniciar
df, query_engine = load_csv_data()


# ------------------------- Herramientas del agente -------------------------

def obtener_fecha_actual():
    """Devuelve la fecha actual en formato YYYY-MM-DD."""
    return datetime.now().strftime("%Y-%m-%d")


def verificar_disponibilidad(habitacion: str, fecha: str):
    """Indica si una habitacion esta disponible en una fecha (YYYY-MM-DD)."""
    fecha_dt = pd.to_datetime(fecha)
    reservas = df[(df["Habitacion"] == int(habitacion)) & (df["Fecha"] == fecha_dt)]
    return bool(reservas.empty)


def habitaciones_max():
    """Devuelve las 5 habitaciones con mas reservas."""
    return df["Habitacion"].value_counts().nlargest(5).to_dict()


def calcular_ingresos(mes: str):
    """Calcula los ingresos de las reservas confirmadas de un mes (1-12)."""
    reservas_mes = df[(df["Fecha"].dt.month == int(mes)) & (df["Estado"] == "Confirmada")]
    ingresos = (reservas_mes["Precio"] * reservas_mes["Duracion"]).sum()
    return f"${ingresos:.2f}"


def search_data(query: str):
    """Busca informacion en los datos indexados usando lenguaje natural."""
    return query_engine.query(query)


tools = [
    FunctionTool.from_defaults(
        obtener_fecha_actual,
        name="obtener_fecha_actual",
        description="Obtiene la fecha actual en formato YYYY-MM-DD.",
    ),
    FunctionTool.from_defaults(
        verificar_disponibilidad,
        name="verificar_disponibilidad",
        description="Verifica si una habitacion (numero) esta disponible en una fecha (YYYY-MM-DD).",
    ),
    FunctionTool.from_defaults(
        habitaciones_max,
        name="habitaciones_max",
        description="Obtiene las 5 habitaciones mas reservadas.",
    ),
    FunctionTool.from_defaults(
        calcular_ingresos,
        name="calcular_ingresos",
        description="Calcula los ingresos de un mes (1-12), solo reservas confirmadas.",
    ),
    FunctionTool.from_defaults(
        search_data,
        name="buscar_reservas",
        description="Busca reservas por habitacion, fecha o criterios personalizados.",
    ),
]

INSTRUCCIONES = """\
Eres "Hector", un asistente de consultas de un hotel. Ayudas a los usuarios a
consultar disponibilidad, habitaciones mas reservadas, ingresos estimados y a
buscar informacion en las reservas. Responde de forma amable y clara, usando emojis.

Funciones disponibles:
1. Verificar disponibilidad de una habitacion en una fecha.
2. Listar las 5 habitaciones mas reservadas.
3. Calcular los ingresos de un mes (reservas confirmadas).
4. Buscar reservas por habitacion, fecha o criterios personalizados.

Reglas:
- Usa 2 decimales en los resultados numericos.
- Formatea los resultados con encabezados claros.
- Solo entregas INFORMACION: no creas ni modificas reservas. Si te lo piden, acláralo.

Ejemplo:
Usuario: "¿Esta disponible la habitacion 101 el 2025-02-01?"
Bot: "La habitacion 101 esta disponible el 2025-02-01. ✅"
"""


def query_validation(query):
    """Valida la consulta del usuario."""
    if not query.strip():
        return "Por favor ingresa una consulta válida"
    if len(query) > 200:
        return "Consulta demasiado larga (máx. 200 caracteres)"
    return None


if __name__ == "__main__":
    llm = OpenAI(model="gpt-4", temperature=0, max_tokens=500)

    agent = ReActAgent.from_tools(
        tools=tools,
        llm=llm,
        verbose=False,
        system_prompt=INSTRUCCIONES,
    )

    print("----- 🏨 Asistente de Consultas de Hotel -----")
    print("Bot: ¡Hola! Soy Héctor, tu asistente de consultas de hotel. ¿En qué puedo ayudarte?\n")

    while True:
        try:
            query = input("Tú: ")

            if query.lower() == "salir":
                break

            if error := query_validation(query):
                print(f"\n⚠️ {error}\n")
                continue

            response = agent.query(query)
            print(f"\nBot: {response}\n")

        except KeyboardInterrupt:
            print("\n\n⚠️ Sesión interrumpida\n")
            break

    print("\n¡Hasta pronto! 🏨✨")
