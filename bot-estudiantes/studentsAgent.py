"""
Asistente analitico academico (seccion A, 5to anio).

Agente conversacional construido con LlamaIndex (ReActAgent) que combina:
  - Herramientas (FunctionTool) sobre las notas de los estudiantes (pandas).
  - Recuperacion aumentada (RAG) sobre el CSV indexado (VectorStoreIndex).
  - Un LLM de OpenAI.

Proyecto academico. Ver README.md para instrucciones de uso.
"""
from pathlib import Path
import sys

import pandas as pd
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI

load_dotenv()  # carga OPENAI_API_KEY desde el archivo .env

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

MATERIAS = ["Historia", "Biologia", "Quimica", "Matematicas", "Geografia"]


def load_data():
    """Carga las notas en pandas y el CSV indexado para RAG."""
    try:
        df = pd.read_csv(DATA_DIR / "estudiantes.csv")
        df["Promedio"] = df[MATERIAS].mean(axis=1)

        # Indexamos los documentos de la carpeta para RAG
        documents = SimpleDirectoryReader(str(DATA_DIR)).load_data()
        vector_index = VectorStoreIndex.from_documents(documents)

        return df, vector_index.as_query_engine()
    except Exception as e:
        print(f"Error cargando datos: {e}")
        sys.exit(1)


def build_functions(df, query_engine):
    """Crea las funciones que el agente usara como herramientas."""

    def general_mean():
        """Calcula el promedio general por materia."""
        return df[MATERIAS].mean().round(2).to_dict()

    def top_students():
        """Devuelve los 3 estudiantes con mejor promedio."""
        return df.nlargest(3, "Promedio")[["Nombre", "Promedio"]].round(2).to_dict("records")

    def search_data(query: str):
        """Busca estudiantes por nombre, notas o criterios personalizados."""
        return query_engine.query(query)

    return general_mean, top_students, search_data


def create_tools(general_mean, top_students, search_data):
    """Envuelve las funciones como herramientas del agente."""
    return [
        FunctionTool.from_defaults(
            general_mean,
            name="calcular_promedios",
            description="Calcula el promedio general por materia.",
        ),
        FunctionTool.from_defaults(
            top_students,
            name="ranking_estudiantes",
            description="Genera el ranking de los 3 mejores promedios.",
        ),
        FunctionTool.from_defaults(
            search_data,
            name="buscar_estudiantes",
            description="Busca estudiantes por nombre, notas o criterios personalizados.",
        ),
    ]


INSTRUCCIONES = """\
Eres un asistente analitico de la seccion A de 5to anio. Ayudas a consultar
promedios, rankings y a buscar estudiantes.

Funciones disponibles:
1. Calculo de promedios por materia.
2. Ranking de los 3 mejores promedios.
3. Busquedas por nombre, notas o criterios personalizados.

Reglas:
- Usa 2 decimales en los resultados numericos.
- Formatea los resultados con encabezados claros.
- Cuida la privacidad de los datos de los estudiantes.

Ejemplo:
Usuario: "¿Cual es el promedio general?"
Bot: "El promedio general es 85.20. 📊"
"""


def format_response(response):
    """Anade un encabezado visual segun el tipo de respuesta."""
    mejoras = {
        "promedio general": "📊 Promedio General",
        "top 3": "🏆 Top 3 Estudiantes",
        "resultados de búsqueda": "🔍 Resultados",
    }
    for k, v in mejoras.items():
        if k in response.lower():
            return f"{v}:\n{response.split(':')[-1].strip()}"
    return response


def query_validation(query):
    """Valida la consulta del usuario."""
    if not query.strip():
        return "Por favor ingresa una consulta válida"
    if len(query) > 200:
        return "Consulta demasiado larga (máx. 200 caracteres)"
    return None


if __name__ == "__main__":
    df, query_engine = load_data()
    general_mean, top_students, search_data = build_functions(df, query_engine)
    agent_tools = create_tools(general_mean, top_students, search_data)

    llm = OpenAI(model="gpt-3.5-turbo")

    agent = ReActAgent.from_tools(
        tools=agent_tools,
        llm=llm,
        verbose=False,
        system_prompt=INSTRUCCIONES,
    )

    print("Bot: 🎓 Asistente Académico - Sección A 5to Año")
    print("     Escribe 'ayuda' para opciones o 'salir' para terminar\n")

    while True:
        try:
            query = input("Tú: ")

            if query.lower() == "salir":
                break

            if query.lower() == "ayuda":
                print("\nFunciones disponibles:")
                print("- Mostrar el promedio de las materias de la sección")
                print("- Listar los 3 mejores promedios")
                print("- Realizar búsquedas específicas\n")
                continue

            if error := query_validation(query):
                print(f"\n⚠️ {error}\n")
                continue

            response = agent.query(query)
            formatted_response = format_response(str(response))
            print(f"\nBot: {formatted_response}\n")

        except KeyboardInterrupt:
            print("\n\n⚠️ Sesión interrumpida\n")
            break

    print("\n¡Hasta pronto! 📚✨")
