
import streamlit as st
import pymongo
import logging
from google import genai
from google.genai import types

# ==========================
# CONFIGURACION DE LA APP
# ==========================

st.set_page_config(
    page_title="Asistente RAG de Tesis",
    page_icon="📚",
    layout="centered"
)

GOOGLE_API_KEY = st.secrets["app"]["GOOGLE_API_KEY"]
MONGODB_URI = st.secrets["app"]["MONGODB_URI"]

# ==========================
# CLIENTES
# ==========================

@st.cache_resource
def get_genai_client():
    return genai.Client(api_key=GOOGLE_API_KEY)

@st.cache_resource
def get_mongo_collection():
    client = pymongo.MongoClient(
        MONGODB_URI,
        serverSelectionTimeoutMS=15000
    )

    db = client["pdf_embeddings_db"]
    return db["tesis_vectors"]

client_genai = get_genai_client()
collection = get_mongo_collection()

# ==========================
# GENERAR EMBEDDINGS
# ==========================

def crear_embedding(pregunta):
    response = client_genai.models.embed_content(
        model="gemini-embedding-001",
        contents=pregunta,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY"
        )
    )

    return response.embeddings[0].values

# ==========================
# BUSQUEDA VECTORIAL
# ==========================

def buscar_similares(embedding, k=5):
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": embedding,
                "numCandidates": 100,
                "limit": k
            }
        },
        {
            "$project": {
                "_id": 0,
                "id": 1,
                "texto": 1,
                "score": {
                    "$meta": "vectorSearchScore"
                }
            }
        }
    ]

    return list(collection.aggregate(pipeline))

# ==========================
# GENERAR RESPUESTA RAG
# ==========================

def generar_respuesta(pregunta, contextos):

    contexto = "\n\n".join(
        f"Fragmento {c['id']}:\n{c['texto']}"
        for c in contextos
    )

    prompt = f"""
Eres un asistente academico especializado en un
plan de tesis de Ingenieria de Sistemas de Informacion.

La investigacion trata sobre el diseño e implementacion
de una arquitectura integrada con deteccion no supervisada
de anomalias para mejorar la trazabilidad de los servicios
operativos en una empresa recicladora.

Tu funcion es responder preguntas relacionadas con
el documento de tesis proporcionado.

REGLAS:

1. Utiliza EXCLUSIVAMENTE la informacion del contexto.
2. Responde de manera clara, natural y en español.
3. Puedes explicar conceptos tecnicos de forma sencilla.
4. Si la respuesta no esta en el documento, indicalo.
5. No inventes datos, resultados ni referencias.
6. Distingue entre objetivos, hipotesis, propuestas y
   resultados realmente obtenidos.
7. No afirmes que la arquitectura ya fue implementada
   si el documento solo describe su implementacion futura.
8. Prioriza las partes relevantes para la pregunta.

CONTEXTO:
{contexto}

PREGUNTA:
{pregunta}

RESPUESTA:
"""

    response = client_genai.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    return response.text or "No se pudo generar una respuesta."

# ==========================
# INTERFAZ STREAMLIT
# ==========================

st.title("📚 Asistente RAG de Tesis")

st.write(
    "Consulta informacion sobre el problema de investigacion, "
    "los objetivos, las variables, la metodologia y la deteccion "
    "de anomalias a partir del documento de tesis."
)

k = st.selectbox(
    "Cantidad de fragmentos a recuperar (k)",
    options=[3, 5, 10],
    index=1
)

if "historial" not in st.session_state:
    st.session_state.historial = []

# Mostrar conversaciones anteriores
for mensaje in st.session_state.historial:

    rol = (
        "user" if mensaje["rol"] == "usuario"
        else "assistant"
    )

    with st.chat_message(rol):
        st.write(mensaje["texto"])

        if mensaje.get("fuentes"):
            with st.expander("Fragmentos recuperados"):
                for i, c in enumerate(mensaje["fuentes"], 1):
                    st.markdown(
                        f"**Fragmento {i}** "
                        f"(ID: {c['id']}) "
                        f"— score: `{c['score']:.4f}`"
                    )
                    st.write(c["texto"][:600])
                    st.divider()


# ==========================
# NUEVA PREGUNTA
# ==========================

pregunta = st.chat_input(
    "Escribe tu pregunta sobre la tesis..."
)

if pregunta:

    with st.chat_message("user"):
        st.write(pregunta)

    st.session_state.historial.append({
        "rol": "usuario",
        "texto": pregunta
    })

    similares = []

    with st.chat_message("assistant"):

        with st.spinner("Analizando el documento..."):
            try:
                embedding = crear_embedding(pregunta)

                similares = buscar_similares(
                    embedding,
                    k=k
                )

                if similares:
                    respuesta = generar_respuesta(
                        pregunta,
                        similares
                    )
                else:
                    respuesta = (
                        "No encontre informacion relevante "
                        "en el documento."
                    )

            except Exception:
                logging.exception(
                    "Error al consultar el RAG de tesis"
                )
                respuesta = (
                    "Ocurrio un error al consultar el documento. "
                    "Intenta nuevamente."
                )

        st.write(respuesta)

        if similares:
            with st.expander("Fragmentos recuperados"):
                for i, c in enumerate(similares, 1):
                    st.markdown(
                        f"**Fragmento {i}** "
                        f"(ID: {c['id']}) "
                        f"— score: `{c['score']:.4f}`"
                    )
                    st.write(c["texto"][:600])
                    st.divider()

    st.session_state.historial.append({
        "rol": "bot",
        "texto": respuesta,
        "fuentes": similares
    })
