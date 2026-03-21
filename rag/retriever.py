from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import VECTOR_DB_PATH


def get_retriever():

    # ✅ Use HuggingFace embeddings (NO Ollama)
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # ✅ Load persistent Chroma DB
    vectordb = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings
    )

    # ✅ Retriever
    retriever = vectordb.as_retriever(
       search_kwargs={"k": 10}
    )

    return retriever