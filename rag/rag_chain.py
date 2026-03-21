import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from rag.retriever import get_retriever

load_dotenv()


def ask_question(query):

    # ---------------- RETRIEVER ----------------
    retriever = get_retriever()
    docs = retriever.invoke(query)

    if not docs:
        return "No relevant information found."

    context = "\n\n".join([d.page_content[:1500] for d in docs])

    # ---------------- GROQ LLM ----------------
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return "❌ GROQ_API_KEY not set"

    llm = ChatGroq(
        temperature=0,
        model_name="llama-3.1-8b-instant",
        groq_api_key=api_key
    )

    # ---------------- PROMPT ----------------
    prompt = f"""
            You are a senior software engineer.

            Use the provided context to answer the question.

            If partial info is available, try to infer logically.
            DO NOT say "Not found" unless absolutely no relevant info exists.

            Context:
            {context}

            Question:
            {query}

            Answer with:
            - Clear explanation
            - Reference to code behavior
            """
    # ---------------- INVOKE ----------------
    try:
        response = llm.invoke([
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": prompt}
        ])

        return response.content.strip()

    except Exception as e:
        return f"❌ Error: {str(e)}"