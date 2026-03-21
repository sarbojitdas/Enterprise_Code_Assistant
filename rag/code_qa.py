import os
from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_groq import ChatGroq


def ask_question_from_code(code: str, question: str):

    # ---------------- LOAD DOC ----------------
    docs = [Document(page_content=code)]

    # ---------------- SPLIT ----------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    splits = splitter.split_documents(docs)

    # ---------------- EMBEDDINGS ----------------
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # ---------------- VECTOR DB ----------------
    vectordb = Chroma.from_documents(
        splits,
        embedding=embeddings
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    # ---------------- GROQ LLM ----------------
    api_key = os.getenv("GROQ_API_KEY")

    llm = ChatGroq(
        temperature=0,
        model_name="llama-3.1-8b-instant",
        groq_api_key=api_key
    )

    # ---------------- PROMPT ----------------
    prompt = ChatPromptTemplate.from_template(
        """
You are an expert software engineer.

Code:
{context}

Question:
{question}

Tasks:
1. Explain the code clearly
2. Identify bugs/issues
3. Suggest fixes
4. Suggest improvements
"""
    )

    # ---------------- RETRIEVE ----------------
    retrieved_docs = retriever.invoke(question)

    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # ---------------- CHAIN ----------------
    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({
        "context": context,
        "question": question
    })

    return answer