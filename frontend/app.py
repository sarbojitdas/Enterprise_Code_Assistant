import streamlit as st

# Import your core logic directly
from ingestion.repo_indexer import index_repository
from rag.rag_chain import ask_question
from rag.code_qa import ask_question_from_code

st.set_page_config(page_title="Enterprise Code Assistant", layout="wide")

st.title("💻 Enterprise Code RAG Assistant")

tab1, tab2, tab3 = st.tabs([
    "📦 Index Repository",
    "🔎 Ask Repo Question",
    "🐞 Code Bug Analyzer"
])

# ==============================
# INDEX REPOSITORY
# ==============================
with tab1:
    st.header("Index GitHub Repository")

    repo_url = st.text_input("Enter GitHub Repository URL")

    if st.button("Index Repository") and "indexed" not in st.session_state:
        st.session_state.indexed = True
        if repo_url == "":
            st.warning("Please enter a repository URL")
        else:
            with st.spinner("Indexing repository... This may take a few minutes"):
                try:
                    index_repository(repo_url)
                    st.success("Repository indexed successfully!")
                except Exception as e:
                    st.error(f"Error: {e}")

# ==============================
# ASK QUESTION FROM REPO
# ==============================
with tab2:
    st.header("Ask Question From Repository")

    query = st.text_input("Ask a question about the codebase")

    if st.button("Ask Question"):
        if query == "":
            st.warning("Please enter a question")
        else:
            with st.spinner("Searching repository..."):
                try:
                    answer = ask_question(query)

                    st.subheader("📌 Answer")
                    st.markdown(answer)

                except Exception as e:
                    st.error(f"Error: {e}")

# ==============================
# CODE BUG ANALYZER
# ==============================
with tab3:
    st.header("Analyze Code For Bugs")

    code = st.text_area("Paste your code here", height=300)

    question = st.text_input(
        "What do you want to analyze?",
        value="Find bugs and correct the code"
    )

    if st.button("Analyze Code"):
        if code == "":
            st.warning("Please paste some code")
        else:
            with st.spinner("Analyzing code..."):
                try:
                    answer = ask_question_from_code(code, question)

                    st.subheader("Analysis Result")
                    st.write(answer)

                except Exception as e:
                    st.error(f"Error: {e}")