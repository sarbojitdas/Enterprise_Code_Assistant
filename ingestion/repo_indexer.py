import os
import shutil
import subprocess
import stat
import time

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# ==============================
# CONFIG
# ==============================
SUPPORTED_EXTENSIONS = (
    ".py", ".js", ".ts", ".java", ".cpp", ".go"
)

IGNORE_DIRS = [
    "node_modules", ".git", "venv", "__pycache__",
    "tests", "test", "build", "dist"
]

MAX_FILES = 600             # 🔥 Limit files
MAX_FILE_SIZE = 150_000      # 🔥 150 KB max per file


# ==============================
# HANDLE WINDOWS DELETE ISSUE
# ==============================
def handle_remove_readonly(func, path, exc):
    os.chmod(path, stat.S_IWRITE)
    func(path)


# ==============================
# CLONE REPO
# ==============================
def clone_repo(repo_url, repo_path="repos/temp_repo"):

    if os.path.exists(repo_path):
        shutil.rmtree(repo_path, onerror=handle_remove_readonly)

    os.makedirs("repos", exist_ok=True)

    print("Cloning repository...")

    result = subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, repo_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        print(result.stderr)
        raise Exception("Git clone failed")

    time.sleep(1)

    return repo_path


# ==============================
# LOAD FILES (OPTIMIZED)
# ==============================
def load_code_files(repo_path):

    documents = []

    for root, dirs, files in os.walk(repo_path):

        # 🚫 Skip heavy folders
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        for file in files:

            if len(documents) >= MAX_FILES:
                break

            if file.endswith(SUPPORTED_EXTENSIONS):

                file_path = os.path.join(root, file)

                try:
                    # 🚫 Skip large files
                    if os.path.getsize(file_path) > MAX_FILE_SIZE:
                        continue

                    # ✅ Fix encoding issues
                    loader = TextLoader(file_path, autodetect_encoding=True)
                    docs = loader.load()

                    for d in docs:
                        d.metadata["source"] = file_path

                    documents.extend(docs)

                except Exception as e:
                    print(f"Skipping {file_path}: {e}")

        if len(documents) >= MAX_FILES:
            break

    return documents


# ==============================
# INDEX REPO (FAST)
# ==============================
def index_repository(repo_url, persist_directory="vectorstore"):

    # 🛑 Prevent re-indexing
    if os.path.exists(persist_directory):
        print("Vector DB already exists. Skipping indexing.")
        return Chroma(persist_directory=persist_directory)

    repo_path = clone_repo(repo_url)

    print("Loading files...")
    documents = load_code_files(repo_path)

    print(f"Loaded {len(documents)} files")

    if len(documents) == 0:
        raise ValueError("No supported files found")

    # ⚡ Faster chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks")

    # ⚡ FAST embeddings (no Ollama)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"batch_size": 64}
    )

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    print("✅ Repository indexed successfully")

    return vectordb