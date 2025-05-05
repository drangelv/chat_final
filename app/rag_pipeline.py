# app/rag_pipeline.py

import os
from dotenv import load_dotenv
import mlflow

from langchain.globals import set_verbose, get_verbose
set_verbose(True)  # Si quieres ver logs detallados

from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

load_dotenv()

DATA_DIR    = "data/pdfs"
PROMPT_DIR  = "app/prompts"
VECTOR_DIR  = "vectorstore"

def load_documents(path=DATA_DIR):
    """Carga todos los PDFs en la carpeta `data/pdfs` usando PyPDFLoader."""
    docs = []
    for file in os.listdir(path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(path, file))
            docs.extend(loader.load())
    return docs

def save_vectorstore(chunk_size=512, chunk_overlap=50, persist_path=VECTOR_DIR):
    """Crea y persiste el FAISS vectorstore, y registra métricas en MLflow."""
    docs = load_documents()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.from_documents(chunks, embedding=embeddings)
    vectordb.save_local(persist_path)

    mlflow.set_experiment("vectorstore_tracking")
    with mlflow.start_run(run_name="vectorstore_build"):
        mlflow.log_param("chunk_size", chunk_size)
        mlflow.log_param("chunk_overlap", chunk_overlap)
        mlflow.log_param("n_chunks", len(chunks))
        mlflow.log_param("n_docs", len(docs))
        mlflow.set_tag("vectorstore", persist_path)

def load_vectorstore(chunk_size=512, chunk_overlap=50):
    docs = load_documents()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(chunks, embedding=embeddings)

def load_vectorstore_from_disk(persist_path=VECTOR_DIR):
    """Carga el FAISS vectorstore desde disco."""
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(persist_path, embeddings, allow_dangerous_deserialization=True)

def load_prompt(version="v1_asistente_entrenamiento"):
    prompt_path = os.path.join(PROMPT_DIR, f"{version}.txt")
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt no encontrado: {prompt_path}")
    with open(prompt_path, "r", encoding="utf-8") as f:
        template_text = f.read()

    return PromptTemplate(
        input_variables=["context", "question", "profile"],
        template=template_text,
        template_format="jinja2",      # <<< aquí le indicas que use Jinja2
    )


def build_chain(vectordb, prompt_version="v1_asistente_entrenamiento"):
    prompt = load_prompt(prompt_version)
    retriever = vectordb.as_retriever()
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    return ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=False
    )
