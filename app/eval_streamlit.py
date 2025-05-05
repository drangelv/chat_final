# app/main_interface.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
st.set_page_config(page_title="📚 Chatbot GenAI + Métricas", layout="wide")

import pandas as pd
import mlflow
import json
from app.rag_pipeline import load_vectorstore_from_disk, build_chain

modo = "📊 Métricas"

vectordb = load_vectorstore_from_disk()
chain = build_chain(vectordb)

if modo == "📊 Métricas":
    st.title("📈 Resultados de Evaluación")

    client = mlflow.tracking.MlflowClient()
    experiments = [exp for exp in client.search_experiments() if exp.name.startswith("eval_")]

    if not experiments:
        st.warning("No se encontraron experimentos de evaluación.")
        st.stop()

    exp_names = [exp.name for exp in experiments]
    selected_exp = st.selectbox("Selecciona un experimento:", exp_names)

    experiment = client.get_experiment_by_name(selected_exp)
    runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"])

    if not runs:
        st.warning("No hay ejecuciones registradas.")
        st.stop()

    # Armar dataframe
    data = []
    for run in runs:
        params = run.data.params
        metrics = run.data.metrics
        data.append({
            "Pregunta": params.get("question"),
            "Prompt": params.get("prompt_version"),
            "Chunk Size": int(params.get("chunk_size", 0)),
            "Correcto (LC)": metrics.get("lc_is_correct", 0)
        })

    df = pd.DataFrame(data)
    st.dataframe(df)