import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import mlflow
from dotenv import load_dotenv
from app.rag_pipeline import load_vectorstore_from_disk, build_chain

from langchain_openai import ChatOpenAI
from langchain.evaluation.qa import QAEvalChain
from langchain.evaluation.criteria import LabeledCriteriaEvalChain

load_dotenv()

# Configuración
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v1_asistente_entrenamiento")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
DATASET_PATH = "tests/eval_dataset.json"

DEFAULT_PROFILE = {
    "genero": "Masculino",
    "edad": 25,
    "estatura": 180,
    "peso": 80,
    "lesion": False,
    "lesion_descripcion": ""
}

# Cargar dataset
with open(DATASET_PATH) as f:
    dataset = json.load(f)

# Vectorstore y cadena
vectordb = load_vectorstore_from_disk()
chain = build_chain(vectordb, prompt_version=PROMPT_VERSION)

# LangChain Evaluator
llm = ChatOpenAI(temperature=0)
langchain_eval = QAEvalChain.from_llm(llm)

# ✅ Establecer experimento una vez
mlflow.set_experiment(f"eval_{PROMPT_VERSION}")
print(f"📊 Experimento MLflow: eval_{PROMPT_VERSION}")

# Configurar criterios de evaluación como un diccionario
CRITERIA = {
    "correctness": "Is the response factually correct based on the reference?",
    "relevance": "Is the response relevant to the question asked?",
    "coherence": "Is the response coherent, well-structured, and easy to follow?"
}

# Crear evaluador de criterios etiquetados
criteria_eval = LabeledCriteriaEvalChain.from_llm(llm, criteria=CRITERIA)

# Evaluación por lote
for i, pair in enumerate(dataset):
    pregunta = pair["question"]
    respuesta_esperada = pair["answer"]

    with mlflow.start_run(run_name=f"eval_q{i+1}"):
        result = chain.invoke({"question": pregunta, "chat_history": [], "profile": DEFAULT_PROFILE})
        respuesta_generada = result["answer"]

        # Evaluación con LangChain
        graded = langchain_eval.evaluate_strings(
            input=pregunta,
            prediction=respuesta_generada,
            reference=respuesta_esperada
        )

        # Evaluación con LabeledCriteriaEvalChain
        criteria_results = criteria_eval.evaluate_strings(
            prediction=respuesta_generada,
            reference=respuesta_esperada,
            input=pregunta
        )

        # 🔍 Imprimir el contenido real
        print(f"\n📦 Resultado evaluación LangChain para pregunta {i+1}/{len(dataset)}:")
        print(graded)
        
        print(f"\n📦 Resultado evaluación por criterios para pregunta {i+1}/{len(dataset)}:")
        print(criteria_results)

        lc_verdict = graded.get("value", "UNKNOWN")
        is_correct = graded.get("score", 0)

        # Log en MLflow
        mlflow.log_param("question", pregunta)
        mlflow.log_param("prompt_version", PROMPT_VERSION)
        mlflow.log_param("chunk_size", CHUNK_SIZE)
        mlflow.log_param("chunk_overlap", CHUNK_OVERLAP)

        mlflow.log_metric("lc_is_correct", is_correct)
        
        # Log de criterios en MLflow - CORREGIDO
        criteria_reasoning = criteria_results.get('reasoning', '')
        criteria_value = criteria_results.get('value', '')
        criteria_score = criteria_results.get('score', 0)
        
        # Registrar el score general en MLflow
        mlflow.log_metric("criteria_score", criteria_score)
        
        # Intentar extraer puntuaciones individuales para cada criterio del texto de reasoning
        # Esto es opcional y depende de si necesitas métricas separadas
        import re
        
        # Buscar patrones como "1. Correctness:" seguido de evaluación
        for criterion in CRITERIA.keys():
            # Buscar menciones del criterio en el texto
            if criterion.lower() in criteria_reasoning.lower():
                # Intentar determinar si la evaluación es positiva
                positive_indicators = ["meets", "satisfies", "correct", "relevant", "coherent", "good", "excellent"]
                negative_indicators = ["incorrect", "irrelevant", "incoherent", "poor", "fails", "does not meet"]
                
                # Extraer el párrafo que contiene el criterio
                pattern = r'(?i)\d+\.\s*' + criterion + r'[^\n]*\n(.*?)(?=\d+\.\s*\w+:|$)'
                matches = re.search(pattern, criteria_reasoning, re.DOTALL)
                
                if matches:
                    criterion_text = matches.group(1).strip()
                    # Determinar puntuación basada en indicadores
                    positive_count = sum(1 for ind in positive_indicators if ind.lower() in criterion_text.lower())
                    negative_count = sum(1 for ind in negative_indicators if ind.lower() in criterion_text.lower())
                    
                    # Asignar puntuación simple basada en balance de indicadores
                    if positive_count > negative_count:
                        estimated_score = 1.0
                    else:
                        estimated_score = 0.0
                    
                    mlflow.log_metric(f"criteria_{criterion}", estimated_score)

        print(f"✅ Pregunta: {pregunta}")
        print(f"🧠 LangChain Eval: {lc_verdict}")
        print(f"🔍 Criterios Eval: valor={criteria_value}, puntuación={criteria_score}")
        print(f"📝 Razonamiento: {criteria_reasoning[:100]}...")  # Mostrar primeros 100 caracteres