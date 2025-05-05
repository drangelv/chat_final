# 🧠 Entrenador Virtual Inteligente

Este proyecto desarrolla un agente conversacional que funciona como entrenador virtual, capaz de generar y ajustar rutinas de entrenamiento funcional según las necesidades del usuario, combinando NLP, recuperación aumentada por generación (RAG) e interacción en tiempo real.

---
# 🏗️ Infraestructura y arquitectura
El sistema se estructura en cinco componentes principales:

### 1. Fuente de datos:
Documentos en PDF relacionados con entrenamiento funcional y fisioterapia que conforman la base de conocimiento del agente.
### 2. Pipeline RAG: 
Procesamiento de los documentos mediante técnicas de chunking, generación de embeddings y construcción de un índice semántico para recuperación de información relevante.
### 3. Modelo conversacional: 
Un agente basado en un modelo de lenguaje (GPT-4o) que interactúa con los usuarios, combinando contexto y resultados del motor de recuperación.
### 4. Evaluación automatizada: 
Módulo que prueba el rendimiento del sistema con preguntas reales y mide la calidad de las respuestas generadas usando métricas automáticas.
### 5. Interfaz de usuario: 
Aplicación web interactiva desarrollada en Streamlit para facilitar la conversación con el agente y visualizar métricas de rendimiento.

---

# ⚙️ Herramientas principales
- **LangChain** para la construcción del pipeline RAG y la orquestación del agente.
- **OpenAI GPT-4o** como modelo de lenguaje principal.
- **FAISS** para almacenamiento y búsqueda eficiente de vectores semánticos.
- **Streamlit** para la interfaz interactiva.
- **MLflow** para el tracking de experimentos y visualización de métricas.
- **Supabase** para gestión de datos adicionales y posibles registros de retroalimentación del usuario.
- **dotenv** y **os** para la gestión de variables de entorno.

# 📊 Evaluación del sistema
El sistema incluye un módulo de evaluación automatizada que permite medir la calidad de las respuestas del agente. A través de un conjunto de preguntas de prueba, se evalúa su desempeño en criterios como precisión, completitud y coherencia usando el módulo LabeledCriteriaEvalChain de LangChain. Los resultados se registran automáticamente en MLflow y pueden visualizarse mediante dashboard.


# 🧪 Cómo ejecutar
### 1. Clona el repositorio:

```bash
git clone https://github.com/drangelv/chat_final.git
cd nlp_u
```

### 2. Crea un entorno virtual y activa:

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Instala las dependencias:

```bash
pip install -r requirements.txt
```

### 4. Define tus variables de entorno en un archivo .env:

```bash
cp .env.example .env  # Agrega tus API KEY de OpenAI y Supabase
```


OPENAI_API_KEY=tu_api_key

SUPABASE_URL=tu_url

SUPABASE_KEY=tu_clave


### 5. Crear base de datos vectorial:

```bash
python -c "from app.rag_pipeline import save_vectorstore; save_vectorstore()"
```

Para reutilizarlo directamente:
```python
vectordb = load_vectorstore_from_disk()
```

### 6. Ejecuta la interfaz:
Chat de entrenamiento
```bash
streamlit run app/ui_streamlit.py
```

### 7. Evaluación:

```bash
python app/run_eval.py
```

Métricas de evaluación:
```bash
streamlit run app/eval_streamlit.py
```


## 📂 Estructura del Proyecto

```
├── app/
│   ├── ui_streamlit.py           ← interfaz simple del chatbot
│   ├── eval_streamlit.py         ← interfaz combinada con métricas
│   ├── supabase_client.py        ← Almacenamiento y autenticación de BD
│   ├── run_eval.py               ← evaluación automática
│   ├── rag_pipeline.py           ← lógica de ingestión y RAG
│   └── prompts/
│       ├── v1_asistente_entrenamiento.txt
│       └── v2_resumido_directo.txt
├── data/pdfs/                    ← documentos fuente
├── tests/
│   ├── test_run_eval.py
│   ├── eval_dataset.json         ← dataset de evaluación
│   └── eval_dataset.csv
├── .env.example
├── Dockerfile
├── .devcontainer/
│   └── devcontainer.json
├── .github/workflows/
│   ├── eval.yml
│   └── test.yml    

---
