# app/ui_streamlit.py
import sys, os, uuid
import streamlit as st
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.rag_pipeline import load_vectorstore_from_disk, build_chain
from app.services.chat_service import save_message
from app.services.profile_service import upsert_profile, fetch_profile

# Configuración de la página
st.set_page_config(
    page_title="Chatbot Entrenamiento",
    layout="centered",
)

# Inicialización de user_id
if "user_id" not in st.session_state:
    # Genera un UUID único
    st.session_state.user_id = str(uuid.uuid4())

# Inicialización de profile
if "profile" not in st.session_state:
    perfil_bd = fetch_profile(st.session_state.user_id)
    if perfil_bd:
        st.session_state.profile = perfil_bd
    else:
        st.session_state.profile = {
            "genero": None,
            "edad": None,
            "estatura": None,
            "peso": None,
            "lesion": False,
            "lesion_descripcion": ""
        }

# Redirección de pestaña
if "next_tab" in st.session_state:
    st.session_state.tab_selector = st.session_state.next_tab
    del st.session_state.next_tab

# Selector de pestañas
if "tab_selector" not in st.session_state:
    st.session_state.tab_selector = "Editar Perfil"

selected_widget = st.radio(
    "", 
    ["Editar Perfil", "Chat"],
    key="tab_selector",
    horizontal=True
)
st.markdown("---")

# Vista 'Editar Perfil'
if selected_widget == "Editar Perfil":
    st.header("⚙️ Editar Perfil")

    # Prellenar con los valores actuales (o placeholders)
    genero_init = st.session_state.profile.get("genero", "Selecciona tu género")
    edad_init   = st.session_state.profile.get("edad",    None)
    est_init    = st.session_state.profile.get("estatura", None)
    peso_init   = st.session_state.profile.get("peso",    None)
    lesion_init = "Si" if st.session_state.profile.get("lesion") else "No"
    desc_init   = st.session_state.profile.get("lesion_descripcion", "")

    with st.form("form_editar_perfil"):
        genero_options = ["Selecciona tu género", "Masculino", "Femenino", "Otro"]
        genero = st.selectbox(
            "Género",
            genero_options,
            index=genero_options.index(genero_init) if genero_init in genero_options else 0
        )

        edad = st.number_input(
            "Edad", 
            min_value=1, 
            max_value=120, 
            value=edad_init, 
            placeholder="Ingresa tu edad", 
            step=1
        )

        estatura = st.number_input(
            "Estatura (cm)", 
            min_value=50, 
            max_value=250, 
            value=est_init,
            placeholder="Ingresa tu estatura", 
            step=1
        )

        peso = st.number_input(
            "Peso (kg)", 
            min_value=20, 
            max_value=300, 
            value=peso_init,
            placeholder="Ingresa tu Peso",
            step=1
        )  

        lesion = st.radio(
            "¿Presentas alguna lesión actualmente?",
            options=["No", "Si"],
            index=0 if lesion_init=="No" else 1
        )

        lesion_descripcion = ""
        if lesion == "Si":
            lesion_descripcion = st.text_area(
                "Describe tu lesión",
                value=desc_init,
                placeholder="ej. Dolor de hombro derecho hace 2 semanas"
            )

        submit = st.form_submit_button("Guardar")

    if submit:
        errores = []
        # Validaciones
        if genero == genero_options[0]:
            errores.append("Debes seleccionar un género.")
        if edad is None:
            errores.append("Debes ingresar tu edad.")
        if estatura is None:
            errores.append("Debes ingresar tu estatura.")
        if peso is None:
            errores.append("Debes ingresar tu peso.")
        if lesion == "Si" and not lesion_descripcion.strip():
            errores.append("Describe tu lesión o desmarca la opción si no tienes.")

        if errores:
            for e in errores:
                st.error(e)
        else:
            new_profile = {
                "genero": genero,
                "edad": edad,
                "estatura": estatura,
                "peso": peso,
                "lesion": lesion == "Si",
                "lesion_descripcion": lesion_descripcion,
            }
            st.session_state.profile = new_profile
            upsert_profile(st.session_state.user_id, new_profile)
            st.success("Perfil actualizado correctamente")
            st.session_state.next_tab = "Chat"
            st.experimental_rerun()

# Vista 'Chat'
else:
    st.subheader("💬 Chat de Entrenamiento")

    # Inicializar historial
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Cargar vectorstore y cadena RAG
    vectordb = load_vectorstore_from_disk()
    chain = build_chain(vectordb)

    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).write(msg["content"])

    prompt = st.chat_input("¿En qué puedo ayudarte hoy?")

    if prompt:
        st.chat_message("user").write(prompt)
        save_message(st.session_state.user_id, "user", prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                result = chain.invoke({
                    "question": prompt,
                    "chat_history": [
                        (m["content"], n["content"])
                        for m,n in zip(
                            st.session_state.chat_history[0::2],
                            st.session_state.chat_history[1::2]
                        )
                    ],
                    "profile": st.session_state.profile
                })
            st.write(result["answer"])

        save_message(st.session_state.user_id, "assistant", result["answer"])
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result["answer"]
        })