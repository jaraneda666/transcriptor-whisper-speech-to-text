"""
Aplicación Web Interactiva Local en Streamlit para Transcriptor_Whisper_LLM.
Transcripción acelerada con Faster-Whisper y post-procesamiento con LLMs locales (Ollama y LM Studio).
"""

import os
import sys
import tempfile
import time
from pathlib import Path
import streamlit as st

# Asegurar importación de los módulos del core
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.audio_processor import AudioProcessor
from core.transcriber import WhisperTranscriber
from core.llm_client import LocalLLMClient
from core.export_manager import ExportManager

# Configuración de página
st.set_page_config(
    page_title="Transcriptor Multimedia AI",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #888888;
        margin-bottom: 1.5rem;
    }
    .status-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-active { background-color: #1b4d3e; color: #7cfc00; }
    .badge-inactive { background-color: #4a1515; color: #ff6b6b; }
    .metric-card {
        background-color: #1e1e24;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #333333;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_audio_processor():
    return AudioProcessor()


@st.cache_resource
def load_whisper_model(model_name: str, device: str, compute_type: str):
    return WhisperTranscriber(
        model_size=model_name,
        device=device,
        compute_type=compute_type if compute_type != "auto" else None
    )


def main():
    st.markdown('<div class="main-title">🎙️ Transcriptor Multimedia AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Inferencia local optimizada con <b>Faster-Whisper</b> (CTranslate2) y análisis inteligente vía <b>Ollama / LM Studio</b>.</div>',
        unsafe_allow_html=True
    )

    processor = get_audio_processor()
    exporter = ExportManager()

    # --- BARRA LATERAL: CONFIGURACIÓN ---
    st.sidebar.header("⚙️ Configuración del Motor")

    # 1. Configuración de Whisper
    st.sidebar.subheader("1. Faster-Whisper")
    whisper_model = st.sidebar.selectbox(
        "Tamaño del modelo Whisper",
        options=["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"],
        index=1,
        help="Modelos más grandes tienen mayor precisión pero exigen más memoria y tiempo."
    )

    device_option = st.sidebar.selectbox(
        "Dispositivo de cómputo",
        options=["auto", "cuda", "cpu"],
        index=0,
        help="'auto' selecciona GPU si CUDA está disponible."
    )

    compute_option = st.sidebar.selectbox(
        "Tipo de Cuantización",
        options=["auto", "int8", "float16", "int8_float16"],
        index=0,
        help="En GPU Nvidia, float16 o int8_float16 es óptimo. En CPU, int8 es el más veloz."
    )

    lang_code = st.sidebar.text_input(
        "Forzar idioma (opcional)",
        value="",
        placeholder="ej: es, en (vacío = auto-detección)"
    ).strip()
    forced_lang = lang_code if lang_code else None

    st.sidebar.markdown("---")

    # 2. Configuración de LLM Local
    st.sidebar.subheader("2. Modelo de Lenguaje Local")
    llm_backend = st.sidebar.radio(
        "Servidor LLM",
        options=["Auto-detectar", "Ollama", "LM Studio", "Desactivar"],
        index=0
    )

    backend_map = {
        "Auto-detectar": "auto",
        "Ollama": "ollama",
        "LM Studio": "lm_studio",
        "Desactivar": "none"
    }

    selected_backend = backend_map[llm_backend]
    llm_client = None
    llm_models = []

    if selected_backend != "none":
        llm_client = LocalLLMClient(backend=selected_backend)
        health = llm_client.check_health()
        
        status_col1, status_col2 = st.sidebar.columns(2)
        with status_col1:
            if health["ollama_active"]:
                st.markdown('<span class="status-badge badge-active">🟢 Ollama Activo</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-badge badge-inactive">🔴 Ollama Inactivo</span>', unsafe_allow_html=True)
        with status_col2:
            if health["lm_studio_active"]:
                st.markdown('<span class="status-badge badge-active">🟢 LM Studio Activo</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-badge badge-inactive">🔴 LM Studio Inactivo</span>', unsafe_allow_html=True)

        if health["current_active"]:
            llm_models = health["available_models"]
            selected_llm_model = st.sidebar.selectbox(
                "Modelo LLM a utilizar",
                options=llm_models if llm_models else ["Predeterminado"],
                index=0
            )
            llm_client.model_name = selected_llm_model if selected_llm_model != "Predeterminado" else None
        else:
            st.sidebar.warning(
                "Ningún servidor LLM detectado en los puertos locales (11434 o 1234). "
                "Inicie Ollama ('ollama serve') o active el servidor local de LM Studio."
            )
            llm_client = None

    # Tareas adicionales de LLM
    st.sidebar.subheader("3. Tareas con Inteligencia Artificial")
    do_correct = st.sidebar.checkbox("Corregir ortografía y puntuar párrafos", value=True, disabled=(llm_client is None))
    do_summary = st.sidebar.checkbox("Generar síntesis y puntos clave", value=True, disabled=(llm_client is None))
    do_minutes = st.sidebar.checkbox("Elaborar minuta de sesión / acuerdos", value=False, disabled=(llm_client is None))

    # --- PANTALLA PRINCIPAL ---
    uploaded_file = st.file_uploader(
        "Seleccione un archivo de video o audio para procesar",
        type=list(ext.lstrip(".") for ext in processor.SUPPORTED_EXTENSIONS),
        help="Formatos admitidos: MP4, MKV, AVI, MOV, FLV, WEBM, MP3, WAV, M4A, AAC, OGG, FLAC."
    )

    if uploaded_file is not None:
        file_details_col1, file_details_col2, file_details_col3 = st.columns(3)
        file_size_mb = round(uploaded_file.size / (1024 * 1024), 2)
        with file_details_col1:
            st.metric("Archivo", uploaded_file.name)
        with file_details_col2:
            st.metric("Tamaño", f"{file_size_mb} MB")
        with file_details_col3:
            st.metric("Extensión", Path(uploaded_file.name).suffix.upper())

        process_button = st.button("🚀 Iniciar Transcripción y Análisis", type="primary", use_container_width=True)

        if process_button:
            progress_bar = st.progress(0, text="Guardando archivo temporal...")
            
            # Guardar archivo cargado temporalmente
            suffix = Path(uploaded_file.name).suffix
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_in:
                tmp_in.write(uploaded_file.read())
                temp_input_path = tmp_in.name

            try:
                # 1. Extracción de audio
                progress_bar.progress(15, text="Extrayendo y normalizando pista de audio con FFmpeg...")
                wav_path, is_temp = processor.extract_audio(temp_input_path)

                # 2. Carga de modelo Whisper e inferencia
                progress_bar.progress(35, text=f"Cargando modelo Faster-Whisper ({whisper_model})...")
                transcriber = load_whisper_model(whisper_model, device_option, compute_option)

                progress_bar.progress(50, text="Transcribiendo habla a texto con marcas de tiempo...")
                t_data = transcriber.transcribe(wav_path, language=forced_lang, word_timestamps=True)

                # 3. Post-procesamiento con LLM
                summary_text = None
                minutes_text = None
                corrected_text = None

                if llm_client:
                    if do_correct:
                        progress_bar.progress(70, text="Refinando puntuación y ortografía con LLM...")
                        try:
                            corrected_text = llm_client.correct_and_format(t_data["text"])
                        except Exception as e:
                            st.warning(f"No se pudo completar la corrección con LLM: {e}")

                    if do_summary:
                        progress_bar.progress(85, text="Generando síntesis ejecutiva y conclusiones...")
                        try:
                            summary_text = llm_client.summarize(t_data["text"])
                        except Exception as e:
                            st.warning(f"No se pudo generar el resumen con LLM: {e}")

                    if do_minutes:
                        progress_bar.progress(92, text="Elaborando minuta estructurada...")
                        try:
                            minutes_text = llm_client.generate_meeting_minutes(t_data["text"])
                        except Exception as e:
                            st.warning(f"No se pudo generar la minuta con LLM: {e}")

                # 4. Exportar a todos los formatos
                progress_bar.progress(98, text="Compilando archivos finales en la carpeta /output...")
                base_name = Path(uploaded_file.name).stem
                exported_paths = exporter.export_all(
                    base_name,
                    t_data,
                    summary=summary_text,
                    meeting_minutes=minutes_text,
                    corrected_text=corrected_text
                )

                progress_bar.progress(100, text="¡Proceso finalizado exitosamente!")
                st.success(f"Procesamiento completado. Archivos guardados en: `{exporter.output_dir}`")

                # Guardar en estado de sesión para visualización
                st.session_state["results"] = {
                    "base_name": base_name,
                    "t_data": t_data,
                    "summary_text": summary_text,
                    "minutes_text": minutes_text,
                    "corrected_text": corrected_text,
                    "exported_paths": exported_paths
                }

            finally:
                if os.path.exists(temp_input_path):
                    os.unlink(temp_input_path)
                if 'wav_path' in locals() and is_temp and os.path.exists(wav_path):
                    os.unlink(wav_path)

    # --- RENDERIZADO DE RESULTADOS ---
    if "results" in st.session_state:
        res = st.session_state["results"]
        t_data = res["t_data"]
        exported = res["exported_paths"]

        st.markdown("---")
        st.subheader("📊 Métricas de Transcripción")

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric("Idioma Detectado", f"{t_data.get('language', 'N/A').upper()} ({round(t_data.get('language_probability', 0)*100, 1)}%)")
        with m_col2:
            st.metric("Tiempo de Audio", f"{t_data.get('duration', 'N/A')} s")
        with m_col3:
            st.metric("Tiempo Inferencia", f"{t_data.get('transcription_time_seconds', 'N/A')} s")
        with m_col4:
            st.metric("Segmentos de Voz", len(t_data.get("segments", [])))

        # Pestañas de contenido
        tab_raw, tab_corr, tab_sum, tab_min, tab_down = st.tabs([
            "📄 Transcripción Original",
            "✍️ Texto Corregido",
            "📈 Síntesis y Resumen",
            "📋 Minuta de Sesión",
            "📥 Descarga de Archivos"
        ])

        with tab_raw:
            st.write(t_data.get("text", ""))
            with st.expander("Ver segmentos con marcas de tiempo"):
                for seg in t_data.get("segments", []):
                    st.text(f"[{seg['start']}s -> {seg['end']}s] {seg['text']}")

        with tab_corr:
            if res.get("corrected_text"):
                st.markdown(res["corrected_text"])
            else:
                st.info("No se ejecutó corrección de texto con LLM.")

        with tab_sum:
            if res.get("summary_text"):
                st.markdown(res["summary_text"])
            else:
                st.info("No se generó resumen con LLM.")

        with tab_min:
            if res.get("minutes_text"):
                st.markdown(res["minutes_text"])
            else:
                st.info("No se generó minuta con LLM.")

        with tab_down:
            st.markdown("### Descarga directa de entregables compilados")
            st.caption("Todos los archivos se encuentran confinados permanentemente en la carpeta `output/` del proyecto.")

            d_col1, d_col2, d_col3 = st.columns(3)
            d_col4, d_col5, d_col6 = st.columns(3)

            def get_bytes(file_path):
                if file_path and os.path.exists(file_path):
                    with open(file_path, "rb") as f:
                        return f.read()
                return b""

            with d_col1:
                st.download_button(
                    label="📄 Descargar TXT",
                    data=get_bytes(exported.get("txt")),
                    file_name=f"{res['base_name']}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            with d_col2:
                st.download_button(
                    label="🎬 Subtítulos SRT",
                    data=get_bytes(exported.get("srt")),
                    file_name=f"{res['base_name']}.srt",
                    mime="application/x-subrip",
                    use_container_width=True
                )
            with d_col3:
                st.download_button(
                    label="🌐 Subtítulos VTT",
                    data=get_bytes(exported.get("vtt")),
                    file_name=f"{res['base_name']}.vtt",
                    mime="text/vtt",
                    use_container_width=True
                )
            with d_col4:
                st.download_button(
                    label="📊 Datos JSON",
                    data=get_bytes(exported.get("json")),
                    file_name=f"{res['base_name']}.json",
                    mime="application/json",
                    use_container_width=True
                )
            with d_col5:
                st.download_button(
                    label="📝 Informe Markdown",
                    data=get_bytes(exported.get("md")),
                    file_name=f"{res['base_name']}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            with d_col6:
                st.download_button(
                    label="📘 Documento Word DOCX",
                    data=get_bytes(exported.get("docx")),
                    file_name=f"{res['base_name']}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True
                )


if __name__ == "__main__":
    main()
