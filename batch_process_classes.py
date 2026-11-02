"""
Script de procesamiento por lotes autónomo para Transcriptor_Whisper_LLM.
Procesa todos los videos *.mp4 de una carpeta raíz y subcarpetas, transcribe con Faster-Whisper (CUDA)
y ejecuta post-procesamiento jerárquico con LLM local (LM Studio).
Guarda los entregables en output/ y copia el archivo .docx generado en cada subcarpeta junto al video.
"""

import os
import sys
import json
import time
import shutil
import logging
import requests
from pathlib import Path

# Configuración de codificación UTF-8 en stdout/stderr para Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.audio_processor import AudioProcessor
from core.transcriber import WhisperTranscriber
from core.export_manager import ExportManager

# Configuración de logging y archivos
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = OUTPUT_DIR / "batch_execution.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("BatchProcessor")

LM_STUDIO_URL = "http://127.0.0.1:1234/v1"
LM_STUDIO_MODEL = "qwen/qwen3-vl-8b"


def query_llm(prompt: str, system_prompt: str, max_tokens: int = 1200, temperature: float = 0.2) -> str:
    """Envía solicitud a LM Studio con manejo de excepciones."""
    url = f"{LM_STUDIO_URL}/chat/completions"
    payload = {
        "model": LM_STUDIO_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    headers = {"Content-Type": "application/json"}
    
    r = requests.post(url, json=payload, headers=headers, timeout=240)
    if r.status_code != 200:
        raise RuntimeError(f"Error {r.status_code} desde LM Studio: {r.text}")
    res = r.json()
    return res["choices"][0]["message"]["content"].strip()


def format_seconds(seconds: float) -> str:
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


def clean_anti_slop(text: str) -> str:
    """Aplica normas estrictas anti-slop sobre el texto generado."""
    cleaned = text.replace(" — ", " (").replace("— ", ": ").replace("—", " - ")
    cliches = [
        ("es importante destacar que", "conviene notar que"),
        ("Es importante destacar que", "Conviene notar que"),
        ("cabe mencionar que", "se precisa que"),
        ("Cabe mencionar que", "Se precisa que"),
        ("en resumen,", "en conclusión,"),
        ("En resumen,", "En conclusión,"),
        ("no solo ..., sino también ...", "... y además ..."),
        ("normativamente robusto como culturalmente sostenible", "normativamente consistente y culturalmente sostenible"),
        ("formatos más robustos", "formatos más exhaustivos"),
        ("robusto", "consistente"),
        ("robusta", "consistente"),
        ("robustos", "exhaustivos"),
    ]
    for old, new in cliches:
        cleaned = cleaned.replace(old, new)
    return cleaned


def create_dynamic_chunks(segments, max_words=3600):
    """Agrupa segmentos contiguos garantizando que ningún bloque supere max_words (<8192 tokens)."""
    chunks = []
    current_segs = []
    current_words = 0
    chunk_idx = 1
    
    for seg in segments:
        seg_words = len(seg["text"].split())
        if current_words + seg_words > max_words and current_segs:
            t_start = current_segs[0]["start"]
            t_end = current_segs[-1]["end"]
            chunks.append({
                "id": chunk_idx,
                "start": t_start,
                "end": t_end,
                "segments": current_segs,
                "text": " ".join([s["text"] for s in current_segs]),
                "words": current_words
            })
            chunk_idx += 1
            current_segs = [seg]
            current_words = seg_words
        else:
            current_segs.append(seg)
            current_words += seg_words
            
    if current_segs:
        chunks.append({
            "id": chunk_idx,
            "start": current_segs[0]["start"],
            "end": current_segs[-1]["end"],
            "segments": current_segs,
            "text": " ".join([s["text"] for s in current_segs]),
            "words": current_words
        })
        
    return chunks


def sanitize_filename(name: str) -> str:
    """Limpia caracteres inválidos en Windows para nombres de archivo."""
    invalid_chars = '<>:"/\\|?*'
    clean = name
    for c in invalid_chars:
        clean = clean.replace(c, "_")
    return clean.strip()


def run_llm_postprocessing(base_name: str, transcription_data: dict) -> dict:
    """Ejecuta el pipeline jerárquico de análisis y síntesis en LM Studio."""
    segments = transcription_data["segments"]
    duration = transcription_data.get("duration", 0)
    
    logger.info(f"[{base_name}] Iniciando post-procesamiento jerárquico con LM Studio ({LM_STUDIO_MODEL})...")
    chunks = create_dynamic_chunks(segments, max_words=3600)
    logger.info(f"[{base_name}] Dividido en {len(chunks)} bloques dinámicos para inferencia.")
    
    cache_file = OUTPUT_DIR / f"{sanitize_filename(base_name)}_blocks.json"
    cached_blocks = {}
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached_blocks = json.load(f)
        except Exception:
            cached_blocks = {}
            
    block_summaries = []
    system_block_prompt = (
        "Eres un analista y perito técnico en seguridad de la información (ISO/IEC 27001). "
        "Analiza el fragmento de transcripción de la clase universitaria/técnica y produce una síntesis clara y estructurada. "
        "Normas obligatorias:\n"
        "- Idioma: Español latino estrictamente neutro.\n"
        "- Tono: Formal, técnico, sobrio y pedagógico.\n"
        "- Prohibición estricta de guiones largos (—).\n"
        "- Identifica conceptos normativos, ejemplos prácticos dados por el docente, herramientas mostradas y dudas de alumnos."
    )
    
    for ch in chunks:
        ch_id = ch["id"]
        t_start = ch["start"]
        t_end = ch["end"]
        words_count = ch["words"]
        time_range = f"{format_seconds(t_start)} - {format_seconds(t_end)}"
        
        if str(ch_id) in cached_blocks:
            logger.info(f"[{base_name}] Bloque {ch_id}/{len(chunks)} [{time_range}]: Recuperado desde caché local.")
            block_summaries.append(cached_blocks[str(ch_id)])
            continue
            
        logger.info(f"[{base_name}] Analizando Bloque {ch_id}/{len(chunks)} [{time_range}] ({words_count} palabras)...")
        prompt = (
            f"Fragmento de transcripción de clase [{time_range}]:\n\n"
            f"{ch['text']}\n\n"
            "Instrucción de análisis:\n"
            "Elabora un resumen técnico conciso de este bloque:\n"
            "1. Temas centrales y desarrollo metodológico.\n"
            "2. Conceptos normativos explicados (cláusulas ISO 27001, marcos de seguridad, definiciones).\n"
            "3. Metodologías prácticas, herramientas y ejercicios mostrados.\n"
            "4. Consultas clave de estudiantes y aclaraciones dadas."
        )
        
        t0 = time.time()
        try:
            summary = query_llm(prompt, system_block_prompt, max_tokens=900)
            summary = clean_anti_slop(summary)
            elapsed = round(time.time() - t0, 1)
            logger.info(f"[{base_name}] [+] Bloque {ch_id} analizado en {elapsed}s.")
            
            b_data = {
                "block_id": ch_id,
                "time_range": time_range,
                "analysis": summary
            }
            block_summaries.append(b_data)
            cached_blocks[str(ch_id)] = b_data
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cached_blocks, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"[{base_name}] [!] Error procesando bloque {ch_id}: {e}")
            raise
            
    # Reducción jerárquica de dos mitades
    mid_idx = len(block_summaries) // 2
    half1 = block_summaries[:mid_idx]
    half2 = block_summaries[mid_idx:]
    
    half1_text = "\n\n".join([f"### Bloque {b['block_id']} [{b['time_range']}]\n{b['analysis']}" for b in half1])
    half2_text = "\n\n".join([f"### Bloque {b['block_id']} [{b['time_range']}]\n{b['analysis']}" for b in half2])
    
    system_intermediate = (
        "Eres un auditor senior y docente de ISO/IEC 27001. "
        "Sintetiza la fase de una clase técnica en un resumen consolidado estructurado en español neutro, sin guiones largos."
    )
    
    logger.info(f"[{base_name}] Sintetizando Etapa 1 (Bloques 1 al {mid_idx})...")
    p_h1 = (
        f"A partir de los análisis de la primera mitad de la clase '{base_name}':\n\n"
        f"{half1_text}\n\n"
        "Genera un resumen consolidado con los pilares abordados, metodologías explicadas y conclusiones de esta primera etapa."
    )
    synth_h1 = clean_anti_slop(query_llm(p_h1, system_intermediate, max_tokens=1000))
    
    logger.info(f"[{base_name}] Sintetizando Etapa 2 (Bloques {mid_idx+1} al {len(block_summaries)})...")
    p_h2 = (
        f"A partir de los análisis de la segunda mitad de la clase '{base_name}':\n\n"
        f"{half2_text}\n\n"
        "Genera un resumen consolidado con los temas avanzados, prácticas, ejercicios, dudas resueltas y cierre."
    )
    synth_h2 = clean_anti_slop(query_llm(p_h2, system_intermediate, max_tokens=1000))
    
    # Síntesis Global Final (Resumen Ejecutivo y Minuta Oficial)
    logger.info(f"[{base_name}] Compilando Resumen Ejecutivo Global y Minuta Oficial...")
    combined_halves = (
        f"### ETAPA 1: FUNDAMENTOS Y DESARROLLO INICIAL\n\n{synth_h1}\n\n"
        f"### ETAPA 2: APLICACIÓN PRÁCTICA, EJERCICIOS Y CIERRE\n\n{synth_h2}"
    )
    
    system_exec = (
        "Eres un Director de Seguridad de la Información (CISO) y auditor líder ISO 27001. "
        "Redacta un resumen ejecutivo de alto nivel corporativo y académico en español latino neutro sin clichés ni guiones largos."
    )
    prompt_exec = (
        f"A partir de la síntesis consolidada de la clase de {format_seconds(duration)} '{base_name}' (ISO 27001):\n\n"
        f"{combined_halves}\n\n"
        "Redacta el Resumen Ejecutivo Global con:\n"
        f"1. Propósito estratégico de la sesión '{base_name}'.\n"
        "2. Fundamentos normativos según ISO/IEC 27001 y metodologías de gestión.\n"
        "3. Pilares desarrollados y dinámicas prácticas implementadas.\n"
        "4. Conclusiones y nivel de preparación técnico alcanzado por los estudiantes."
    )
    executive_summary = clean_anti_slop(query_llm(prompt_exec, system_exec, max_tokens=1200))
    
    system_minutes = (
        "Eres el Secretario Técnico y Relator Académico de la certificación de seguridad. "
        "Redacta actas formales de sesiones técnicas con precisión operativa, sin clichés ni guiones largos."
    )
    prompt_minutes = (
        f"A partir de la síntesis consolidada de la sesión sobre ISO 27001 '{base_name}':\n\n"
        f"{combined_halves}\n\n"
        "Redacta el Acta / Minuta Oficial de Sesión Académica-Técnica con la siguiente estructura formal:\n"
        "1. Identificación y Objetivo General de la Sesión\n"
        "2. Resumen Cronológico de las Fases de la Sesión\n"
        "3. Decisiones Metodológicas y Acuerdos de Trabajo\n"
        "4. Tareas, Compromisos y Preparación para la Siguiente Clase"
    )
    meeting_minutes = clean_anti_slop(query_llm(prompt_minutes, system_minutes, max_tokens=1400))
    
    return {
        "summary": executive_summary,
        "meeting_minutes": meeting_minutes,
        "block_analyses": block_summaries,
        "stages": {"stage_1": synth_h1, "stage_2": synth_h2}
    }


def process_single_video(
    mp4_path: Path,
    processor: AudioProcessor,
    transcriber: WhisperTranscriber,
    exporter: ExportManager
):
    """Procesa un archivo MP4 completo: extracción, transcripción, LLM, exportación y copia."""
    raw_name = mp4_path.stem.strip()
    clean_name = sanitize_filename(raw_name)
    target_docx_in_folder = mp4_path.parent / f"{raw_name}.docx"
    clean_docx_in_folder = mp4_path.parent / f"{clean_name}.docx"
    
    logger.info("=" * 70)
    logger.info(f"PROCESANDO: {raw_name}")
    logger.info(f"Carpeta origen: {mp4_path.parent}")
    logger.info("=" * 70)
    
    # 0. Verificación de existencia previa en la carpeta destino
    if target_docx_in_folder.exists() and target_docx_in_folder.stat().st_size > 50000:
        logger.info(f"[OK] El archivo DOCX ya existe en la subcarpeta del video ({target_docx_in_folder.name}). Omitiendo proceso.")
        return
    if clean_docx_in_folder.exists() and clean_docx_in_folder.stat().st_size > 50000:
        logger.info(f"[OK] El archivo DOCX ya existe en la subcarpeta del video ({clean_docx_in_folder.name}). Omitiendo proceso.")
        return

    # 1. Extracción de audio con FFmpeg
    logger.info("[1/4] Extrayendo audio a 16kHz mono WAV...")
    wav_path, is_temp = processor.extract_audio(str(mp4_path))
    
    try:
        # 2. Transcripción con Faster-Whisper
        logger.info(f"[2/4] Transcribiendo con Faster-Whisper ({transcriber.model_size} en {transcriber.device})...")
        t0 = time.time()
        t_data = transcriber.transcribe(wav_path, language="es", word_timestamps=True)
        t_time = round(time.time() - t0, 2)
        dur = t_data.get("duration", 0)
        speed = round(dur / t_time, 1) if t_time > 0 else 0
        total_words = len(t_data["text"].split())
        logger.info(
            f"[+] Transcripción completada en {t_time}s ({speed}x tiempo real). "
            f"Segmentos: {len(t_data['segments'])} | Palabras: {total_words}"
        )
        
        # Guardar inmediatamente archivos base para respaldo
        exporter.export_txt(clean_name, t_data["text"], segments=t_data["segments"], include_timestamps=True)
        exporter.export_srt(clean_name, t_data["segments"])
        exporter.export_vtt(clean_name, t_data["segments"])
        
        # 3. Post-procesamiento con LLM
        logger.info("[3/4] Iniciando post-procesamiento con LLM local...")
        llm_results = run_llm_postprocessing(clean_name, t_data)
        
        # 4. Exportar entregables consolidados
        logger.info("[4/4] Generando entregables finales consolidados...")
        chronological_breakdown = "\n\n".join([
            f"### Bloque {b['block_id']} [{b['time_range']}]\n\n{b['analysis']}"
            for b in llm_results["block_analyses"]
        ])
        full_summary = f"{llm_results['summary']}\n\n## Desglose Cronológico por Bloques Horarios\n\n{chronological_breakdown}"
        
        exported = exporter.export_all(
            clean_name,
            t_data,
            summary=full_summary,
            meeting_minutes=llm_results["meeting_minutes"]
        )
        for fmt, path in exported.items():
            logger.info(f"  [+] Generado en output/ ({fmt.upper()}): {Path(path).name}")
            
        # 5. Copiar el DOCX generado a la subcarpeta del video
        docx_source = Path(exported["docx"])
        if docx_source.exists():
            target_copy = mp4_path.parent / f"{raw_name}.docx"
            shutil.copy2(docx_source, target_copy)
            logger.info(f"[✓] DOCX copiado exitosamente a la subcarpeta del video:")
            logger.info(f"    -> {target_copy}")
        else:
            logger.error(f"[!] No se encontró el archivo DOCX generado en {docx_source}")
            
    finally:
        # Limpieza de archivo temporal de audio
        if is_temp and os.path.exists(wav_path):
            try:
                os.unlink(wav_path)
                logger.info("[-] Archivo temporal de audio eliminado.")
            except Exception:
                pass


def main():
    root_dir = Path("G:/Mi unidad/Estudios/ISO 27001/Hacker Mentor")
    if not root_dir.exists():
        logger.error(f"La ruta especificada no existe: {root_dir}")
        sys.exit(1)
        
    logger.info("======================================================================")
    logger.info("INICIANDO PROCESAMIENTO POR LOTES AUTÓNOMO (ISO 27001 - HACKER MENTOR)")
    logger.info(f"Ruta base: {root_dir}")
    logger.info(f"Backend LLM: LM Studio en {LM_STUDIO_URL} ({LM_STUDIO_MODEL})")
    logger.info(f"Whisper Model: small (CUDA float16 en GPU Nvidia RTX 3070)")
    logger.info("======================================================================")
    
    # 1. Identificar todos los videos MP4 recursivamente
    all_videos = sorted(list(root_dir.rglob("*.mp4")))
    logger.info(f"Total de videos encontrados: {len(all_videos)}")
    for idx, v in enumerate(all_videos, 1):
        sz = round(v.stat().st_size / (1024 * 1024), 1)
        logger.info(f"  {idx}. [{v.parent.name}] {v.name} ({sz} MB)")
        
    # 2. Inicializar componentes una sola vez para reutilizar en todos los videos
    logger.info("\n[*] Inicializando AudioProcessor, WhisperTranscriber y ExportManager...")
    processor = AudioProcessor()
    transcriber = WhisperTranscriber(model_size="small", device="cuda", compute_type="float16")
    exporter = ExportManager(output_dir=str(OUTPUT_DIR))
    
    total_start_time = time.time()
    
    # 3. Iterar secuencialmente por cada video
    for idx, video_path in enumerate(all_videos, 1):
        logger.info(f"\n>>> PROCESANDO VIDEO {idx}/{len(all_videos)}: {video_path.name}")
        try:
            process_single_video(video_path, processor, transcriber, exporter)
        except Exception as e:
            logger.error(f"[!] Error crítico procesando {video_path.name}: {e}", exc_info=True)
            logger.info("Continuando con el siguiente video en la cola...")
            
    total_elapsed = round(time.time() - total_start_time, 1)
    logger.info("\n" + "=" * 70)
    logger.info(f"[✓] PROCESAMIENTO POR LOTES FINALIZADO EN {format_seconds(total_elapsed)}")
    logger.info("======================================================================")


if __name__ == "__main__":
    main()
