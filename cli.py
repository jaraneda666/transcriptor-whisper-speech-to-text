"""
Interfaz de Línea de Comandos (CLI) para Transcriptor_Whisper_LLM.
Permite procesar archivos multimedia individuales o carpetas por lotes.
"""

import sys
import os
import argparse
import time
from pathlib import Path
from typing import Optional

# Asegurar importación del paquete core
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.audio_processor import AudioProcessor
from core.transcriber import WhisperTranscriber
from core.llm_client import LocalLLMClient
from core.export_manager import ExportManager


def process_file(
    file_path: Path,
    processor: AudioProcessor,
    transcriber: WhisperTranscriber,
    exporter: ExportManager,
    llm_client: Optional[LocalLLMClient] = None,
    language: Optional[str] = None,
    enable_summary: bool = False,
    enable_minutes: bool = False,
    enable_correction: bool = False,
    formats: str = "all"
):
    print(f"\n=======================================================")
    print(f"[*] Procesando: {file_path.name}")
    print(f"=======================================================")

    media_info = processor.get_media_info(str(file_path))
    dur_str = f"{media_info['duration_seconds']}s" if media_info['duration_seconds'] else "desconocida"
    print(f"[i] Tamaño: {media_info['filesize_mb']} MB | Duración: {dur_str}")

    # 1. Extracción de audio con FFmpeg
    print("[1/4] Extrayendo y normalizando audio a 16kHz mono...")
    wav_path, is_temp = processor.extract_audio(str(file_path))

    try:
        # 2. Transcripción con Faster-Whisper
        print(f"[2/4] Ejecutando transcripción con Faster-Whisper ({transcriber.model_size} en {transcriber.device})...")
        t_start = time.time()
        t_data = transcriber.transcribe(wav_path, language=language, word_timestamps=True)
        t_dur = round(time.time() - t_start, 2)
        print(f"[+] Transcripción completada en {t_dur}s. Idioma detectado: {t_data['language']} ({round(t_data['language_probability']*100, 1)}%)")
        print(f"    Texto transcrito: {len(t_data['segments'])} segmentos | {len(t_data['text'].split())} palabras.")

        summary_text = None
        minutes_text = None
        corrected_text = None

        # 3. Post-procesamiento con LLM local si está activado
        if llm_client:
            print("[3/4] Ejecutando post-procesamiento con LLM local...")
            if enable_correction:
                print("  -> Corrigiendo puntuación y ortografía...")
                try:
                    corrected_text = llm_client.correct_and_format(t_data["text"])
                    print("  [+] Corrección finalizada.")
                except Exception as e:
                    print(f"  [!] Falló la corrección con LLM: {e}")

            if enable_summary:
                print("  -> Generando resumen ejecutivo y puntos clave...")
                try:
                    summary_text = llm_client.summarize(t_data["text"])
                    print("  [+] Resumen generado.")
                except Exception as e:
                    print(f"  [!] Falló el resumen con LLM: {e}")

            if enable_minutes:
                print("  -> Generando minuta formal de reunión...")
                try:
                    minutes_text = llm_client.generate_meeting_minutes(t_data["text"])
                    print("  [+] Minuta generada.")
                except Exception as e:
                    print(f"  [!] Falló la minuta con LLM: {e}")
        else:
            print("[3/4] Post-procesamiento LLM omitido.")

        # 4. Exportación en formatos solicitados
        print("[4/4] Exportando archivos generados...")
        base_name = file_path.stem

        if formats == "all":
            exported = exporter.export_all(
                base_name,
                t_data,
                summary=summary_text,
                meeting_minutes=minutes_text,
                corrected_text=corrected_text
            )
            for fmt, path in exported.items():
                print(f"  [+] Generado ({fmt.upper()}): {path}")
        else:
            fmt_list = [f.strip().lower() for f in formats.split(",")]
            if "txt" in fmt_list:
                p = exporter.export_txt(base_name, t_data["text"], segments=t_data["segments"], include_timestamps=True)
                print(f"  [+] TXT: {p}")
            if "srt" in fmt_list:
                p = exporter.export_srt(base_name, t_data["segments"])
                print(f"  [+] SRT: {p}")
            if "vtt" in fmt_list:
                p = exporter.export_vtt(base_name, t_data["segments"])
                print(f"  [+] VTT: {p}")
            if "json" in fmt_list:
                p = exporter.export_json(base_name, t_data)
                print(f"  [+] JSON: {p}")
            if "md" in fmt_list:
                p = exporter.export_markdown(base_name, t_data, summary_text, minutes_text, corrected_text)
                print(f"  [+] MD: {p}")
            if "docx" in fmt_list:
                p = exporter.export_docx(base_name, t_data, summary_text, minutes_text, corrected_text)
                print(f"  [+] DOCX: {p}")

    finally:
        # Limpieza de archivo temporal de audio
        if is_temp and os.path.exists(wav_path):
            try:
                os.unlink(wav_path)
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(
        description="Transcriptor_Whisper_LLM: Transcripción rápida con Faster-Whisper y post-procesamiento con LLMs locales."
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Ruta al archivo multimedia (video o audio) o directorio para procesamiento por lotes."
    )
    parser.add_argument(
        "--model", "-m",
        default="base",
        choices=WhisperTranscriber.SUPPORTED_MODELS,
        help="Tamaño del modelo Faster-Whisper (por defecto: 'base')."
    )
    parser.add_argument(
        "--device", "-d",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Dispositivo de cómputo (por defecto: 'auto')."
    )
    parser.add_argument(
        "--compute-type", "-c",
        default=None,
        choices=WhisperTranscriber.COMPUTE_TYPES,
        help="Tipo de cuantización (por defecto: float16 en GPU / int8 en CPU)."
    )
    parser.add_argument(
        "--language", "-l",
        default=None,
        help="Código de idioma forzado (ej. 'es', 'en'). Si se omite, se detecta automáticamente."
    )
    parser.add_argument(
        "--llm-backend",
        default="none",
        choices=["none", "ollama", "lm_studio", "auto"],
        help="Backend de LLM local a utilizar para análisis (por defecto: 'none')."
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Nombre del modelo en Ollama o LM Studio (ej. 'llama3', 'mistral')."
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Generar resumen ejecutivo con el LLM."
    )
    parser.add_argument(
        "--minutes",
        action="store_true",
        help="Generar acta/minuta de reunión con el LLM."
    )
    parser.add_argument(
        "--correct",
        action="store_true",
        help="Refinar ortografía y estructurar párrafos con el LLM."
    )
    parser.add_argument(
        "--formats", "-f",
        default="all",
        help="Formatos a exportar separados por coma (txt,srt,vtt,json,md,docx) o 'all' (por defecto)."
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Directorio de destino de los archivos generados (por defecto: Transcriptor_Whisper_LLM/output)."
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[!] Error: La ruta especificada no existe: {args.input}")
        sys.exit(1)

    # 1. Inicializar procesadores
    try:
        processor = AudioProcessor()
    except Exception as e:
        print(f"[!] Error inicializando AudioProcessor: {e}")
        sys.exit(1)

    print(f"[*] Cargando modelo Whisper '{args.model}' (Dispositivo: {args.device})...")
    try:
        transcriber = WhisperTranscriber(
            model_size=args.model,
            device=args.device,
            compute_type=args.compute_type
        )
    except Exception as e:
        print(f"[!] Error cargando Faster-Whisper: {e}")
        sys.exit(1)

    exporter = ExportManager(output_dir=args.output_dir)

    llm_client = None
    if args.llm_backend != "none":
        print(f"[*] Conectando con LLM local ({args.llm_backend})...")
        llm_client = LocalLLMClient(backend=args.llm_backend, model_name=args.llm_model)
        health = llm_client.check_health()
        if not health["current_active"]:
            print(f"[!] Advertencia: No se pudo conectar al endpoint LLM en {health['current_url']}.")
            print("    El procesamiento continuará sin las funciones del LLM.")
            llm_client = None
        else:
            models_avail = health["available_models"]
            print(f"[+] Conectado a LLM local. Modelos disponibles: {models_avail if models_avail else 'N/A'}")

    # 2. Identificar archivos a procesar
    files_to_process = []
    if input_path.is_file():
        if processor.is_supported(str(input_path)):
            files_to_process.append(input_path)
        else:
            print(f"[!] Extensión no soportada: {input_path.suffix}")
            sys.exit(1)
    elif input_path.is_dir():
        for f in input_path.iterdir():
            if f.is_file() and processor.is_supported(str(f)):
                files_to_process.append(f)

    if not files_to_process:
        print(f"[!] No se encontraron archivos multimedia válidos para procesar en: {input_path}")
        sys.exit(1)

    print(f"[+] Se encontraron {len(files_to_process)} archivo(s) para transcribir.")

    # 3. Procesar en bucle
    for f in files_to_process:
        process_file(
            file_path=f,
            processor=processor,
            transcriber=transcriber,
            exporter=exporter,
            llm_client=llm_client,
            language=args.language,
            enable_summary=args.summary,
            enable_minutes=args.minutes,
            enable_correction=args.correct,
            formats=args.formats
        )

    print("\n[OK] Procesamiento completado exitosamente.")
    print(f"[OK] Todos los entregables están guardados en: {exporter.output_dir.resolve()}\n")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
