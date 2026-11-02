"""
Módulo de extracción y procesamiento de audio con FFmpeg.
Normaliza audio a formato PCM 16kHz mono compatible de forma óptima con Whisper.
"""

import os
import subprocess
import shutil
import tempfile
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple


class AudioProcessor:
    """Gestiona la extracción de pistas de audio y obtención de metadatos mediante FFmpeg."""

    SUPPORTED_EXTENSIONS = {
        ".mp4", ".mkv", ".avi", ".mov", ".flv", ".webm", ".wmv", ".m4v",
        ".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".wma", ".opus"
    }

    def __init__(self, ffmpeg_path: Optional[str] = None):
        """Inicializa el procesador resolviendo la ruta del binario FFmpeg."""
        self.ffmpeg_cmd = ffmpeg_path or shutil.which("ffmpeg")
        if not self.ffmpeg_cmd:
            raise FileNotFoundError(
                "No se encontró el ejecutable FFmpeg en el sistema ni en el PATH. "
                "Asegúrese de tener FFmpeg instalado para continuar."
            )
        self.ffprobe_cmd = shutil.which("ffprobe")

    def is_supported(self, file_path: str) -> bool:
        """Verifica si la extensión del archivo es soportada."""
        return Path(file_path).suffix.lower() in self.SUPPORTED_EXTENSIONS

    def get_media_info(self, file_path: str) -> Dict[str, Any]:
        """Obtiene metadatos básicos del archivo (duración, tamaño, canales)."""
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"El archivo especificado no existe: {file_path}")

        info: Dict[str, Any] = {
            "filename": file_path_obj.name,
            "filepath": str(file_path_obj.resolve()),
            "filesize_bytes": file_path_obj.stat().st_size,
            "filesize_mb": round(file_path_obj.stat().st_size / (1024 * 1024), 2),
            "extension": file_path_obj.suffix.lower(),
            "duration_seconds": None,
        }

        # Intentar obtener duración precisa con ffprobe si está disponible
        if self.ffprobe_cmd:
            try:
                cmd = [
                    self.ffprobe_cmd,
                    "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "json",
                    str(file_path_obj)
                ]
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
                probe_data = json.loads(result.stdout)
                duration = float(probe_data.get("format", {}).get("duration", 0))
                info["duration_seconds"] = round(duration, 2)
            except Exception:
                pass

        return info

    def extract_audio(
        self,
        input_path: str,
        output_wav_path: Optional[str] = None,
        sample_rate: int = 16000,
        channels: int = 1
    ) -> Tuple[str, bool]:
        """
        Extrae la pista de audio hacia un archivo WAV a 16kHz mono.
        Retorna la ruta del archivo WAV generado y un booleano indicando si es temporal.
        """
        input_obj = Path(input_path)
        if not input_obj.exists():
            raise FileNotFoundError(f"Archivo de entrada no encontrado: {input_path}")

        is_temp = False
        if not output_wav_path:
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            output_wav_path = temp_file.name
            temp_file.close()
            is_temp = True

        out_obj = Path(output_wav_path)
        out_obj.parent.mkdir(parents=True, exist_ok=True)

        # Comando ffmpeg optimizado: sin video (-vn), PCM 16-bit little-endian, sample rate y canales especificados
        cmd = [
            self.ffmpeg_cmd,
            "-y",  # Sobrescribir sin preguntar
            "-i", str(input_obj.resolve()),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", str(sample_rate),
            "-ac", str(channels),
            str(out_obj.resolve())
        ]

        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Error en FFmpeg al extraer audio: {result.stderr}")
        except Exception as e:
            if is_temp and os.path.exists(output_wav_path):
                os.unlink(output_wav_path)
            raise e

        return output_wav_path, is_temp
