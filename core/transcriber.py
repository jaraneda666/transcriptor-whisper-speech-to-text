"""
Módulo de transcripción automática del habla usando Faster-Whisper (CTranslate2).
Optimizado para bajo consumo de VRAM y alta velocidad con aceleración GPU/CUDA o CPU.
"""

import time
import logging
from typing import Optional, Dict, Any, List, Generator
from pathlib import Path

import sys
import os

def _setup_cuda_dll_paths():
    """Agrega las carpetas bin de librerías NVIDIA instaladas por pip al PATH y DLL search path en Windows."""
    if sys.platform != "win32":
        return
    for p in sys.path:
        base_nvidia = Path(p) / "nvidia"
        if base_nvidia.is_dir():
            for sub in base_nvidia.iterdir():
                bin_dir = sub / "bin"
                if bin_dir.is_dir():
                    try:
                        os.add_dll_directory(str(bin_dir.resolve()))
                    except Exception:
                        pass
                    if str(bin_dir) not in os.environ.get("PATH", ""):
                        os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")

_setup_cuda_dll_paths()

try:
    import av
    # Parche de compatibilidad para versiones modernas de PyAV (av>=14)
    # donde el parámetro 'metadata_errors' fue retirado de av.open
    _original_av_open = av.open
    def _safe_av_open(*args, **kwargs):
        kwargs.pop("metadata_errors", None)
        return _original_av_open(*args, **kwargs)
    av.open = _safe_av_open
except Exception:
    pass

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WhisperTranscriber:
    """Encapsula el modelo Faster-Whisper y gestiona la inferencia eficiente."""

    SUPPORTED_MODELS = [
        "tiny", "tiny.en",
        "base", "base.en",
        "small", "small.en",
        "medium", "medium.en",
        "large-v2", "large-v3", "large-v3-turbo"
    ]

    COMPUTE_TYPES = ["int8", "float16", "int8_float16", "default"]

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: Optional[str] = None,
        cpu_threads: int = 4,
        download_root: Optional[str] = None
    ):
        """
        Inicializa el modelo de Faster-Whisper.
        :param model_size: Nombre del modelo (tiny, base, small, medium, large-v3).
        :param device: 'cuda', 'cpu' o 'auto'.
        :param compute_type: 'int8', 'float16', 'int8_float16' o None (auto-determinado).
        """
        if not FASTER_WHISPER_AVAILABLE:
            raise ImportError(
                "La biblioteca 'faster-whisper' no está instalada. "
                "Instálela con: pip install faster-whisper"
            )

        self.model_size = model_size
        self.device = self._resolve_device(device)
        self.compute_type = compute_type or self._resolve_compute_type(self.device)
        self.cpu_threads = cpu_threads
        self.download_root = download_root

        logger.info(
            f"Cargando modelo Faster-Whisper: '{self.model_size}' "
            f"en dispositivo '{self.device}' con compute_type='{self.compute_type}'"
        )

        start_load = time.time()
        self.model = WhisperModel(
            model_size_or_path=self.model_size,
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=self.cpu_threads,
            download_root=self.download_root
        )
        self.load_time = round(time.time() - start_load, 2)
        logger.info(f"Modelo cargado exitosamente en {self.load_time} segundos.")

    def _resolve_device(self, device: str) -> str:
        """Determina el dispositivo óptimo si se especifica 'auto'."""
        if device.lower() != "auto":
            return device.lower()

        try:
            import ctranslate2
            cuda_count = ctranslate2.get_cuda_device_count()
            if cuda_count > 0:
                return "cuda"
        except Exception:
            pass

        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass

        return "cpu"

    def _resolve_compute_type(self, device: str) -> str:
        """Asigna el formato de cuantización según el dispositivo para optimizar memoria."""
        if device == "cuda":
            # float16 o int8_float16 para máxima velocidad y menor VRAM en RTX 3070
            return "float16"
        return "int8"  # En CPU, int8 ofrece la mejor velocidad con AVX

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe",
        beam_size: int = 5,
        vad_filter: bool = True,
        word_timestamps: bool = True,
        show_progress: bool = True,
        progress_callback: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Transcribe un archivo de audio y devuelve el resultado completo y estructurado.
        """
        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"Archivo de audio no encontrado: {audio_path}")

        start_time = time.time()

        # Inferencia con Faster-Whisper
        segments_generator, info = self.model.transcribe(
            str(audio_file.resolve()),
            language=language,
            task=task,
            beam_size=beam_size,
            vad_filter=vad_filter,
            word_timestamps=word_timestamps
        )

        total_duration = getattr(info, "duration", None) or 1.0

        segments_list: List[Dict[str, Any]] = []
        full_text_parts: List[str] = []

        for segment in segments_generator:
            seg_dict: Dict[str, Any] = {
                "id": segment.id,
                "seek": segment.seek,
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
                "text": segment.text.strip(),
                "avg_logprob": round(segment.avg_logprob, 4),
                "no_speech_prob": round(segment.no_speech_prob, 4),
                "words": []
            }

            if show_progress:
                pct = min(100.0, round((segment.end / total_duration) * 100, 1))
                m_curr, s_curr = int(segment.end // 60), int(segment.end % 60)
                m_tot, s_tot = int(total_duration // 60), int(total_duration % 60)
                clean_preview = segment.text.strip().replace("\n", " ")[:40]
                print(f"\r  [>>] {pct:5.1f}% [{m_curr:02d}:{s_curr:02d} / {m_tot:02d}:{s_tot:02d}] | {clean_preview}...", end="", flush=True)

            if progress_callback:
                try:
                    progress_callback(segment, total_duration)
                except Exception:
                    pass

            if word_timestamps and segment.words:
                seg_dict["words"] = [
                    {
                        "start": round(w.start, 3),
                        "end": round(w.end, 3),
                        "word": w.word,
                        "probability": round(w.probability, 4)
                    }
                    for w in segment.words
                ]

            segments_list.append(seg_dict)
            full_text_parts.append(segment.text.strip())

        if show_progress:
            print()

        elapsed_time = round(time.time() - start_time, 2)
        full_text = " ".join(full_text_parts).strip()

        return {
            "text": full_text,
            "segments": segments_list,
            "language": info.language,
            "language_probability": round(info.language_probability, 4),
            "duration": round(info.duration, 2) if hasattr(info, "duration") else None,
            "transcription_time_seconds": elapsed_time,
            "model_used": self.model_size,
            "device_used": self.device,
            "compute_type": self.compute_type
        }
