"""
Transcriptor_Whisper_LLM: Suite modular de transcripción con Faster-Whisper
y post-procesamiento con modelos de lenguaje locales (Ollama y LM Studio).
"""

from .audio_processor import AudioProcessor
from .transcriber import WhisperTranscriber
from .llm_client import LocalLLMClient
from .export_manager import ExportManager

__all__ = [
    "AudioProcessor",
    "WhisperTranscriber",
    "LocalLLMClient",
    "ExportManager"
]
