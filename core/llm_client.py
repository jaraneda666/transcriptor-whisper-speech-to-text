"""
Cliente universal para modelos de lenguaje locales (LLM/SLM).
Compatible con la API estándar de OpenAI (/v1) soportada por Ollama y LM Studio.
"""

import requests
import json
import logging
from typing import Optional, Dict, Any, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LocalLLMClient:
    """Cliente unificado para interactuar con Ollama o LM Studio de forma agnóstica."""

    DEFAULT_ENDPOINTS = {
        "ollama": "http://127.0.0.1:1143/v1",
        "ollama_alt": "http://localhost:11434/v1",
        "lm_studio": "http://localhost:1234/v1"
    }

    def __init__(
        self,
        backend: str = "auto",  # 'ollama', 'lm_studio', 'auto' o url personalizada
        api_base: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: int = 120
    ):
        self.timeout = timeout
        self.backend = backend.lower()
        self.api_base = api_base or self._resolve_api_base(self.backend)
        self.model_name = model_name

    def _resolve_api_base(self, backend: str) -> str:
        """Determina la URL base según el backend seleccionado o autodetecta el activo."""
        if backend.startswith("http://") or backend.startswith("https://"):
            return backend.rstrip("/")

        if backend == "ollama":
            for url in ["http://127.0.0.1:1143/v1", "http://127.0.0.1:11434/v1", "http://localhost:11434/v1"]:
                if self._test_connection(url):
                    return url
            return "http://127.0.0.1:1143/v1"
        if backend == "lm_studio":
            return self.DEFAULT_ENDPOINTS["lm_studio"]

        # Si es 'auto', chequear primero Ollama y luego LM Studio
        for b_name, url in self.DEFAULT_ENDPOINTS.items():
            if self._test_connection(url):
                logger.info(f"Backend detectado automáticamente: {b_name} en {url}")
                self.backend = "ollama" if "ollama" in b_name else b_name
                return url

        # Por defecto dejar Ollama si ninguno responde
        return self.DEFAULT_ENDPOINTS["ollama"]

    def _test_connection(self, url: str) -> bool:
        """Prueba rápida de conectividad hacia el endpoint /v1/models."""
        try:
            r = requests.get(f"{url}/models", timeout=1.5)
            return r.status_code == 200
        except Exception:
            return False

    def check_health(self) -> Dict[str, Any]:
        """Verifica el estado del backend actual y de los backends conocidos."""
        results: Dict[str, Any] = {
            "current_backend": self.backend,
            "current_url": self.api_base,
            "current_active": False,
            "ollama_active": self._test_connection(self.DEFAULT_ENDPOINTS["ollama"]),
            "lm_studio_active": self._test_connection(self.DEFAULT_ENDPOINTS["lm_studio"]),
            "available_models": []
        }
        results["current_active"] = self._test_connection(self.api_base)
        if results["current_active"]:
            results["available_models"] = self.list_models()
        return results

    def list_models(self) -> List[str]:
        """Obtiene la lista de modelos disponibles en el servidor local."""
        try:
            r = requests.get(f"{self.api_base}/models", timeout=4)
            if r.status_code == 200:
                data = r.json()
                models_list = data.get("data", [])
                return [m.get("id") or m.get("name") for m in models_list if m.get("id") or m.get("name")]
        except Exception as e:
            logger.warning(f"Error al listar modelos en {self.api_base}: {e}")
        return []

    def chat_completion(
        self,
        prompt: str,
        system_instruction: str = "",
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096
    ) -> str:
        """
        Envía una solicitud de chat completion al endpoint /v1/chat/completions.
        """
        target_model = model or self.model_name
        if not target_model:
            available = self.list_models()
            if available:
                target_model = available[0]
            else:
                target_model = "default"

        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": target_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        headers = {"Content-Type": "application/json"}
        url = f"{self.api_base}/chat/completions"

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            if response.status_code != 200:
                raise RuntimeError(
                    f"Error HTTP {response.status_code} desde {self.api_base}: {response.text}"
                )
            res_json = response.json()
            choices = res_json.get("choices", [])
            if not choices:
                raise RuntimeError("El modelo respondió sin opciones de texto.")
            return choices[0].get("message", {}).get("content", "").strip()
        except requests.exceptions.RequestException as e:
            raise ConnectionError(
                f"No fue posible conectar con el servidor LLM en {self.api_base}. "
                f"Asegúrese de que Ollama o LM Studio esté iniciado. Detalle: {e}"
            )

    def correct_and_format(self, transcript_text: str, model: Optional[str] = None) -> str:
        """
        Corrige la puntuación, ortografía y segmenta en párrafos lógicos el texto.
        Mantiene estrictamente intactos los datos, términos técnicos y nombres propios.
        """
        system_msg = (
            "Eres un editor pericial y transcriptor profesional en idioma español. "
            "Tu tarea es corregir la puntuación, ortografía, acentuación y organizar en párrafos legibles "
            "la transcripción de audio que recibes. "
            "Normas estrictas:\n"
            "1. NO resumas ni elimines información.\n"
            "2. NO inventes hechos, nombres ni datos que no estén en la transcripción.\n"
            "3. Preserva fielmente todos los nombres técnicos, siglas y números.\n"
            "4. No uses guiones largos (—); utiliza puntos o comas.\n"
            "5. Devuelve únicamente el texto corregido sin saludos, preámbulos ni comentarios adicionales."
        )
        return self.chat_completion(transcript_text, system_instruction=system_msg, model=model, temperature=0.1)

    def summarize(self, transcript_text: str, model: Optional[str] = None) -> str:
        """
        Genera un resumen ejecutivo y puntos clave del contenido.
        """
        system_msg = (
            "Eres un analista técnico especializado en síntesis de información en español. "
            "Tu tarea es generar un resumen formal y estructurado de la transcripción proporcionada.\n"
            "Normas de estilo:\n"
            "- Idioma: Español latino estrictamente neutro.\n"
            "- Tono: Formal, sobrio, analítico y directo.\n"
            "- No emplees clichés como 'es importante destacar que', 'cabe mencionar' ni contrastes 'no X, sino Y'.\n"
            "- No uses guiones largos (—).\n"
            "Estructura esperada:\n"
            "### Resumen Ejecutivo\n"
            "(Párrafo claro y directo explicando el tema principal)\n\n"
            "### Puntos Clave y Decisiones\n"
            "(Lista de viñetas con los temas concretos tratados)\n\n"
            "### Conclusiones y Próximos Pasos\n"
            "(Acciones o reflexiones finales mencionadas en la grabación)"
        )
        return self.chat_completion(transcript_text, system_instruction=system_msg, model=model, temperature=0.2)

    def generate_meeting_minutes(self, transcript_text: str, model: Optional[str] = None) -> str:
        """
        Genera una minuta estructurada de reunión o clase a partir de la transcripción.
        """
        system_msg = (
            "Eres un redactor técnico de actas y minutas de reuniones de alto nivel en español. "
            "Convierte la siguiente transcripción en una minuta profesional y operativa.\n"
            "Estructura obligatoria:\n"
            "# Minuta de Sesión\n"
            "## 1. Objetivo General\n"
            "## 2. Temas Tratados\n"
            "## 3. Acuerdos y Decisiones\n"
            "## 4. Tareas Asignadas y Próximas Acciones\n\n"
            "Prohibición de grandilocuencias, clichés de IA y guiones largos. Registro sobrio y formal."
        )
        return self.chat_completion(transcript_text, system_instruction=system_msg, model=model, temperature=0.2)
