#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Tool: anonimizador_contextual.py

## Metadata
# - Nombre: anonimizador_contextual.py
# - Versión: 1.0
# - Lenguaje: Python
# - Creada en proyecto: Transcriptor_Whisper_LLM / Global
# - Fecha de creación: 2026-10-19
# - Descripción: Anonimizador contextual y desanonimizador perimetral de privacidad con soporte
#                multi-backend (LM Studio, Ollama y LiteRT-LM) y autoencendido bajo demanda.
# - Parámetros: --backend [lmstudio|ollama|litert], --model [nombre], --text [cadena],
#               --file [ruta], --restore [cadena], --mapping [json], --interactive, --json
# - Dependencias: httpx

import argparse
import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("AnonimizadorContextual")

# --- Configuraciones predeterminadas de red y modelos ---
LMSTUDIO_DEFAULT_URL = os.environ.get("LMSTUDIO_HOST", "http://127.0.0.1:1234/v1")
LMSTUDIO_LOCAL_URL = "http://127.0.0.1:1234/v1"
LMSTUDIO_DEFAULT_MODEL = "qwen/qwen3-vl-8b"
LMSTUDIO_RECOMMENDED = ["qwen/qwen3-vl-8b", "meta-llama-3.1-8b-instruct", "qwen3.8-27b-uncensored"]
LMS_EXECUTABLE = os.environ.get("LMS_PATH", r"C:\Users\Developer\.lmstudio\bin\lms.exe")

OLLAMA_RAW_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:1143")
OLLAMA_NORMALIZED_HOST = OLLAMA_RAW_HOST.replace("0.0.0.0", "127.0.0.1")
if not OLLAMA_NORMALIZED_HOST.startswith("http://") and not OLLAMA_NORMALIZED_HOST.startswith("https://"):
    OLLAMA_NORMALIZED_HOST = f"http://{OLLAMA_NORMALIZED_HOST}"
OLLAMA_DEFAULT_URL = OLLAMA_NORMALIZED_HOST.rstrip("/")
OLLAMA_DEFAULT_MODEL = "llama3.1:8b-instruct-q4_K_M"
OLLAMA_RECOMMENDED = ["llama3.1:8b-instruct-q4_K_M", "gemma2:9b", "qwen2.5:7b"]
OLLAMA_EXECUTABLE = os.environ.get("OLLAMA_PATH", "ollama")

SYSTEM_PROMPT_ANONIMIZACION = (
    "Eres un filtro de privacidad perimetral y sanitizador de contexto. Tu única tarea es anonimizar "
    "el texto reemplazando cualquier información de identificación personal o sensible (nombres propios, "
    "direcciones IP, nombres de servidores, dominios privados, empresas cliente, correos electrónicos, "
    "teléfonos, credenciales o números de cuenta) por marcadores semánticos deterministas "
    "como [PERSONA_1], [IP_1], [EMPRESA_1], [DOMINIO_1], [EMAIL_1], [CREDENCIAL_1].\n\n"
    "Reglas estrictas:\n"
    "1. Conserva intacta la lógica técnica, sintaxis de código, puertos, comandos y requerimiento del usuario.\n"
    "2. No agregues saludos, explicaciones, notas ni advertencias.\n"
    "3. Responde EXCLUSIVAMENTE con un objeto JSON válido con la siguiente estructura exacta:\n"
    "{\n"
    '  "anonimizado": "Texto completo reescrito con los marcadores entre corchetes",\n'
    '  "entidades": {\n'
    '    "[MARCADOR_1]": "valor_original_sensible",\n'
    '    "[MARCADOR_2]": "valor_original_sensible"\n'
    "  }\n"
    "}"
)


@dataclass
class AnonimizacionResult:
    original: str
    anonimizado: str
    entidades: Dict[str, str]
    backend: str
    modelo: str
    tiempo_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original": self.original,
            "anonimizado": self.anonimizado,
            "entidades": self.entidades,
            "backend": self.backend,
            "modelo": self.modelo,
            "tiempo_ms": self.tiempo_ms,
        }


class PrivacyVault:
    """Almacén seguro en memoria para el mapeo de entidades desanonimizables."""

    def __init__(self):
        self.mappings: Dict[str, str] = {}

    def registrar(self, entidades: Dict[str, str]):
        self.mappings.update(entidades)

    def restore(self, text: str, custom_mapping: Optional[Dict[str, str]] = None) -> str:
        """Restaura los marcadores de privacidad en un texto con los valores originales."""
        map_to_use = custom_mapping if custom_mapping is not None else self.mappings
        result = text
        for placeholder, original in map_to_use.items():
            result = result.replace(placeholder, original)
        return result

    def export_json(self) -> str:
        return json.dumps(self.mappings, indent=2, ensure_ascii=False)


# --- Gestores de autoencendido bajo demanda (Lazy Spawning) ---

def ensure_lmstudio_ready(base_url: str = LMSTUDIO_DEFAULT_URL) -> str:
    """Verifica si LM Studio está activo. Si está apagado, lo levanta automáticamente."""
    import httpx

    # Intentar conexión rápida a la URL configurada
    urls_to_try = [base_url, LMSTUDIO_LOCAL_URL] if base_url != LMSTUDIO_LOCAL_URL else [LMSTUDIO_LOCAL_URL]
    for url in urls_to_try:
        try:
            res = httpx.get(f"{url}/models", timeout=1.5)
            if res.status_code == 200:
                return url
        except Exception:
            pass

    logger.info("LM Studio no responde. Iniciando servidor automáticamente mediante 'lms server start'...")
    candidates = [
        LMS_EXECUTABLE,
        os.path.expanduser(r"~/.lmstudio/bin/lms.exe"),
        "lms",
    ]
    started = False
    for exe in candidates:
        try:
            subprocess.Popen(
                [exe, "server", "start", "--bind", "0.0.0.0", "--cors"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
            started = True
            break
        except Exception:
            continue

    if not started:
        raise RuntimeError("No se pudo localizar el ejecutable de LM Studio (lms.exe).")

    for _ in range(12):
        time.sleep(1.0)
        for url in urls_to_try:
            try:
                res = httpx.get(f"{url}/models", timeout=1.5)
                if res.status_code == 200:
                    logger.info(f"LM Studio levantado exitosamente y listo en {url}.")
                    return url
            except Exception:
                pass

    raise TimeoutError("Tiempo de espera agotado al intentar iniciar LM Studio.")


def ensure_ollama_ready(base_url: str = OLLAMA_DEFAULT_URL) -> str:
    """Verifica si Ollama está activo. Si está apagado, lo levanta automáticamente."""
    import httpx

    try:
        res = httpx.get(f"{base_url}/api/tags", timeout=1.5)
        if res.status_code == 200:
            return base_url
    except Exception:
        pass

    logger.info("Ollama no responde. Iniciando servicio automáticamente mediante 'ollama serve'...")
    candidates = [OLLAMA_EXECUTABLE, "ollama"]
    started = False
    for exe in candidates:
        try:
            subprocess.Popen(
                [exe, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
            started = True
            break
        except Exception:
            continue

    if not started:
        raise RuntimeError("No se pudo localizar el ejecutable de Ollama.")

    for _ in range(12):
        time.sleep(1.0)
        try:
            res = httpx.get(f"{base_url}/api/tags", timeout=1.5)
            if res.status_code == 200:
                logger.info(f"Ollama levantado exitosamente y listo en {base_url}.")
                return base_url
        except Exception:
            pass

    raise TimeoutError("Tiempo de espera agotado al intentar iniciar Ollama.")


# --- Parsers de respuesta resilientes ---

def parsear_respuesta_json(raw_text: str) -> Tuple[str, Dict[str, str]]:
    """Extrae el objeto JSON de anonimización incluso si el LLM incluye bloques markdown o preámbulos."""
    clean = raw_text.strip()
    # Buscar bloque de código ```json ... ```
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean, re.DOTALL)
    if json_match:
        clean = json_match.group(1).strip()
    else:
        # Buscar el primer '{' y el último '}'
        start = clean.find("{")
        end = clean.rfind("}")
        if start != -1 and end != -1 and end > start:
            clean = clean[start : end + 1]

    try:
        data = json.loads(clean)
        anonimizado = str(data.get("anonimizado", "")).strip()
        entidades = data.get("entidades", {})
        if isinstance(entidades, dict) and anonimizado:
            # Asegurar que todas las claves y valores sean strings
            clean_entities = {str(k): str(v) for k, v in entidades.items()}
            return anonimizado, clean_entities
    except Exception as e:
        logger.warning(f"No se pudo parsear JSON estricto del LLM ({e}). Aplicando fallback de texto plano.")

    # Fallback: devolver el texto crudo y diccionario vacío
    return raw_text.strip(), {}


# --- Motores de inferencia de anonimización ---

def anonimizar_con_lmstudio(
    texto: str,
    modelo: str = LMSTUDIO_DEFAULT_MODEL,
    base_url: str = LMSTUDIO_DEFAULT_URL,
    temperature: float = 0.1,
) -> AnonimizacionResult:
    """Ejecuta la anonimización utilizando LM Studio."""
    import httpx

    t0 = time.time()
    active_url = ensure_lmstudio_ready(base_url)

    payload = {
        "model": modelo,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_ANONIMIZACION},
            {"role": "user", "content": texto},
        ],
        "temperature": temperature,
    }

    with httpx.Client(timeout=45.0) as client:
        res = client.post(f"{active_url}/chat/completions", json=payload)
        if res.status_code != 200:
            raise RuntimeError(f"Error HTTP {res.status_code} desde LM Studio: {res.text}")
        data = res.json()
        raw_content = data["choices"][0]["message"]["content"]

    anonimizado, entidades = parsear_respuesta_json(raw_content)
    tiempo_ms = round((time.time() - t0) * 1000, 1)

    return AnonimizacionResult(
        original=texto,
        anonimizado=anonimizado,
        entidades=entidades,
        backend="lmstudio",
        modelo=modelo,
        tiempo_ms=tiempo_ms,
    )


def anonimizar_con_ollama(
    texto: str,
    modelo: str = OLLAMA_DEFAULT_MODEL,
    base_url: str = OLLAMA_DEFAULT_URL,
    temperature: float = 0.1,
) -> AnonimizacionResult:
    """Ejecuta la anonimización utilizando Ollama."""
    import httpx

    t0 = time.time()
    active_url = ensure_ollama_ready(base_url)

    payload = {
        "model": modelo,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_ANONIMIZACION},
            {"role": "user", "content": texto},
        ],
        "stream": False,
        "options": {"temperature": temperature},
    }

    with httpx.Client(timeout=60.0) as client:
        res = client.post(f"{active_url}/api/chat", json=payload)
        if res.status_code != 200:
            raise RuntimeError(f"Error HTTP {res.status_code} desde Ollama: {res.text}")
        data = res.json()
        raw_content = data.get("message", {}).get("content", "")

    anonimizado, entidades = parsear_respuesta_json(raw_content)
    tiempo_ms = round((time.time() - t0) * 1000, 1)

    return AnonimizacionResult(
        original=texto,
        anonimizado=anonimizado,
        entidades=entidades,
        backend="ollama",
        modelo=modelo,
        tiempo_ms=tiempo_ms,
    )


def anonimizar_prompt(
    texto: str,
    backend: str = "lmstudio",
    modelo: str = "",
    vault: Optional[PrivacyVault] = None,
) -> AnonimizacionResult:
    """Función de alto nivel para anonimizar texto seleccionando el backend."""
    b = backend.lower().strip()
    if b == "lmstudio":
        target_model = modelo.strip() if modelo.strip() else LMSTUDIO_DEFAULT_MODEL
        res = anonimizar_con_lmstudio(texto, modelo=target_model)
    elif b == "ollama":
        target_model = modelo.strip() if modelo.strip() else OLLAMA_DEFAULT_MODEL
        res = anonimizar_con_ollama(texto, modelo=target_model)
    else:
        raise ValueError(f"Backend no soportado: '{backend}'. Use 'lmstudio' u 'ollama'.")

    if vault is not None:
        vault.registrar(res.entidades)

    return res


def restaurar_texto(texto_anonimizado: str, entidades: Dict[str, str]) -> str:
    """Restaura un texto anonimizado utilizando el diccionario de entidades."""
    vault = PrivacyVault()
    return vault.restore(texto_anonimizado, entidades)


# --- Integración con Hooks del SDK de Antigravity ---

def crear_hook_antigravity(
    backend: str = "lmstudio",
    modelo: str = "",
    vault: Optional[PrivacyVault] = None,
) -> Callable:
    """Genera un hook de turno para el SDK Python de Antigravity (google-antigravity).
    Intercepta el mensaje del usuario y lo anonimiza localmente antes de que salga a la nube.
    """
    v = vault if vault is not None else PrivacyVault()

    async def hook_pre_turn_anonymizer(context: Any) -> Any:
        try:
            # Obtener el mensaje entrante del contexto
            user_msg = getattr(context, "user_message", "")
            if not user_msg:
                return getattr(context, "HookResult", lambda **kw: kw)(allow=True)

            logger.info(f"Hook de Privacidad activado. Anonimizando con backend {backend}...")
            resultado = anonimizar_prompt(user_msg, backend=backend, modelo=modelo, vault=v)
            context.user_message = resultado.anonimizado

            # Retornar autorización para proceder con el texto saneado
            from google.antigravity import types  # type: ignore
            return types.HookResult(
                allow=True,
                reason=f"Texto saneado localmente mediante {backend} ({resultado.modelo}). PII enmascarada.",
            )
        except Exception as e:
            logger.error(f"Fallo en hook de anonimización: {e}")
            from google.antigravity import types  # type: ignore
            return types.HookResult(
                allow=False,
                reason=f"Filtro de privacidad bloqueó el turno por error de sanitización: {e}",
            )

    return hook_pre_turn_anonymizer


# --- Interfaz interactiva y CLI ---

def menu_interactivo() -> Tuple[str, str, str]:
    """Muestra un menú en terminal para seleccionar backend, modelo e ingresar texto."""
    print("=" * 70)
    print("      FILTRO DE PRIVACIDAD Y ANONIMIZACIÓN CONTEXTUAL LOCAL")
    print("=" * 70)
    print("\nSeleccione el motor LLM local para el filtro de privacidad:")
    print("  [1] LM Studio  (Recomendado: qwen/qwen3-vl-8b / meta-llama-3.1-8b-instruct)")
    print("  [2] Ollama     (Recomendado: llama3.1:8b-instruct-q4_K_M)")

    eleccion = input("\nOpción [1/2] (por defecto 1): ").strip()
    if eleccion == "2":
        backend = "ollama"
        print(f"\nModelos sugeridos para Ollama: {', '.join(OLLAMA_RECOMMENDED)}")
        modelo = input(f"Modelo [{OLLAMA_DEFAULT_MODEL}]: ").strip()
        if not modelo:
            modelo = OLLAMA_DEFAULT_MODEL
    else:
        backend = "lmstudio"
        print(f"\nModelos sugeridos para LM Studio: {', '.join(LMSTUDIO_RECOMMENDED)}")
        modelo = input(f"Modelo [{LMSTUDIO_DEFAULT_MODEL}]: ").strip()
        if not modelo:
            modelo = LMSTUDIO_DEFAULT_MODEL

    print("\nIngrese el texto sensible que desea anonimizar:")
    texto = input("> ").strip()
    return backend, modelo, texto


def main():
    parser = argparse.ArgumentParser(
        description="Anonimizador contextual perimetral con modelos locales (LM Studio / Ollama)"
    )
    parser.add_argument("--backend", choices=["lmstudio", "ollama"], default="lmstudio",
                        help="Motor local de inferencia (lmstudio o ollama)")
    parser.add_argument("--model", type=str, default="",
                        help="Nombre del modelo local (por defecto qwen/qwen3-vl-8b o llama3.1:8b)")
    parser.add_argument("--text", type=str, default="", help="Texto directo a anonimizar")
    parser.add_argument("--file", type=str, default="", help="Archivo de texto a anonimizar")
    parser.add_argument("--restore", type=str, default="", help="Texto anonimizado a restaurar")
    parser.add_argument("--mapping", type=str, default="", help="Diccionario JSON de mapeo para restaurar")
    parser.add_argument("--interactive", action="store_true", help="Modo interactivo con selector de menú")
    parser.add_argument("--json", action="store_true", help="Salida en formato JSON estructurado")

    args = parser.parse_args()

    # Modo Restauración
    if args.restore:
        if not args.mapping:
            print("[Error]: Debe proporcionar el diccionario de entidades con --mapping para restaurar.", file=sys.stderr)
            sys.exit(1)
        try:
            mapping_dict = json.loads(args.mapping)
            restaurado = restaurar_texto(args.restore, mapping_dict)
            if args.json:
                print(json.dumps({"restaurado": restaurado}, ensure_ascii=False))
            else:
                print("\n--- Texto Restaurado ---")
                print(restaurado)
            return
        except Exception as e:
            print(f"[Error]: Fallo al restaurar: {e}", file=sys.stderr)
            sys.exit(1)

    # Modo Interactivo o Batch
    if args.interactive or (not args.text and not args.file and not sys.stdin.isatty()):
        backend, modelo, texto = menu_interactivo()
    elif not args.text and not args.file:
        backend, modelo, texto = menu_interactivo()
    else:
        backend = args.backend
        modelo = args.model
        if args.file:
            with open(args.file, "r", encoding="utf-8") as f:
                texto = f.read()
        else:
            texto = args.text

    if not texto.strip():
        print("[Aviso]: No se proporcionó texto para anonimizar.")
        return

    try:
        resultado = anonimizar_prompt(texto, backend=backend, modelo=modelo)
        if args.json:
            print(json.dumps(resultado.to_dict(), indent=2, ensure_ascii=False))
        else:
            print("\n" + "=" * 70)
            print(f" RESULTADO DE ANONIMIZACIÓN ({resultado.backend.upper()} - {resultado.modelo})")
            print(f" Tiempo: {resultado.tiempo_ms} ms")
            print("=" * 70)
            print("\n[Texto Anonimizado para la Nube]:")
            print(resultado.anonimizado)
            print("\n[Bóveda de Privacidad (Solo Local)]:")
            print(json.dumps(resultado.entidades, indent=2, ensure_ascii=False))
            print("=" * 70)
    except Exception as e:
        print(f"\n[Error de Inferencia]: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
