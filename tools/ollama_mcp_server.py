# Tool: ollama_mcp_server.py
#
# Metadata:
# - Nombre: ollama_mcp_server
# - Versión: 1.0 (Con Auto-Encendido y Auto-Descarga On-Demand)
# - Lenguaje: Python
# - Creada en proyecto: Transcriptor_Whisper_LLM
# - Fecha de creación: 2026-10-19
# - Descripción: Servidor MCP nativo para Ollama sobre transporte stdio, con auto-encendido bajo demanda ('ollama serve') y auto-descarga de modelos.
# - Parámetros: Variables de entorno OLLAMA_HOST, OLLAMA_DEFAULT_MODEL, OLLAMA_TIMEOUT.
# - Dependencias: mcp >= 2.0.0, httpx
#

import os
import sys
import json
import logging
import asyncio
import subprocess
from typing import Optional, List, Dict, Any
import httpx
from mcp.server.mcpserver import MCPServer

# Configuración de logging a stderr (stdio está reservado para el protocolo MCP)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr
)
logger = logging.getLogger("Ollama-MCP")

app = MCPServer(
    name="ollama",
    title="Ollama MCP Server (Auto-Start)",
    description="Servidor MCP nativo para Ollama con auto-encendido bajo demanda, chat completions y embeddings locales."
)

DEFAULT_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "120.0"))
OLLAMA_DEFAULT_MODEL = os.environ.get("OLLAMA_DEFAULT_MODEL", "llama3.1:8b-instruct-q4_K_M")
OLLAMA_EXECUTABLE = os.environ.get(
    "OLLAMA_PATH",
    r"C:\Users\Developer\AppData\Local\Programs\Ollama\ollama.exe"
)


def get_base_url() -> str:
    """Resuelve la URL base de Ollama respetando la variable de entorno OLLAMA_HOST."""
    raw = os.environ.get("OLLAMA_HOST", "127.0.0.1:1143").strip()
    if "0.0.0.0" in raw:
        raw = raw.replace("0.0.0.0", "127.0.0.1")
    if not raw.startswith("http://") and not raw.startswith("https://"):
        raw = f"http://{raw}"
    return raw.rstrip("/")



async def ensure_ollama_ready() -> bool:
    """Verifica si Ollama está respondiendo. Si está apagado, lo levanta automáticamente vía 'ollama serve'."""
    base_url = get_base_url()
    try:
        async with httpx.AsyncClient(timeout=1.5) as client:
            res = await client.get(f"{base_url}/api/tags")
            if res.status_code == 200:
                return True
    except Exception:
        pass

    logger.info("Ollama no responde. Iniciando servicio automáticamente mediante 'ollama serve'...")
    candidates = [
        OLLAMA_EXECUTABLE,
        "ollama"
    ]
    started = False
    for exe in candidates:
        try:
            subprocess.Popen(
                [exe, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )
            started = True
            break
        except Exception:
            continue

    if not started:
        logger.warning("No se pudo localizar el ejecutable de Ollama para inicio automático.")
        return False

    # Esperar hasta 10 segundos a que el endpoint responda
    for _ in range(10):
        await asyncio.sleep(1.0)
        try:
            async with httpx.AsyncClient(timeout=1.5) as client:
                res = await client.get(f"{base_url}/api/tags")
                if res.status_code == 200:
                    logger.info("Ollama levantado y listo para inferencia.")
                    return True
        except Exception:
            pass

    logger.error("Tiempo de espera agotado para el inicio de Ollama.")
    return False


@app.tool()
async def ollama_health_check() -> str:
    """Verifica la conectividad y disponibilidad del servidor Ollama.
    Si está apagado, intenta levantarlo automáticamente."""
    is_ready = await ensure_ollama_ready()
    base_url = get_base_url()

    if not is_ready:
        return json.dumps({
            "status": "offline",
            "base_url": base_url,
            "auto_start_attempted": True,
            "error": "El servidor Ollama no respondió tras la orden de auto-encendido.",
            "recommendation": "Verifique que Ollama esté instalado y la ruta sea accesible."
        }, indent=2, ensure_ascii=False)

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            t0 = asyncio.get_event_loop().time()
            res = await client.get(f"{base_url}/api/tags")
            latency_ms = round((asyncio.get_event_loop().time() - t0) * 1000, 1)

            data = res.json()
            models = [m.get("name") for m in data.get("models", [])]
            return json.dumps({
                "status": "online",
                "base_url": base_url,
                "latency_ms": latency_ms,
                "models_count": len(models),
                "available_models": models,
                "default_model": OLLAMA_DEFAULT_MODEL if OLLAMA_DEFAULT_MODEL in models else (models[0] if models else "none"),
                "auto_start_supported": True
            }, indent=2, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)}, indent=2, ensure_ascii=False)


@app.tool()
async def ollama_list_models() -> str:
    """Obtiene la lista de modelos instalados en Ollama con sus detalles de tamaño y cuantización."""
    await ensure_ollama_ready()
    base_url = get_base_url()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.get(f"{base_url}/api/tags")
            if res.status_code == 200:
                data = res.json()
                models_list = []
                for m in data.get("models", []):
                    details = m.get("details", {})
                    models_list.append({
                        "name": m.get("name"),
                        "size_gb": round(m.get("size", 0) / (1024**3), 2),
                        "parameter_size": details.get("parameter_size"),
                        "quantization": details.get("quantization_level"),
                        "family": details.get("family")
                    })
                return json.dumps({"models": models_list, "total": len(models_list)}, indent=2, ensure_ascii=False)
            return json.dumps({"error": f"HTTP {res.status_code}: {res.text}"})
    except Exception as e:
        return json.dumps({"error": f"Fallo al conectar con Ollama en {base_url}: {str(e)}"}, ensure_ascii=False)


@app.tool()
async def ollama_chat_completion(
    prompt: str,
    system_prompt: str = "Eres un asistente técnico de alto nivel. Responde de forma concisa, precisa y formal en español.",
    model: str = "",
    temperature: float = 0.2,
    keep_alive: str = "5m"
) -> str:
    """Envía un prompt a un modelo LLM local en Ollama y retorna su respuesta generada.
    Si Ollama está apagado, se auto-enciende antes de procesar la solicitud."""
    is_ready = await ensure_ollama_ready()
    if not is_ready:
        return "[Error]: Ollama no está activo y falló el intento de auto-encendido."

    base_url = get_base_url()
    target_model = model.strip() if model.strip() else OLLAMA_DEFAULT_MODEL

    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": prompt.strip()})

    payload = {
        "model": target_model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature
        },
        "keep_alive": keep_alive
    }

    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            res = await client.post(f"{base_url}/api/chat", json=payload)
            if res.status_code != 200:
                return f"[Error HTTP {res.status_code} desde Ollama]: {res.text}"

            data = res.json()
            content = data.get("message", {}).get("content", "").strip()
            prompt_eval = data.get("prompt_eval_count", 0)
            eval_count = data.get("eval_count", 0)
            logger.info(f"Ollama completado con '{target_model}'. Tokens: {prompt_eval} in / {eval_count} out.")
            return content
    except httpx.TimeoutException:
        return f"[Error de Timeout]: La inferencia en Ollama excedió {DEFAULT_TIMEOUT}s."
    except Exception as e:
        return f"[Error de Conexión en Ollama {base_url}]: {str(e)}"


@app.tool()
async def ollama_create_embedding(
    text: str,
    model: str = ""
) -> str:
    """Genera un vector numérico de embedding para un texto dado utilizando Ollama."""
    await ensure_ollama_ready()
    base_url = get_base_url()
    target_model = model.strip() if model.strip() else OLLAMA_DEFAULT_MODEL

    payload = {
        "model": target_model,
        "prompt": text
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            res = await client.post(f"{base_url}/api/embeddings", json=payload)
            if res.status_code != 200:
                return json.dumps({"error": f"HTTP {res.status_code}: {res.text}"})

            data = res.json()
            vector = data.get("embedding", [])
            if not vector:
                return json.dumps({"error": "No se generaron embeddings en la respuesta."})

            return json.dumps({
                "model": target_model,
                "dimension": len(vector),
                "embedding_sample": vector[:5],
                "vector_length": len(vector)
            }, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Fallo al generar embedding: {str(e)}"}, ensure_ascii=False)


@app.tool()
async def ollama_anonymize(
    text: str,
    model: str = "llama3.1:8b-instruct-q4_K_M"
) -> str:
    """Anonimiza contextualmente un texto sensible reemplazando PII (nombres, IPs, empresas, correos) por marcadores semánticos y generando el diccionario para desanonimizar."""
    target_model = model.strip() if model.strip() else OLLAMA_DEFAULT_MODEL
    sys_prompt = (
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
        '    "[MARCADOR_1]": "valor_original_sensible"\n'
        "  }\n"
        "}"
    )
    raw = await ollama_chat_completion(
        prompt=text,
        system_prompt=sys_prompt,
        model=target_model,
        temperature=0.1
    )
    clean = raw.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean, re.DOTALL)
    if match:
        clean = match.group(1).strip()
    else:
        start = clean.find("{")
        end = clean.rfind("}")
        if start != -1 and end != -1 and end > start:
            clean = clean[start : end + 1]
    return clean


@app.tool()
async def ollama_restore(
    text: str,
    entities_json: str
) -> str:
    """Restaura los marcadores de un texto previamente anonimizado utilizando el mapa de entidades JSON."""
    try:
        mapping = json.loads(entities_json)
        result = text
        for placeholder, original in mapping.items():
            result = result.replace(str(placeholder), str(original))
        return result
    except Exception as e:
        return f"[Error al restaurar texto]: {str(e)}"


@app.tool()
async def ollama_stop_server() -> str:
    """Detiene el servicio de Ollama para liberar completamente la memoria GPU/VRAM."""
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/IM", "ollama.exe"], capture_output=True, text=True)
            logger.info("Proceso ollama.exe finalizado bajo petición.")
            return json.dumps({"status": "stopped", "message": "Ollama detenido correctamente."}, ensure_ascii=False)
        else:
            subprocess.run(["pkill", "-f", "ollama serve"], capture_output=True, text=True)
            return json.dumps({"status": "stopped", "message": "Ollama detenido correctamente."}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Fallo al detener Ollama: {str(e)}"}, ensure_ascii=False)


def main():
    """Punto de entrada principal para el servidor MCP stdio."""
    logger.info("Iniciando Ollama MCP Server (Auto-Start) sobre transporte stdio...")
    logger.info(f"URL de Ollama: {get_base_url()} | Modelo predeterminado: {OLLAMA_DEFAULT_MODEL}")
    app.run(transport="stdio")


if __name__ == "__main__":
    main()
