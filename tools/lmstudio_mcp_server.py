# Tool: lmstudio_mcp_server.py
#
# Metadata:
# - Nombre: lmstudio_mcp_server
# - Versión: 2.0 (Con Auto-Encendido y Auto-Descarga On-Demand)
# - Lenguaje: Python
# - Creada en proyecto: Transcriptor_Whisper_LLM
# - Fecha de creación: 2026-10-19
# - Descripción: Servidor MCP para LM Studio sobre transporte stdio, con detección de estado y levantamiento automático bajo demanda (auto-encendido) vía lms CLI.
# - Parámetros: Variables de entorno LMSTUDIO_HOST, LMSTUDIO_DEFAULT_MODEL, LMSTUDIO_TIMEOUT.
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

# Configuración de logging a stderr (stdio reservado exclusivamente para protocolo MCP)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr
)
logger = logging.getLogger("LMStudio-MCP")

app = MCPServer(
    name="lmstudio",
    title="LM Studio MCP Server (Auto-Start)",
    description="Servidor MCP para inferencia local, chat completions, embeddings y gestión con auto-encendido bajo demanda en LM Studio."
)

DEFAULT_TIMEOUT = float(os.environ.get("LMSTUDIO_TIMEOUT", "120.0"))
DEFAULT_MODEL = os.environ.get("LMSTUDIO_DEFAULT_MODEL", "qwen/qwen3-vl-8b")
LMS_EXECUTABLE = os.environ.get("LMS_PATH", r"C:\Users\Developer\.lmstudio\bin\lms.exe")


def get_base_url() -> str:
    """Resuelve la URL base de LM Studio respetando variables de entorno o valores locales."""
    env_url = os.environ.get("LMSTUDIO_HOST")
    if env_url:
        clean = env_url.rstrip("/")
        if not clean.endswith("/v1"):
            clean = f"{clean}/v1"
        return clean
    return "http://127.0.0.1:1234/v1"


async def ensure_lmstudio_ready() -> bool:
    """Verifica si LM Studio está respondiendo. Si está apagado, lo levanta automáticamente vía lms CLI."""
    base_url = get_base_url()
    try:
        async with httpx.AsyncClient(timeout=1.5) as client:
            res = await client.get(f"{base_url}/models")
            if res.status_code == 200:
                return True
    except Exception:
        pass

    logger.info("LM Studio no responde. Iniciando servidor automáticamente mediante 'lms server start'...")
    candidates = [
        LMS_EXECUTABLE,
        os.path.expanduser(r"~/.lmstudio/bin/lms.exe"),
        "lms"
    ]
    started = False
    for exe in candidates:
        try:
            subprocess.Popen(
                [exe, "server", "start", "--bind", "0.0.0.0", "--cors"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )
            started = True
            break
        except Exception:
            continue

    if not started:
        logger.warning("No se pudo localizar el ejecutable lms para inicio automático.")
        return False

    # Esperar hasta 10 segundos a que el endpoint responda
    for _ in range(10):
        await asyncio.sleep(1.0)
        try:
            async with httpx.AsyncClient(timeout=1.5) as client:
                res = await client.get(f"{base_url}/models")
                if res.status_code == 200:
                    logger.info("LM Studio levantado y listo para inferencia.")
                    return True
        except Exception:
            pass

    logger.error("Tiempo de espera agotado para el inicio de LM Studio.")
    return False


@app.tool()
async def lmstudio_health_check() -> str:
    """Verifica la conectividad y disponibilidad del servidor LM Studio.
    Si el servidor está apagado, intenta levantarlo automáticamente."""
    is_ready = await ensure_lmstudio_ready()
    base_url = get_base_url()
    
    if not is_ready:
        return json.dumps({
            "status": "offline",
            "base_url": base_url,
            "auto_start_attempted": True,
            "error": "El servidor no respondió tras la orden de auto-encendido.",
            "recommendation": "Verifique que LM Studio esté instalado y configurado correctamente en el sistema."
        }, indent=2, ensure_ascii=False)

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            t0 = asyncio.get_event_loop().time()
            res = await client.get(f"{base_url}/models")
            latency_ms = round((asyncio.get_event_loop().time() - t0) * 1000, 1)
            
            data = res.json()
            models = [m.get("id") for m in data.get("data", [])]
            return json.dumps({
                "status": "online",
                "base_url": base_url,
                "latency_ms": latency_ms,
                "models_count": len(models),
                "available_models": models,
                "default_model": DEFAULT_MODEL if DEFAULT_MODEL in models else (models[0] if models else "none"),
                "auto_start_supported": True
            }, indent=2, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)}, indent=2, ensure_ascii=False)


@app.tool()
async def lmstudio_list_models() -> str:
    """Obtiene la lista completa de modelos cargados y disponibles en el servidor LM Studio (con auto-encendido)."""
    await ensure_lmstudio_ready()
    base_url = get_base_url()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.get(f"{base_url}/models")
            if res.status_code == 200:
                data = res.json()
                models_list = [
                    {"id": m.get("id"), "object": m.get("object"), "owned_by": m.get("owned_by")}
                    for m in data.get("data", [])
                ]
                return json.dumps({"models": models_list, "total": len(models_list)}, indent=2, ensure_ascii=False)
            return json.dumps({"error": f"HTTP {res.status_code}: {res.text}"})
    except Exception as e:
        return json.dumps({"error": f"Fallo al conectar con {base_url}: {str(e)}"}, ensure_ascii=False)


@app.tool()
async def lmstudio_chat_completion(
    prompt: str,
    system_prompt: str = "Eres un asistente técnico de alto nivel. Responde de forma concisa, precisa y formal en español.",
    model: str = "",
    temperature: float = 0.2,
    max_tokens: int = 2048
) -> str:
    """Envía un prompt a un modelo LLM local en LM Studio y retorna su respuesta generada.
    Si el servidor está apagado, lo levanta automáticamente antes de procesar la consulta."""
    is_ready = await ensure_lmstudio_ready()
    if not is_ready:
        return "[Error]: LM Studio no está activo y falló el intento de auto-encendido."

    base_url = get_base_url()
    target_model = model.strip() if model.strip() else DEFAULT_MODEL
    
    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": prompt.strip()})
    
    payload = {
        "model": target_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            res = await client.post(f"{base_url}/chat/completions", json=payload)
            if res.status_code != 200:
                return f"[Error HTTP {res.status_code} desde LM Studio]: {res.text}"
            
            data = res.json()
            choices = data.get("choices", [])
            if not choices:
                return "[Error]: El modelo respondió sin contenido."
            
            content = choices[0].get("message", {}).get("content", "").strip()
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            comp_tokens = usage.get("completion_tokens", 0)
            logger.info(f"Completado con '{target_model}'. Tokens: {prompt_tokens} in / {comp_tokens} out.")
            return content
    except httpx.TimeoutException:
        return f"[Error de Timeout]: La inferencia excedió {DEFAULT_TIMEOUT}s."
    except Exception as e:
        return f"[Error de Conexión en {base_url}]: {str(e)}"


@app.tool()
async def lmstudio_summarize(
    text: str,
    style: str = "executive",
    model: str = "",
    max_tokens: int = 1500
) -> str:
    """Genera un resumen estructurado utilizando LM Studio (con auto-encendido previo)."""
    style_prompts = {
        "executive": (
            "Eres un analista ejecutivo senior. Genera un resumen ejecutivo formal del siguiente contenido. "
            "Estructura: 1. Propósito principal, 2. Puntos clave y decisiones, 3. Conclusiones y próximos pasos. "
            "Idioma: Español latino neutro sin clichés ni grandilocuencias."
        ),
        "technical": (
            "Eres un auditor técnico y arquitecto de sistemas. Sintetiza los aspectos técnicos, configuraciones, "
            "estándares y metodologías expuestas en el texto. Registro formal y sobrio."
        ),
        "action_items": (
            "Eres un relator de proyectos. Extrae de forma concisa las tareas, acuerdos, responsables y próximos "
            "compromisos mencionados en el texto."
        ),
        "bullet_points": (
            "Sintetiza las ideas principales del siguiente texto en una lista clara de puntos clave numerados."
        )
    }
    sys_instruction = style_prompts.get(style.lower(), style_prompts["executive"])
    return await lmstudio_chat_completion(
        prompt=text,
        system_prompt=sys_instruction,
        model=model,
        temperature=0.2,
        max_tokens=max_tokens
    )


@app.tool()
async def lmstudio_create_embedding(
    text: str,
    model: str = "text-embedding-nomic-embed-text-v1.5"
) -> str:
    """Genera un vector numérico denso de embedding utilizando un modelo en LM Studio (con auto-encendido)."""
    await ensure_lmstudio_ready()
    base_url = get_base_url()
    payload = {
        "model": model,
        "input": text
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            res = await client.post(f"{base_url}/embeddings", json=payload)
            if res.status_code != 200:
                return json.dumps({"error": f"HTTP {res.status_code}: {res.text}"})
            
            data = res.json()
            emb_list = data.get("data", [])
            if not emb_list:
                return json.dumps({"error": "No se generaron embeddings en la respuesta."})
            
            vector = emb_list[0].get("embedding", [])
            return json.dumps({
                "model": model,
                "dimension": len(vector),
                "embedding_sample": vector[:5],
                "vector_length": len(vector)
            }, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Fallo al generar embedding: {str(e)}"}, ensure_ascii=False)


@app.tool()
async def lmstudio_anonymize(
    text: str,
    model: str = "qwen/qwen3-vl-8b"
) -> str:
    """Anonimiza contextualmente un texto sensible reemplazando PII (nombres, IPs, empresas, correos) por marcadores semánticos y generando el diccionario para desanonimizar."""
    target_model = model.strip() if model.strip() else DEFAULT_MODEL
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
    raw = await lmstudio_chat_completion(
        prompt=text,
        system_prompt=sys_prompt,
        model=target_model,
        temperature=0.1
    )
    # Extraer JSON si viene envuelto en markdown
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
async def lmstudio_restore(
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
async def lmstudio_stop_server() -> str:
    """Detiene el servidor HTTP de LM Studio para liberar completamente la memoria GPU/VRAM y CPU."""
    candidates = [
        LMS_EXECUTABLE,
        os.path.expanduser(r"~/.lmstudio/bin/lms.exe"),
        "lms"
    ]
    for exe in candidates:
        try:
            res = subprocess.run([exe, "server", "stop"], capture_output=True, text=True)
            output = res.stdout.strip() or res.stderr.strip()
            logger.info("Servidor LM Studio detenido bajo petición.")
            return json.dumps({"status": "stopped", "detail": output}, ensure_ascii=False)
        except Exception:
            continue
    return json.dumps({"error": "No se pudo ejecutar el comando de detención."}, ensure_ascii=False)


def main():
    """Punto de entrada principal para el servidor MCP stdio."""
    logger.info("Iniciando LM Studio MCP Server (Auto-Start) sobre transporte stdio...")
    logger.info(f"URL de LM Studio: {get_base_url()} | Modelo predeterminado: {DEFAULT_MODEL}")
    app.run(transport="stdio")


if __name__ == "__main__":
    main()
