#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Tool: orquestador_hibrido.py

## Metadata
# - Nombre: orquestador_hibrido.py
# - Versión: 2.0 (Harness² Recursive Support)
# - Lenguaje: Python
# - Creada en proyecto: Transcriptor_Whisper_LLM / Global
# - Fecha de creación: 2026-10-19
# - Descripción: Orquestador híbrido agéntico con framework Harness² (Recursive Agent Harnessing).
#                Gemini 3.8 Flash actúa como supervisor central y delega tareas intensivas en tokens
#                a LM Studio u Ollama locales mediante el protocolo Propose–Probe–Compose y auto-mejora continua.
# - Parámetros: --task [texto], --model-orchestrator [gemini-3.8-flash], --api-key [clave], --interactive, --harness2
# - Dependencias: openai, httpx

import argparse
import datetime
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("OrquestadorHibrido")

# Importar funciones de autoencendido desde las tools canónicas locales
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

try:
    from anonimizador_contextual import (
        ensure_lmstudio_ready,
        ensure_ollama_ready,
        anonimizar_prompt,
        restaurar_texto
    )
except ImportError:
    ensure_lmstudio_ready = None
    ensure_ollama_ready = None
    anonimizar_prompt = None
    restaurar_texto = None

import httpx
from openai import OpenAI

# Constantes de configuración
DEFAULT_ORCHESTRATOR_MODEL = "gemini-3.8-flash"
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

LMSTUDIO_HOST = os.environ.get("LMSTUDIO_HOST", "http://127.0.0.1:1234/v1")
LMSTUDIO_DEFAULT_MODEL = "qwen/qwen3-vl-8b"

OLLAMA_RAW_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:1143")
OLLAMA_HOST = OLLAMA_RAW_HOST.replace("0.0.0.0", "127.0.0.1").rstrip("/")
if not OLLAMA_HOST.startswith("http"):
    OLLAMA_HOST = f"http://{OLLAMA_HOST}"
OLLAMA_DEFAULT_MODEL = "llama3.1:8b-instruct-q4_K_M"

SYSTEM_PROMPT_ORCHESTRATOR = (
    "Eres el Arquitecto Orquestador y Supervisor de Calidad impulsado por Gemini bajo el framework Harness².\n"
    "Tu función es resolver problemas complejos optimizando radicalmente el consumo de tokens de cloud LLM, "
    "delegando tareas técnicas, exploración de variantes y auditorías preliminares a tus agentes locales "
    "(LM Studio y Ollama) que se ejecutan en la GPU privada del usuario.\n\n"
    "Flujo de trabajo obligatorio (Protocolo Harness²: Propose–Probe–Compose):\n"
    "1. Planificación: Descompón la tarea del usuario identificando requisitos y posibles bifurcaciones.\n"
    "2. Propose (Generación Local): Ante tareas complejas, código o redacción extensa, invoca 'proponer_variantes_local' "
    "para generar 2 o 3 enfoques técnicos alternativos en hardware local (0 tokens cloud).\n"
    "3. Probe (Sondeo y QA Local): Invoca 'sondear_y_verificar_local' para evaluar en seco cada variante generada, "
    "auditando consistencia sintáctica, posibles bugs, consumo de recursos y reglas anti-slop en hardware local.\n"
    "4. Compose (Composición y Síntesis Cloud): Con las evaluaciones locales concisas en mano, selecciona lo mejor de "
    "cada propuesta y compón con tu superior capacidad de razonamiento la solución final definitiva.\n"
    "5. Harness Recursion: Si identificas un patrón recurrente, corrección o gotcha, invoca 'retroalimentar_harness' "
    "para que el aprendizaje persista en la memoria del workspace para sesiones futuras.\n"
    "6. Delegación Puntual: Para subtareas directas puedes usar 'delegar_a_lmstudio' o 'delegar_a_ollama'.\n"
    "7. Dictamen Final: Entrega una respuesta consolidada, técnica y exhaustiva sin clichés ni términos inflados."
)

# Definición de herramientas (Tools / Function Calling) para el orquestador
TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "proponer_variantes_local",
            "description": "Protocolo Harness² [Propose]: Genera 2 o 3 propuestas o alternativas de implementación sobre una especificación usando modelos locales (LM Studio u Ollama) sin costo de tokens cloud.",
            "parameters": {
                "type": "object",
                "properties": {
                    "especificacion": {
                        "type": "string",
                        "description": "Especificación de la tarea técnica o problema que requiere generación de alternativas."
                    },
                    "n_variantes": {
                        "type": "integer",
                        "default": 2,
                        "description": "Número de variantes a generar (típicamente 2 o 3)."
                    },
                    "backend": {
                        "type": "string",
                        "enum": ["lmstudio", "ollama"],
                        "default": "lmstudio",
                        "description": "Motor local a utilizar para la generación."
                    },
                    "modelo": {
                        "type": "string",
                        "description": "Modelo local opcional (ej. 'qwen/qwen3-vl-8b', 'meta-llama-3.1-8b-instruct')."
                    }
                },
                "required": ["especificacion"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "sondear_y_verificar_local",
            "description": "Protocolo Harness² [Probe]: Audita y evalúa en seco una variante técnica o bloque de código (sintaxis, consistencia, seguridad, anti-slop) usando modelos locales.",
            "parameters": {
                "type": "object",
                "properties": {
                    "variante": {
                        "type": "string",
                        "description": "El contenido, código o texto de la variante técnica a evaluar."
                    },
                    "criterios": {
                        "type": "string",
                        "description": "Criterios específicos de auditoría (ej. 'verificar rendimiento, ausencia de bucles infinitos, normas APA 7')."
                    },
                    "backend": {
                        "type": "string",
                        "enum": ["lmstudio", "ollama"],
                        "default": "lmstudio",
                        "description": "Motor local para la evaluación."
                    },
                    "modelo": {
                        "type": "string",
                        "description": "Modelo local a emplear."
                    }
                },
                "required": ["variante"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "retroalimentar_harness",
            "description": "Protocolo Harness² [Recursion]: Persiste un nuevo patrón técnico, gotcha o lección aprendida en la memoria episódica global (memory/aprendizaje.md) para auto-mejora del harness.",
            "parameters": {
                "type": "object",
                "properties": {
                    "aprendizaje": {
                        "type": "string",
                        "description": "Descripción clara del problema detectado y la acción o patrón de solución adoptado."
                    },
                    "tipo": {
                        "type": "string",
                        "enum": ["mejora", "patrón", "error"],
                        "default": "patrón",
                        "description": "Tipo de aprendizaje detectado."
                    },
                    "proyecto": {
                        "type": "string",
                        "default": "Global",
                        "description": "Proyecto de origen o 'Global'."
                    }
                },
                "required": ["aprendizaje"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delegar_a_lmstudio",
            "description": "Delega una tarea especializada al agente local de LM Studio (en GPU local). Ideal para análisis técnico en español, visión o redacción detallada.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tarea": {
                        "type": "string",
                        "description": "La instrucción técnica o contenido que debe procesar el especialista."
                    },
                    "rol": {
                        "type": "string",
                        "description": "El rol asignado al especialista (ej. 'Experto en Seguridad Web', 'Revisor de Código Python')."
                    },
                    "modelo": {
                        "type": "string",
                        "description": "Modelo de LM Studio a utilizar (ej. 'qwen/qwen3-vl-8b', 'meta-llama-3.1-8b-instruct')."
                    }
                },
                "required": ["tarea"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delegar_a_ollama",
            "description": "Delega una tarea especializada al agente local de Ollama (en GPU/CPU local). Ideal para resúmenes rápidos, extracción de entidades o generación de scripts ligeros.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tarea": {
                        "type": "string",
                        "description": "La instrucción técnica que debe ejecutar el especialista."
                    },
                    "rol": {
                        "type": "string",
                        "description": "El rol asignado (ej. 'Extractor de Datos', 'Generador de Comandos Bash')."
                    },
                    "modelo": {
                        "type": "string",
                        "description": "Modelo de Ollama a utilizar (ej. 'llama3.1:8b-instruct-q4_K_M')."
                    }
                },
                "required": ["tarea"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "anonimizar_datos_sensibles",
            "description": "Ejecuta un filtro perimetral de privacidad en un texto sensible antes de procesarlo, sustituyendo nombres, IPs y empresas por marcadores.",
            "parameters": {
                "type": "object",
                "properties": {
                    "texto": {
                        "type": "string",
                        "description": "El texto que contiene información confidencial a sanitizar."
                    },
                    "backend": {
                        "type": "string",
                        "enum": ["lmstudio", "ollama"],
                        "description": "Motor local a utilizar para la anonimización."
                    }
                },
                "required": ["texto"]
            }
        }
    }
]


def ejecutar_tarea_lmstudio(tarea: str, rol: str = "", modelo: str = "") -> str:
    """Ejecuta una solicitud en el servidor local de LM Studio."""
    target_model = modelo.strip() if modelo.strip() else LMSTUDIO_DEFAULT_MODEL
    sys_prompt = rol.strip() if rol.strip() else "Eres un asistente técnico de alto nivel. Responde de forma precisa y formal."
    
    logger.info(f"Delegando a LM Studio [Modelo: {target_model}] [Rol: {sys_prompt[:40]}...]")
    if ensure_lmstudio_ready:
        try:
            active_url = ensure_lmstudio_ready()
        except Exception as e:
            logger.warning(f"No se pudo verificar autoencendido de LM Studio: {e}")
            active_url = LMSTUDIO_HOST
    else:
        active_url = LMSTUDIO_HOST

    payload = {
        "model": target_model,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": tarea}
        ],
        "temperature": 0.2
    }
    with httpx.Client(timeout=60.0) as client:
        res = client.post(f"{active_url}/chat/completions", json=payload)
        if res.status_code != 200:
            return f"[Error HTTP {res.status_code} desde LM Studio]: {res.text}"
        data = res.json()
        return data["choices"][0]["message"]["content"].strip()


def ejecutar_tarea_ollama(tarea: str, rol: str = "", modelo: str = "") -> str:
    """Ejecuta una solicitud en el servidor local de Ollama."""
    target_model = modelo.strip() if modelo.strip() else OLLAMA_DEFAULT_MODEL
    sys_prompt = rol.strip() if rol.strip() else "Eres un asistente técnico de alto nivel. Responde de forma precisa y formal."
    
    logger.info(f"Delegando a Ollama [Modelo: {target_model}] [Rol: {sys_prompt[:40]}...]")
    if ensure_ollama_ready:
        try:
            active_url = ensure_ollama_ready()
        except Exception as e:
            logger.warning(f"No se pudo verificar autoencendido de Ollama: {e}")
            active_url = OLLAMA_HOST
    else:
        active_url = OLLAMA_HOST

    payload = {
        "model": target_model,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": tarea}
        ],
        "stream": False,
        "options": {"temperature": 0.2}
    }
    with httpx.Client(timeout=60.0) as client:
        res = client.post(f"{active_url}/api/chat", json=payload)
        if res.status_code != 200:
            return f"[Error HTTP {res.status_code} desde Ollama]: {res.text}"
        data = res.json()
        return data.get("message", {}).get("content", "").strip()


def ejecutar_proponer_variantes_local(
    especificacion: str,
    n_variantes: int = 2,
    backend: str = "lmstudio",
    modelo: str = ""
) -> str:
    """Protocolo Harness² [Propose]: Genera alternativas de implementación usando LLM local sin gastar tokens cloud."""
    prompt_propose = (
        f"Genera exactamente {n_variantes} propuestas o variantes técnicas diferenciadas para resolver la siguiente especificación:\n\n"
        f"--- ESPECIFICACIÓN ---\n{especificacion}\n\n"
        "Estructura tu respuesta exactamente así para cada variante:\n"
        "### VARIANTE [A/B/C]: [Nombre de la estrategia]\n"
        "- Enfoque arquitectónico: [Descripción concisa]\n"
        "- Pros técnicos: [Puntos fuertes]\n"
        "- Contras / Riesgos: [Limitaciones o dependencias]\n"
        "- Implementación / Código base: [Bloque de código o pseudocódigo riguroso]\n"
    )
    rol = "Eres un Arquitecto de Software y Sistemas Senior. Especialista en proponer arquitecturas limpias, modulares y de alto rendimiento."
    if backend.lower() == "ollama":
        return ejecutar_tarea_ollama(tarea=prompt_propose, rol=rol, modelo=modelo)
    else:
        return ejecutar_tarea_lmstudio(tarea=prompt_propose, rol=rol, modelo=modelo)


def ejecutar_sondear_y_verificar_local(
    variante: str,
    criterios: str = "",
    backend: str = "lmstudio",
    modelo: str = ""
) -> str:
    """Protocolo Harness² [Probe]: Audita en seco una propuesta técnica usando LLM local sin gastar tokens cloud."""
    criterios_str = criterios.strip() if criterios.strip() else "Sintaxis, rendimiento, seguridad, modularidad, edge-cases y estilo sobrio sin clichés."
    prompt_probe = (
        "Realiza una auditoría pericial y sondeo en seco (dry-run probe) sobre la siguiente propuesta técnica o código:\n\n"
        f"--- PROPUESTA / CÓDIGO ---\n{variante}\n\n"
        f"--- CRITERIOS EXIGIDOS ---\n{criterios_str}\n\n"
        "Devuelve tu auditoría con el siguiente formato estructurado:\n"
        "1. Diagnóstico de Integridad: [Válido / Advertencia / Falla crítica]\n"
        "2. Análisis de Sintaxis y Bugs: [Detalle de posibles fallos de ejecución o excepciones no manejadas]\n"
        "3. Rendimiento y Seguridad: [Impacto en CPU/RAM, validación de inputs, control de errores]\n"
        "4. Filtro Anti-Slop / Calidad: [Verificación de tono técnico, ausencia de clichés LLM]\n"
        "5. Puntuación de Viabilidad (1 a 10): [Puntaje numérico y veredicto final para el orquestador]"
    )
    rol = "Eres un Auditor Pericial de Código y Calidad de Software (QA Gatekeeper). Eres implacable detectando inconsistencias y errores de ejecución."
    if backend.lower() == "ollama":
        return ejecutar_tarea_ollama(tarea=prompt_probe, rol=rol, modelo=modelo)
    else:
        return ejecutar_tarea_lmstudio(tarea=prompt_probe, rol=rol, modelo=modelo)


def ejecutar_retroalimentar_harness(
    aprendizaje: str,
    tipo: str = "patrón",
    proyecto: str = "Global"
) -> str:
    """Protocolo Harness² [Recursion]: Persiste un nuevo patrón o mejora en memory/aprendizaje.md."""
    try:
        workspace_root = os.path.abspath(os.path.join(_current_dir, "..", ".."))
        aprendizaje_file = os.path.join(workspace_root, "memory", "aprendizaje.md")
        if not os.path.exists(aprendizaje_file):
            return f"[Error]: No se encontró el archivo de memoria en '{aprendizaje_file}'."

        fecha_hoy = datetime.date.today().strftime("%Y-%m-%d")
        bloque = (
            f"\n\n---\n\n"
            f"## {fecha_hoy} — [{tipo}]\n"
            f"- Proyecto: {proyecto}\n"
            f"- Contexto: Optimización recursiva del Harness agéntico (Harness²).\n"
            f"- Problema detectado: {aprendizaje}\n"
            f"- Acción tomada: Se formalizó el patrón y se aplicó al orquestador híbrido con ahorro de tokens cloud.\n"
            f"- Aplicable a futuros proyectos: sí."
        )

        with open(aprendizaje_file, "a", encoding="utf-8") as f:
            f.write(bloque)

        logger.info(f"Aprendizaje Harness² registrado en '{aprendizaje_file}'.")
        return f"[Éxito]: Aprendizaje registrado en memory/aprendizaje.md bajo fecha {fecha_hoy} y tipo '{tipo}'."
    except Exception as e:
        logger.error(f"Fallo al registrar en aprendizaje.md: {e}")
        return f"[Error]: No se pudo escribir en la memoria: {e}"


def despachar_herramienta(nombre: str, argumentos: Dict[str, Any]) -> str:
    """Ejecuta la herramienta seleccionada por Gemini."""
    if nombre == "proponer_variantes_local":
        return ejecutar_proponer_variantes_local(
            especificacion=argumentos.get("especificacion", ""),
            n_variantes=argumentos.get("n_variantes", 2),
            backend=argumentos.get("backend", "lmstudio"),
            modelo=argumentos.get("modelo", "")
        )
    elif nombre == "sondear_y_verificar_local":
        return ejecutar_sondear_y_verificar_local(
            variante=argumentos.get("variante", ""),
            criterios=argumentos.get("criterios", ""),
            backend=argumentos.get("backend", "lmstudio"),
            modelo=argumentos.get("modelo", "")
        )
    elif nombre == "retroalimentar_harness":
        return ejecutar_retroalimentar_harness(
            aprendizaje=argumentos.get("aprendizaje", ""),
            tipo=argumentos.get("tipo", "patrón"),
            proyecto=argumentos.get("proyecto", "Global")
        )
    elif nombre == "delegar_a_lmstudio":
        return ejecutar_tarea_lmstudio(
            tarea=argumentos.get("tarea", ""),
            rol=argumentos.get("rol", ""),
            modelo=argumentos.get("modelo", "")
        )
    elif nombre == "delegar_a_ollama":
        return ejecutar_tarea_ollama(
            tarea=argumentos.get("tarea", ""),
            rol=argumentos.get("rol", ""),
            modelo=argumentos.get("modelo", "")
        )
    elif nombre == "anonimizar_datos_sensibles":
        if anonimizar_prompt:
            res = anonimizar_prompt(
                texto=argumentos.get("texto", ""),
                backend=argumentos.get("backend", "lmstudio")
            )
            return json.dumps(res.to_dict(), ensure_ascii=False)
        else:
            return "[Aviso]: Modulo anonimizador no disponible."
    else:
        return f"[Error]: Herramienta desconocida '{nombre}'."


def orquestar_flujo(
    tarea_usuario: str,
    api_key: str,
    modelo_orquestador: str = DEFAULT_ORCHESTRATOR_MODEL,
    max_turnos: int = 5
) -> str:
    """Bucle agéntico supervisado: Gemini 3.8 Flash coordina y audita a los agentes locales."""
    client = OpenAI(
        api_key=api_key,
        base_url=GEMINI_OPENAI_BASE_URL
    )

    mensajes: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT_ORCHESTRATOR},
        {"role": "user", "content": tarea_usuario}
    ]

    print(f"\n[Orquestador]: Iniciando coordinación con '{modelo_orquestador}'...")
    print(f"[Tarea]: {tarea_usuario}\n")

    for turno in range(max_turnos):
        logger.info(f"Turno de orquestación #{turno + 1}...")
        response = client.chat.completions.create(
            model=modelo_orquestador,
            messages=mensajes,
            tools=TOOLS_SPEC,
            tool_choice="auto",
            temperature=0.2
        )

        mensaje_respuesta = response.choices[0].message

        # Si el modelo decidió llamar a una o más herramientas
        if mensaje_respuesta.tool_calls:
            mensajes.append(mensaje_respuesta)

            for call in mensaje_respuesta.tool_calls:
                func_name = call.function.name
                func_args = json.loads(call.function.arguments)

                print(f"  └── ⚡ [Gemini delega a]: {func_name}")
                print(f"      Args: {json.dumps(func_args, ensure_ascii=False)}")

                # Ejecutar en hardware local
                t0 = time.time()
                resultado_local = despachar_herramienta(func_name, func_args)
                latencia = round(time.time() - t0, 2)

                print(f"      ✔ [Completado en {latencia}s] Caracteres: {len(resultado_local)}")

                # Devolver el resultado del especialista a Gemini para supervisión
                mensajes.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": resultado_local
                })
        else:
            # Gemini ha finalizado la supervisión y entrega la síntesis final
            contenido_final = mensaje_respuesta.content
            return contenido_final

    return "[Aviso]: Se alcanzó el número máximo de turnos de supervisión sin cierre formal."


def main():
    parser = argparse.ArgumentParser(
        description="Orquestador Híbrido Supervisado: Gemini 3.8 Flash + Agentes Locales (LM Studio / Ollama)"
    )
    parser.add_argument("--task", type=str, default="", help="Tarea compleja a resolver")
    parser.add_argument("--model", type=str, default=DEFAULT_ORCHESTRATOR_MODEL,
                        help=f"Modelo de Gemini a usar como orquestador (por defecto {DEFAULT_ORCHESTRATOR_MODEL})")
    parser.add_argument("--api-key", type=str, default="",
                        help="API Key de Google Gemini (o definir variable GEMINI_API_KEY)")
    parser.add_argument("--interactive", action="store_true", help="Modo interactivo en consola")

    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("\n[Aviso]: No se detectó GEMINI_API_KEY en variables de entorno.")
        api_key = input("Ingrese su Google Gemini API Key: ").strip()
        if not api_key:
            print("[Error]: Se requiere una API Key para utilizar Gemini 3.8 Flash.", file=sys.stderr)
            sys.exit(1)

    tarea = args.task
    if args.interactive or not tarea:
        print("=" * 70)
        print("      ORQUESTADOR HÍBRIDO SUPERVISADO (GEMINI 3.8 FLASH + LLM LOCAL)")
        print("=" * 70)
        print(f"Orquestador: {args.model}")
        print("Especialistas locales: LM Studio (127.0.0.1:1234) | Ollama (127.0.0.1:1143)")
        print("\nIngrese la tarea compleja que desea que Gemini orqueste y supervise:")
        tarea = input("> ").strip()

    if not tarea:
        print("[Aviso]: Tarea vacía. Saliendo.")
        return

    try:
        resultado = orquestar_flujo(
            tarea_usuario=tarea,
            api_key=api_key,
            modelo_orquestador=args.model
        )
        print("\n" + "=" * 70)
        print(" DICTAMEN FINAL SUPERVISADO POR GEMINI")
        print("=" * 70)
        print(resultado)
        print("=" * 70)
    except Exception as e:
        print(f"\n[Error de Orquestación]: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
