"""
# Tool: calculador_vram_llm.py

## Metadata
- Nombre: calculador_vram_llm
- Versión: 1.0
- Lenguaje: Python
- Creada en proyecto: H_LLM_IA_BBB
- Fecha de creación: 2026-09-27
- Descripción: Calcula la VRAM requerida para inferencia de LLMs locales en función del tamaño de parámetros, nivel de cuantización y ventana de contexto (KV Cache). Estima rendimiento teórico (tokens/segundo) según ancho de banda de memoria de la GPU/RAM y analiza viabilidad en Windows y Linux.
- Parámetros:
  - --params, -p: Parámetros del modelo en miles de millones (e.g. 7, 8, 14, 32, 70).
  - --quant, -q: Cuantización (FP16, Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_M, Q2_K, IQ4_XS).
  - --context, -c: Longitud de contexto en tokens (por defecto 4096).
  - --vram, -v: VRAM disponible de la GPU en GB (opcional).
  - --bandwidth, -b: Ancho de banda de memoria de la GPU en GB/s (opcional).
  - --ram-type: Tipo de RAM del sistema para offloading CPU (DDR4, DDR5, LPDDR5).
  - --json: Salida estructurada en JSON.
- Dependencias: sys, argparse, json, math (biblioteca estándar de Python).

## Código
"""

import sys
import argparse
import json
import math

BPW_TABLE = {
    "FP16": 16.0,
    "BF16": 16.0,
    "Q8_0": 8.5,
    "Q6_K": 6.56,
    "Q5_K_M": 5.5,
    "Q5_0": 5.5,
    "Q4_K_M": 4.5,
    "Q4_0": 4.5,
    "IQ4_XS": 4.25,
    "Q3_K_M": 3.44,
    "Q2_K": 2.63,
}

SYSTEM_OVERHEAD_VRAM = {
    "windows_cuda": 0.8,
    "windows_directml": 0.6,
    "linux_cuda": 0.5,
    "linux_rocm": 0.7,
    "linux_vulkan": 0.4
}

RAM_BANDWIDTH = {
    "DDR4_DUAL": 45.0,     # ~3200 MT/s Dual Channel
    "DDR5_DUAL": 85.0,     # ~5600-6000 MT/s Dual Channel
    "LPDDR5_QUAD": 120.0   # En laptops / mini PCs APU
}

def estimar_dimensiones(params_b: float):
    # Estimación de capas y hidden dim aproximadas según el tamaño estándar Llama/Mistral/Qwen
    if params_b <= 3.5:
        layers = 28
        hidden_dim = 3072
        kv_heads = 8
        head_dim = 128
    elif params_b <= 9.0:
        layers = 32
        hidden_dim = 4096
        kv_heads = 8
        head_dim = 128
    elif params_b <= 16.0:
        layers = 48
        hidden_dim = 5120
        kv_heads = 8
        head_dim = 128
    elif params_b <= 35.0:
        layers = 64
        hidden_dim = 5120
        kv_heads = 8
        head_dim = 128
    else:
        layers = 80
        hidden_dim = 8192
        kv_heads = 8
        head_dim = 128
    return layers, hidden_dim, kv_heads, head_dim

def calcular_vram_requerida(params_b: float, quant: str, context_tokens: int, kv_cache_bits: int = 16):
    quant_upper = quant.upper()
    bpw = BPW_TABLE.get(quant_upper, 4.5)
    
    # 1. Tamaño del modelo en GiB (pesos + tensores auxiliares ~4% de sobrecarga)
    peso_bytes = (params_b * 1e9 * bpw) / 8.0
    peso_gib = (peso_bytes / (1024**3)) * 1.04
    
    # 2. KV Cache (GQA asumido para modelos modernos 2024-2026)
    layers, hidden_dim, kv_heads, head_dim = estimar_dimensiones(params_b)
    # KV cache bytes = 2 (K y V) * layers * kv_heads * head_dim * context * (kv_cache_bits / 8)
    kv_cache_bytes = 2 * layers * kv_heads * head_dim * context_tokens * (kv_cache_bits / 8.0)
    kv_cache_gib = kv_cache_bytes / (1024**3)
    
    # 3. Buffer de computación y activación de contexto
    activacion_gib = max(0.25, (context_tokens * hidden_dim * 2 * 2) / (1024**3))
    
    # Margen base del motor de inferencia (llama.cpp / Ollama / vLLM)
    overhead_motor_gib = 0.4
    
    total_min_gib = peso_gib + kv_cache_gib + activacion_gib + overhead_motor_gib
    
    return {
        "params_b": params_b,
        "quant": quant_upper,
        "bpw": bpw,
        "context_tokens": context_tokens,
        "peso_modelo_gib": round(peso_gib, 2),
        "kv_cache_gib": round(kv_cache_gib, 2),
        "activaciones_gib": round(activacion_gib, 2),
        "vram_total_requerida_gib": round(total_min_gib, 2)
    }

def evaluar_hardware(calculo: dict, vram_gpu: float = None, bandwidth_gpu: float = None, ram_type: str = "DDR5_DUAL"):
    res = dict(calculo)
    if vram_gpu is not None:
        res["vram_gpu_disponible"] = vram_gpu
        cabe_100_vram = calculo["vram_total_requerida_gib"] <= (vram_gpu * 0.95)
        res["cabe_completamente_en_vram"] = cabe_100_vram
        
        if cabe_100_vram:
            res["modo_ejecucion"] = "100% GPU Offload (Óptimo, baja latencia)"
            res["offload_ram_gib"] = 0.0
            if bandwidth_gpu:
                # tokens/segundo generados ~ Ancho de banda de memoria / tamaño de pesos del modelo en GiB
                # En generación autorregresiva (batch 1), cada token lee todos los pesos
                tps_teorico = (bandwidth_gpu / calculo["peso_modelo_gib"]) * 0.75 # Factor de eficiencia ~75%
                res["tokens_por_segundo_teorico"] = round(tps_teorico, 1)
        else:
            deficit = calculo["vram_total_requerida_gib"] - (vram_gpu * 0.90)
            res["modo_ejecucion"] = "Offloading Mixto (GPU + CPU/RAM)"
            res["offload_ram_gib"] = round(deficit, 2)
            
            # Ancho de banda cuello de botella en offload mixto gobernado por el bus PCIe y RAM
            ram_bw = RAM_BANDWIDTH.get(ram_type, 60.0)
            bw_efectivo = min(ram_bw, 32.0) # Limitado por PCIe 4.0 x16 o DDR
            tps_teorico = (bw_efectivo / calculo["peso_modelo_gib"]) * 0.70
            res["tokens_por_segundo_teorico"] = round(tps_teorico, 1)
            res["advertencia"] = f"El modelo excede la VRAM por {round(deficit, 1)} GiB. La velocidad caerá drásticamente por cuello de botella en RAM ({ram_type})."
            
    return res

def main():
    parser = argparse.ArgumentParser(description="Calculador de VRAM y Rendimiento para LLMs Locales (H_LLM_IA_BBB)")
    parser.add_argument("--params", "-p", type=float, required=True, help="Número de parámetros en miles de millones (e.g. 7, 8, 14, 32)")
    parser.add_argument("--quant", "-q", type=str, default="Q4_K_M", help="Cuantización (FP16, Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_M, Q2_K)")
    parser.add_argument("--context", "-c", type=int, default=4096, help="Ventana de contexto en tokens (default: 4096)")
    parser.add_argument("--vram", "-v", type=float, default=None, help="VRAM disponible en tarjeta gráfica (GB)")
    parser.add_argument("--bandwidth", "-b", type=float, default=None, help="Ancho de banda GPU en GB/s (e.g. 288 para RTX 3060 12GB)")
    parser.add_argument("--ram-type", type=str, default="DDR5_DUAL", choices=["DDR4_DUAL", "DDR5_DUAL", "LPDDR5_QUAD"], help="Tipo de RAM del sistema")
    parser.add_argument("--json", action="store_true", help="Formato de salida en JSON")
    
    args = parser.parse_args()
    
    calculo = calcular_vram_requerida(args.params, args.quant, args.context)
    resultado = evaluar_hardware(calculo, args.vram, args.bandwidth, args.ram_type)
    
    if args.json:
        print(json.dumps(resultado, indent=2))
        return
        
    print("\n" + "="*60)
    print(f"  DIMENSIONAMIENTO DE HARDWARE LLM: {resultado['params_b']}B ({resultado['quant']})")
    print("="*60)
    print(f"- Peso del modelo:           {resultado['peso_modelo_gib']} GiB")
    print(f"- KV Cache ({resultado['context_tokens']} tokens):       {resultado['kv_cache_gib']} GiB")
    print(f"- Activación / Overhead:     {resultado['activaciones_gib']} GiB")
    print(f"- VRAM MÍNIMA REQUERIDA:     {resultado['vram_total_requerida_gib']} GiB")
    print("-" * 60)
    
    if args.vram is not None:
        print(f"- VRAM GPU asignada:         {resultado['vram_gpu_disponible']} GiB")
        print(f"- Modo de ejecución:         {resultado['modo_ejecucion']}")
        if "tokens_por_segundo_teorico" in resultado:
            print(f"- Velocidad estimada:        ~{resultado['tokens_por_segundo_teorico']} tokens/segundo")
        if "advertencia" in resultado:
            print(f"\n[!] AVISO: {resultado['advertencia']}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
