# Guía de Uso: Transcriptor_Whisper_LLM

Suite modular de transcripción de audio y video utilizando **Faster-Whisper** (CTranslate2) con post-procesamiento y análisis mediante modelos de lenguaje locales (**Ollama** y **LM Studio**).

---

## 1. Requisitos Previos

1. **Python 3.10+** instalado en el sistema.
2. **FFmpeg** instalado y accesible desde la consola (`ffmpeg -version`).
3. *(Opcional)* GPU Nvidia con soporte CUDA para aceleración por hardware.
4. *(Opcional)* **Ollama** (`ollama serve`) o **LM Studio** con servidor local activo (`http://localhost:1234`).

---

## 2. Dependencias

Si no se han instalado las dependencias del proyecto, ejecute en PowerShell:

```powershell
pip install -r requirements.txt
```

---

## 3. Uso desde la Línea de Comandos (CLI)

El script `cli.py` permite procesar un archivo individual o un directorio completo por lotes:

### Transcripción básica individual
```powershell
python cli.py -i "C:\ruta\al\video.mp4" -m base
```

### Transcripción acelerada por GPU Nvidia (CUDA)
```powershell
python cli.py -i "C:\ruta\al\video.mp4" -m medium -d cuda -c float16
```

### Transcripción + Resumen y Corrección con Ollama
```powershell
python cli.py -i "C:\ruta\al\video.mp4" -m small --llm-backend ollama --llm-model llama3 --summary --correct
```

### Transcripción + Minuta formal con LM Studio
```powershell
python cli.py -i "C:\ruta\al\video.mp4" -m medium --llm-backend lm_studio --minutes
```

### Procesamiento por lotes de una carpeta completa
```powershell
python cli.py -i "C:\mis_videos\" -m base -f all
```

### Parámetros Disponibles

| Parámetro | Descripción | Valores Disponibles |
| :--- | :--- | :--- |
| `-i, --input` | Archivo o carpeta de entrada | Ruta válida en disco |
| `-m, --model` | Tamaño de modelo Faster-Whisper | `tiny`, `base`, `small`, `medium`, `large-v3`, `large-v3-turbo` |
| `-d, --device` | Dispositivo de cómputo | `auto`, `cuda`, `cpu` |
| `-c, --compute-type` | Tipo de cuantización | `int8`, `float16`, `int8_float16`, `default` |
| `-l, --language` | Idioma forzado | Código ISO (ej. `es`, `en`, `fr`). Por defecto: auto-detección |
| `--llm-backend` | Servidor LLM local | `none`, `ollama`, `lm_studio`, `auto` |
| `--llm-model` | Modelo en Ollama o LM Studio | Nombre del modelo (ej. `llama3`, `mistral`, `qwen2.5`) |
| `--summary` | Generar resumen ejecutivo | Flag booleano |
| `--correct` | Corregir ortografía y puntuar | Flag booleano |
| `--minutes` | Generar acta/minuta de reunión | Flag booleano |
| `-f, --formats` | Formatos a exportar | `all`, o lista: `txt,srt,vtt,json,md,docx` |
| `-o, --output-dir` | Carpeta de salida | Por defecto: `Transcriptor_Whisper_LLM/output` |

---

## 4. Uso de la Interfaz Web (Streamlit)

Para iniciar la interfaz gráfica interactiva, ejecute:

```powershell
streamlit run app.py
```

Se abrirá automáticamente en su navegador web (`http://localhost:8501`).

### Características de la Interfaz Web:
* Carga de archivos por arrastrar y soltar (Drag and drop).
* Detección en tiempo real del estado de **Ollama** y **LM Studio**.
* Selector dinámico de modelos cargados en el servidor LLM.
* Visor de segmentos con marcas de tiempo en vivo.
* Pestañas independientes para texto original, texto corregido, resumen y minuta.
* Botones de descarga directa para los 6 formatos (`.txt`, `.srt`, `.vtt`, `.json`, `.md`, `.docx`).

---

## 5. Entregables Generados

Todos los archivos generados se almacenan estrictamente en la subcarpeta `output/`:
- **`[nombre].txt`**: Transcripción completa en texto plano con marcas de tiempo.
- **`[nombre].srt`**: Subtítulos formateados para reproductores de video (VLC, MPC).
- **`[nombre].vtt`**: Subtítulos WebVTT para navegadores y plataformas web.
- **`[nombre].json`**: Archivo de intercambio con todos los metadatos y palabras individuales.
- **`[nombre].md`**: Documento Markdown con tablas técnicas, resúmenes y transcripción.
- **`[nombre].docx`**: Documento Word bajo estándar institucional (fuente Calibri negro).
