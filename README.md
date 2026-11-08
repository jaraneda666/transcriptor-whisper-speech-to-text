# Faster-Whisper Transcription & Minutes Pipeline

Pipeline de transcripción de audio de alto rendimiento basado en Faster-Whisper (CTranslate2), con extracción multimedia mediante FFmpeg, segmentación por marcas de tiempo y estructuración de minutas técnicas.

---

## Arquitectura y Componentes

- **Motor de voz:** Faster-Whisper con cuantización FP16/INT8 para GPU y CPU.
- **Preprocesamiento:** Extracción y conversión de audio a formato WAV 16 kHz mono con FFmpeg.
- **Interfaces:** Línea de comandos para procesamiento en lote (`cli.py`) y aplicación web interactiva (`app.py`).

---

## Instrucciones de Uso

```bash
# Transcripción de archivo de audio por consola
python cli.py --input grabacion.mp3 --model medium --output minuta.md
```

---

## Consideraciones de Seguridad

- Este repositorio almacena código fuente, utilitarios de prueba y documentación técnica de ingeniería.
- Todas las credenciales, llaves privadas y variables de entorno deben suministrarse de forma local mediante archivos no versionados.
- Las herramientas de auditoría contenidas deben utilizarse exclusivamente en entornos autorizados y laboratorios de prueba.

---

**Autor:** Juan Carlos Araneda M.  
**Licencia:** MIT License (consulte [LICENSE](LICENSE))
