"""
Módulo de exportación multi-formato para Transcriptor_Whisper_LLM.
Soporta generación de archivos TXT, SRT, VTT, JSON, MD y DOCX.
Garantiza el confinamiento estricto de entregables en la carpeta /output.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

try:
    from docx import Document
    from docx.shared import Pt, Inches, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False


class ExportManager:
    """Gestiona la compilación y guardado de resultados en todos los formatos requeridos."""

    def __init__(self, output_dir: Optional[str] = None):
        """Inicializa el gestor fijando la carpeta de salida confinada."""
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            # Directorio 'output' relativo a la raíz del proyecto
            self.output_dir = Path(__file__).resolve().parent.parent / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _format_timestamp_srt(seconds: float) -> str:
        """Formatea segundos a marca temporal SRT: HH:MM:SS,mmm"""
        hrs = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        msecs = int(round((seconds - int(seconds)) * 1000))
        return f"{hrs:02d}:{mins:02d}:{secs:02d},{msecs:03d}"

    @staticmethod
    def _format_timestamp_vtt(seconds: float) -> str:
        """Formatea segundos a marca temporal WebVTT: HH:MM:SS.mmm"""
        hrs = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        msecs = int(round((seconds - int(seconds)) * 1000))
        return f"{hrs:02d}:{mins:02d}:{secs:02d}.{msecs:03d}"

    def export_txt(self, base_name: str, text: str, segments: Optional[List[Dict[str, Any]]] = None, include_timestamps: bool = False) -> str:
        """Exporta el texto a formato .txt simple o con marcas de tiempo."""
        out_path = self.output_dir / f"{base_name}.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            if include_timestamps and segments:
                for seg in segments:
                    start_str = self._format_timestamp_vtt(seg["start"])
                    end_str = self._format_timestamp_vtt(seg["end"])
                    f.write(f"[{start_str} -> {end_str}] {seg['text']}\n")
            else:
                f.write(text.strip() + "\n")
        return str(out_path)

    def export_srt(self, base_name: str, segments: List[Dict[str, Any]]) -> str:
        """Exporta los segmentos a formato de subtítulos estándar SubRip (.srt)."""
        out_path = self.output_dir / f"{base_name}.srt"
        with open(out_path, "w", encoding="utf-8") as f:
            for idx, seg in enumerate(segments, start=1):
                start = self._format_timestamp_srt(seg["start"])
                end = self._format_timestamp_srt(seg["end"])
                f.write(f"{idx}\n{start} --> {end}\n{seg['text']}\n\n")
        return str(out_path)

    def export_vtt(self, base_name: str, segments: List[Dict[str, Any]]) -> str:
        """Exporta los segmentos a formato de subtítulos para web (.vtt)."""
        out_path = self.output_dir / f"{base_name}.vtt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")
            for idx, seg in enumerate(segments, start=1):
                start = self._format_timestamp_vtt(seg["start"])
                end = self._format_timestamp_vtt(seg["end"])
                f.write(f"{idx}\n{start} --> {end}\n{seg['text']}\n\n")
        return str(out_path)

    def export_json(self, base_name: str, data: Dict[str, Any]) -> str:
        """Exporta el payload completo con metadatos, transcripción y LLM a formato .json."""
        out_path = self.output_dir / f"{base_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return str(out_path)

    def export_markdown(
        self,
        base_name: str,
        transcription_data: Dict[str, Any],
        summary: Optional[str] = None,
        meeting_minutes: Optional[str] = None,
        corrected_text: Optional[str] = None
    ) -> str:
        """Genera un informe estructurado en formato Markdown (.md)."""
        out_path = self.output_dir / f"{base_name}.md"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"# Transcripción y Análisis: {base_name}\n\n")
            
            # Tabla de metadatos técnicos
            f.write("## Metadatos de Procesamiento\n\n")
            f.write("| Parámetro | Valor |\n")
            f.write("| :--- | :--- |\n")
            f.write(f"| Idioma detectado | `{transcription_data.get('language', 'desconocido')}` ({round(transcription_data.get('language_probability', 0)*100, 1)}%) |\n")
            f.write(f"| Duración del audio | {transcription_data.get('duration', 'N/A')} s |\n")
            f.write(f"| Tiempo de transcripción | {transcription_data.get('transcription_time_seconds', 'N/A')} s |\n")
            f.write(f"| Modelo Faster-Whisper | `{transcription_data.get('model_used', 'N/A')}` |\n")
            f.write(f"| Dispositivo de cómputo | `{transcription_data.get('device_used', 'N/A')}` ({transcription_data.get('compute_type', 'N/A')}) |\n\n")

            if summary:
                f.write("## Síntesis Ejecutiva\n\n")
                f.write(summary.strip() + "\n\n")

            if meeting_minutes:
                f.write("## Minuta y Acuerdos de Sesión\n\n")
                f.write(meeting_minutes.strip() + "\n\n")

            if corrected_text:
                f.write("## Transcripción Fluida\n\n")
                f.write(corrected_text.strip() + "\n\n")
            else:
                f.write("## Transcripción Fluida\n\n")
                f.write(transcription_data.get("text", "").strip() + "\n\n")

            segments = transcription_data.get("segments", [])
            if segments:
                f.write("<details>\n<summary><strong>Desplegar registro técnico con marcas de tiempo</strong></summary>\n\n")
                for seg in segments:
                    t_start = self._format_timestamp_vtt(seg["start"])
                    t_end = self._format_timestamp_vtt(seg["end"])
                    f.write(f"- `[{t_start} - {t_end}]`: {seg['text']}\n")
                f.write("\n</details>\n\n")

        return str(out_path)

    def export_docx(
        self,
        base_name: str,
        transcription_data: Dict[str, Any],
        summary: Optional[str] = None,
        meeting_minutes: Optional[str] = None,
        corrected_text: Optional[str] = None
    ) -> str:
        """
        Genera un documento Word (.docx) cumpliendo con el estándar tipográfico institucional:
        - Títulos: Calibri 14pt negrita, negro puro (#000000).
        - Subtítulos: Calibri 12pt negrita, negro puro (#000000).
        - Párrafos: Calibri 12pt regular, interlineado 1.15, justificado, negro puro (#000000).
        """
        if not DOCX_AVAILABLE:
            raise ImportError("La biblioteca 'python-docx' no está disponible. Instálela con: pip install python-docx")

        out_path = self.output_dir / f"{base_name}.docx"
        doc = Document()

        # Configuración de márgenes estándar (1 pulgada / 2.54 cm)
        sections = doc.sections
        for section in sections:
            section.top_margin = Inches(1)
            section.bottom_margin = Inches(1)
            section.left_margin = Inches(1)
            section.right_margin = Inches(1)

        black = RGBColor(0, 0, 0)

        # Helper para aplicar estilo formal a párrafos
        def add_styled_paragraph(text: str, is_bold: bool = False, font_size: int = 12, is_heading: bool = False, heading_level: int = 1):
            if is_heading:
                p = doc.add_paragraph()
                p.paragraph_format.space_before = Pt(12)
                p.paragraph_format.space_after = Pt(6)
                p.paragraph_format.line_spacing = 1.15
                run = p.add_run(text)
                run.font.name = "Calibri"
                run.font.size = Pt(14 if heading_level <= 2 else 12)
                run.font.bold = True
                run.font.color.rgb = black
                return p
            else:
                p = doc.add_paragraph()
                p.paragraph_format.space_before = Pt(0)
                p.paragraph_format.space_after = Pt(6)
                p.paragraph_format.line_spacing = 1.15
                p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                run = p.add_run(text)
                run.font.name = "Calibri"
                run.font.size = Pt(font_size)
                run.font.bold = is_bold
                run.font.color.rgb = black
                return p

        # Título principal del documento
        add_styled_paragraph(f"INFORME DE TRANSCRIPCIÓN: {base_name.upper()}", is_heading=True, heading_level=1)

        # Resumen de metadatos técnicos
        add_styled_paragraph("Metadatos Técnicos", is_heading=True, heading_level=2)
        lang = transcription_data.get('language', 'N/A')
        prob = round(transcription_data.get('language_probability', 0) * 100, 1)
        dur = transcription_data.get('duration', 'N/A')
        t_time = transcription_data.get('transcription_time_seconds', 'N/A')
        model = transcription_data.get('model_used', 'N/A')
        device = transcription_data.get('device_used', 'N/A')
        c_type = transcription_data.get('compute_type', 'N/A')

        meta_text = (
            f"Idioma detectado: {lang} ({prob}% de certeza) | "
            f"Duración: {dur} segundos | Tiempo de proceso: {t_time} segundos | "
            f"Modelo: Faster-Whisper {model} | Dispositivo: {device} ({c_type})"
        )
        add_styled_paragraph(meta_text, is_bold=False, font_size=11)

        # Secciones condicionales de LLM
        if summary:
            add_styled_paragraph("Síntesis Ejecutiva", is_heading=True, heading_level=2)
            for line in summary.strip().split("\n"):
                if line.strip():
                    if line.startswith("#"):
                        clean_title = line.lstrip("#").strip()
                        add_styled_paragraph(clean_title, is_heading=True, heading_level=3)
                    else:
                        add_styled_paragraph(line.strip())

        if meeting_minutes:
            add_styled_paragraph("Minuta de Sesión y Acuerdos", is_heading=True, heading_level=2)
            for line in meeting_minutes.strip().split("\n"):
                if line.strip():
                    if line.startswith("#"):
                        clean_title = line.lstrip("#").strip()
                        add_styled_paragraph(clean_title, is_heading=True, heading_level=3)
                    else:
                        add_styled_paragraph(line.strip())

        if corrected_text:
            add_styled_paragraph("Transcripción Fluida y Puntuada", is_heading=True, heading_level=2)
            for p_text in corrected_text.strip().split("\n\n"):
                if p_text.strip():
                    add_styled_paragraph(p_text.strip())
        else:
            add_styled_paragraph("Transcripción Fluida", is_heading=True, heading_level=2)
            add_styled_paragraph(full_text.strip())

        doc.save(str(out_path))
        return str(out_path)

    def export_all(
        self,
        base_name: str,
        transcription_data: Dict[str, Any],
        summary: Optional[str] = None,
        meeting_minutes: Optional[str] = None,
        corrected_text: Optional[str] = None
    ) -> Dict[str, str]:
        """Exporta automáticamente todos los formatos disponibles en un solo llamado."""
        results: Dict[str, str] = {}
        segments = transcription_data.get("segments", [])
        full_text = transcription_data.get("text", "")

        # 1. TXT: Texto fluido sin marcas de tiempo
        text_for_txt = corrected_text if corrected_text else full_text
        results["txt"] = self.export_txt(base_name, text_for_txt, include_timestamps=False)
        # 2. SRT
        results["srt"] = self.export_srt(base_name, segments)
        # 3. VTT
        results["vtt"] = self.export_vtt(base_name, segments)
        # 4. JSON
        payload = {
            "metadata": {
                "base_name": base_name,
                "language": transcription_data.get("language"),
                "language_probability": transcription_data.get("language_probability"),
                "duration": transcription_data.get("duration"),
                "transcription_time_seconds": transcription_data.get("transcription_time_seconds"),
                "model_used": transcription_data.get("model_used"),
                "device_used": transcription_data.get("device_used"),
                "compute_type": transcription_data.get("compute_type"),
            },
            "full_text": full_text,
            "corrected_text": corrected_text,
            "summary": summary,
            "meeting_minutes": meeting_minutes,
            "segments": segments
        }
        results["json"] = self.export_json(base_name, payload)
        # 5. MD
        results["md"] = self.export_markdown(base_name, transcription_data, summary, meeting_minutes, corrected_text)
        # 6. DOCX
        if DOCX_AVAILABLE:
            results["docx"] = self.export_docx(base_name, transcription_data, summary, meeting_minutes, corrected_text)

        return results
