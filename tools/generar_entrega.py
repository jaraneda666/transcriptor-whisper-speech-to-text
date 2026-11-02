#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generar_entrega.py — Automatización Pericial de Inyección Word (.docx) (Antigravity)
Toma el borrador validado 'informe.md', actualiza la portada de 'plantilla_estudiante_aula40.docx'
y maqueta el contenido técnico estructurado con estilos institucionales.
DIRECTRIZ PERMANENTE: Entrega exclusiva en formato Word (.docx). Jamás genera archivos PDF.
"""

import sys
import os
import re
import shutil
import zipfile
import argparse
from datetime import datetime
import docx
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

# Colores institucionales: Todo el texto en negro estricto (#000000), excepto el título de la evaluación (portada) en blanco (#FFFFFF)
COLOR_TEXTO = RGBColor(0, 0, 0)               # #000000 - Texto universal en negro estricto
COLOR_TITULO_EVAL = RGBColor(255, 255, 255)   # #FFFFFF - Título de la evaluación en portada institucional
COLOR_NAVY = COLOR_TEXTO                      # Títulos en negro
COLOR_DARK = COLOR_TEXTO                      # Subtítulos y párrafos en negro
COLOR_GRAY = COLOR_TEXTO                      # Metadatos en negro
HEX_BOX_BG = "F8F9FA"                         # Fondo cajas de código/comandos
HEX_BOX_BORDER = "333333"                     # Borde cajas en gris oscuro/negro


def respaldar_plantilla(ruta_plantilla: str):
    """Crea una copia de seguridad .bak de la plantilla si no existe."""
    ruta_bak = ruta_plantilla + ".bak"
    if not os.path.exists(ruta_bak) and os.path.exists(ruta_plantilla):
        shutil.copy2(ruta_plantilla, ruta_bak)
        print(f"[OK] Respaldo de plantilla creado: {ruta_bak}")


def actualizar_portada_xml(ruta_origen: str, ruta_destino: str, datos: dict):
    """
    Actualiza directamente los campos de texto de la portada en word/document.xml
    dentro del archivo .docx para garantizar compatibilidad exacta con los textboxes OpenXML.
    """
    shutil.copy2(ruta_origen, ruta_destino)
    
    with zipfile.ZipFile(ruta_origen, "r") as zin:
        xml_content = zin.read("word/document.xml").decode("utf-8")
        
        # 1. Asignatura
        xml_content = re.sub(
            r"<w:t[^>]*>Nombre asignatura</w:t>",
            f"<w:t>{datos['asignatura'].upper()}</w:t>",
            xml_content
        )
        
        # 2. Semana
        xml_content = re.sub(
            r"<w:t[^>]*>Semana X</w:t>",
            f"<w:t>{datos['semana']}</w:t>",
            xml_content
        )
        
        # 3. Estudiante
        xml_content = re.sub(
            r"(<w:t[^>]*>Nombre del estudiante</w:t>.*?</w:r>.*?<w:r[^>]*>.*?<w:t[^>]*>):(</w:t>)",
            rf"\g<1>: {datos['estudiante']}\g<2>",
            xml_content,
            flags=re.DOTALL
        )
        # Reemplazo alternativo si ya estaba poblado
        xml_content = re.sub(
            r"<w:t[^>]*>Nombre del estudiante:\s*[^<]*</w:t>",
            f"<w:t>Nombre del estudiante: {datos['estudiante']}</w:t>",
            xml_content
        )

        # 4. Fecha de entrega
        xml_content = re.sub(
            r"(<w:t[^>]*>Fecha de entrega</w:t>.*?</w:r>.*?<w:r[^>]*>.*?<w:t[^>]*>):(</w:t>)",
            rf"\g<1>: {datos['fecha']}\g<2>",
            xml_content,
            flags=re.DOTALL
        )
        xml_content = re.sub(
            r"<w:t[^>]*>Fecha de entrega:\s*[^<]*</w:t>",
            f"<w:t>Fecha de entrega: {datos['fecha']}</w:t>",
            xml_content
        )

        # 5. Carrera
        xml_content = re.sub(
            r"(<w:t[^>]*>Carrera</w:t>.*?</w:r>.*?<w:r[^>]*>.*?<w:t[^>]*>):(</w:t>)",
            rf"\g<1>: {datos['carrera']}\g<2>",
            xml_content,
            flags=re.DOTALL
        )
        xml_content = re.sub(
            r"<w:t[^>]*>Carrera:\s*[^<]*</w:t>",
            f"<w:t>Carrera: {datos['carrera']}</w:t>",
            xml_content
        )

        # Re-empaquetar zip
        with zipfile.ZipFile(ruta_destino, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                if item.filename == "word/document.xml":
                    zout.writestr(item, xml_content.encode("utf-8"))
                else:
                    zout.writestr(item, zin.read(item.filename))


def aplicar_formato_run(run, bold=False, italic=False, font_name="Calibri", font_size=11, color=COLOR_DARK):
    run.font.name = font_name
    run.font.size = Pt(font_size)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = color


def agregar_caja_codigo(doc, texto_codigo: str):
    """Agrega una caja estilizada para comandos, scripts o salidas de terminal."""
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    
    # Sombreado de fondo y borde izquierdo
    shading_xml = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{HEX_BOX_BG}"/>')
    cell._tc.get_or_add_tcPr().append(shading_xml)
    
    borders_xml = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'  <w:left w:val="single" w:sz="18" w:space="0" w:color="{HEX_BOX_BORDER}"/>'
        f'  <w:top w:val="single" w:sz="4" w:space="0" w:color="D5D8DC"/>'
        f'  <w:right w:val="single" w:sz="4" w:space="0" w:color="D5D8DC"/>'
        f'  <w:bottom w:val="single" w:sz="4" w:space="0" w:color="D5D8DC"/>'
        f'</w:tcBorders>'
    )
    cell._tc.get_or_add_tcPr().append(borders_xml)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.15
    
    for linea in texto_codigo.strip().split("\n"):
        p_line = cell.add_paragraph() if p.text else p
        p_line.paragraph_format.space_before = Pt(0)
        p_line.paragraph_format.space_after = Pt(2)
        run = p_line.add_run(linea)
        aplicar_formato_run(run, font_name="Consolas", font_size=9, color=COLOR_DARK)
        p = p_line

    p_spacer = doc.add_paragraph()
    p_spacer.paragraph_format.space_before = Pt(0)
    p_spacer.paragraph_format.space_after = Pt(4)


def parsear_texto_con_formato(paragraph, texto: str, font_name="Calibri", font_size=12, base_color=COLOR_DARK):
    """Parsea markdown simple (negritas **texto**, cursivas *texto*, código `texto`) dentro de un párrafo."""
    patron = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)")
    partes = patron.split(texto)
    
    for parte in partes:
        if not parte:
            continue
        if parte.startswith("**") and parte.endswith("**"):
            run = paragraph.add_run(parte[2:-2])
            aplicar_formato_run(run, bold=True, font_name=font_name, font_size=font_size, color=base_color)
        elif parte.startswith("*") and parte.endswith("*"):
            run = paragraph.add_run(parte[1:-1])
            aplicar_formato_run(run, italic=True, font_name=font_name, font_size=font_size, color=base_color)
        elif parte.startswith("`") and parte.endswith("`"):
            run = paragraph.add_run(parte[1:-1])
            aplicar_formato_run(run, font_name="Consolas", font_size=9.5, color=COLOR_NAVY)
        else:
            run = paragraph.add_run(parte)
            aplicar_formato_run(run, font_name=font_name, font_size=font_size, color=base_color)


def procesar_markdown_a_docx(ruta_md: str, docx_path: str):
    """Lee el archivo markdown e inyecta el contenido formateado en el cuerpo del documento."""
    doc = docx.Document(docx_path)
    
    # Directiva obligatoria del estudiante:
    # Eliminar completamente todo el contenido preexistente en la plantilla institucional
    # desde 'DESARROLLO' hasta el final de la bibliografía ('BIBLIOGRAFIA' y ejemplos APA).
    idx_inicio = -1
    for i, p in enumerate(doc.paragraphs):
        if "DESARROLLO" in p.text.upper():
            idx_inicio = i
            break
            
    if idx_inicio != -1:
        parrafos_a_eliminar = doc.paragraphs[idx_inicio:]
        for p in parrafos_a_eliminar:
            p_elem = p._element
            p_elem.getparent().remove(p_elem)
        print(f"[OK] Depuración completa: Purgados {len(parrafos_a_eliminar)} párrafos preexistentes de la plantilla (desde DESARROLLO hasta BIBLIOGRAFÍA).")

    with open(ruta_md, "r", encoding="utf-8") as f:
        lineas = f.readlines()

    en_bloque_codigo = False
    buffer_codigo = []
    
    en_tabla = False
    buffer_tabla = []
    en_seccion_referencias = False

    for linea in lineas:
        linea_strip = linea.strip()

        # Omitir metadatos de portada redundantes si existieran al inicio del archivo
        if re.match(r"^#\s+INFORME\b", linea_strip, re.IGNORECASE):
            continue
        if re.match(r"^\*\*(?:Caso de Estudio|Asignatura|Semana|Estudiante|Institución|Fecha):\*\*", linea_strip, re.IGNORECASE):
            continue

        # Detección de bloques de código (```)
        if linea_strip.startswith("```"):
            if en_bloque_codigo:
                # Cierre de bloque de código
                agregar_caja_codigo(doc, "\n".join(buffer_codigo))
                buffer_codigo = []
                en_bloque_codigo = False
            else:
                en_bloque_codigo = True
            continue

        if en_bloque_codigo:
            buffer_codigo.append(linea.rstrip("\r\n"))
            continue

        # Detección de tablas markdown (| col | col |)
        if linea_strip.startswith("|") and linea_strip.endswith("|"):
            if not en_tabla:
                en_tabla = True
                buffer_tabla = []
            # Omitir fila separadora |---|---|
            if re.match(r"^\|(\s*:?-+:?\s*\|)+$", linea_strip):
                continue
            buffer_tabla.append([c.strip() for c in linea_strip.strip("|").split("|")])
            continue
        else:
            if en_tabla:
                # Inyectar tabla acumulada
                if buffer_tabla:
                    num_rows = len(buffer_tabla)
                    num_cols = max(len(r) for r in buffer_tabla)
                    tbl = doc.add_table(rows=num_rows, cols=num_cols)
                    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
                    for r_idx, fila in enumerate(buffer_tabla):
                        for c_idx, val in enumerate(fila):
                            if c_idx < num_cols:
                                cell = tbl.cell(r_idx, c_idx)
                                p = cell.paragraphs[0]
                                p.paragraph_format.space_before = Pt(3)
                                p.paragraph_format.space_after = Pt(3)
                                if r_idx == 0:
                                    parsear_texto_con_formato(p, val, font_size=10, base_color=COLOR_NAVY)
                                    p.runs[0].bold = True
                                    shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="EBF2F7"/>')
                                    cell._tc.get_or_add_tcPr().append(shading)
                                else:
                                    parsear_texto_con_formato(p, val, font_size=9.5, base_color=COLOR_DARK)
                    doc.add_paragraph()
                en_tabla = False
                buffer_tabla = []

        # Marcadores de evidencias visuales [INSERTAR CAPTURA DE PANTALLA: ...]
        if "[INSERTAR CAPTURA DE PANTALLA:" in linea_strip.upper():
            tbl = doc.add_table(rows=1, cols=1)
            tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
            cell = tbl.cell(0, 0)
            shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="FFF9E6"/>') # Amarillo pastel suave
            cell._tc.get_or_add_tcPr().append(shading)
            borders = parse_xml(
                f'<w:tcBorders {nsdecls("w")}>'
                f'  <w:left w:val="single" w:sz="18" w:space="0" w:color="D4AC0D"/>'
                f'  <w:top w:val="single" w:sz="4" w:space="0" w:color="F1C40F"/>'
                f'  <w:right w:val="single" w:sz="4" w:space="0" w:color="F1C40F"/>'
                f'  <w:bottom w:val="single" w:sz="4" w:space="0" w:color="F1C40F"/>'
                f'</w:tcBorders>'
            )
            cell._tc.get_or_add_tcPr().append(borders)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(6)
            run = p.add_run(f"📷 {linea_strip}")
            aplicar_formato_run(run, bold=True, font_name="Calibri", font_size=10, color=COLOR_TEXTO)
            doc.add_paragraph()
            continue

        # Encabezados Markdown
        if linea_strip.startswith("# "):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(14)
            p.paragraph_format.space_after = Pt(6)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(linea_strip[2:])
            aplicar_formato_run(run, bold=True, font_name="Calibri", font_size=14, color=COLOR_NAVY)
            en_seccion_referencias = False
            continue
        elif linea_strip.startswith("## "):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(12)
            p.paragraph_format.space_after = Pt(5)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(linea_strip[3:])
            aplicar_formato_run(run, bold=True, font_name="Calibri", font_size=14, color=COLOR_NAVY)
            if re.search(r"referencias|bibliograf[ií]a", linea_strip, re.IGNORECASE):
                en_seccion_referencias = True
            else:
                en_seccion_referencias = False
            continue
        elif linea_strip.startswith("### "):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(10)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(linea_strip[4:])
            aplicar_formato_run(run, bold=True, font_name="Calibri", font_size=12, color=COLOR_DARK)
            en_seccion_referencias = False
            continue
        elif linea_strip.startswith("#### "):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(linea_strip[5:])
            aplicar_formato_run(run, bold=True, font_name="Calibri", font_size=12, color=COLOR_DARK)
            en_seccion_referencias = False
            continue

        # Citas o Guías de captura Markdown (> texto)
        if linea_strip.startswith(">"):
            contenido_cita = linea_strip.lstrip("> ").strip()
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.3)
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing = 1.15
            parsear_texto_con_formato(p, contenido_cita, font_name="Calibri", font_size=10, base_color=COLOR_TEXTO)
            continue

        # Líneas horizontales (---)
        if linea_strip == "---":
            continue

        # Referencias bibliográficas (Norma APA 7: Sangría francesa, Calibri tamaño 12)
        if en_seccion_referencias and linea_strip:
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.5)
            p.paragraph_format.first_line_indent = Inches(-0.5)
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(6)
            p.paragraph_format.line_spacing = 1.15
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            texto_ref = re.sub(r"^[\*\-]\s+", "", linea_strip)
            parsear_texto_con_formato(p, texto_ref, font_name="Calibri", font_size=12, base_color=COLOR_DARK)
            continue

        # Listas con viñetas (- o *)
        if re.match(r"^[\*\-]\s+(.+)$", linea_strip):
            m = re.match(r"^[\*\-]\s+(.+)$", linea_strip)
            p = doc.add_paragraph(style="List Bullet")
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing = 1.15
            parsear_texto_con_formato(p, m.group(1), font_name="Calibri", font_size=12, base_color=COLOR_DARK)
            continue

        # Listas numeradas (1. 2. etc.)
        if re.match(r"^\d+\.\s+(.+)$", linea_strip):
            m = re.match(r"^\d+\.\s+(.+)$", linea_strip)
            p = doc.add_paragraph(style="List Number")
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing = 1.15
            parsear_texto_con_formato(p, m.group(1), font_name="Calibri", font_size=12, base_color=COLOR_DARK)
            continue

        # Párrafos regulares
        if linea_strip:
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(6)
            p.paragraph_format.line_spacing = 1.15
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            parsear_texto_con_formato(p, linea_strip, font_name="Calibri", font_size=12, base_color=COLOR_DARK)


    doc.save(docx_path)
    print(f"[OK] Archivo Word generado exitosamente: {docx_path}")


def exportar_a_pdf(ruta_docx: str, ruta_pdf: str):
    """Exporta el archivo .docx a PDF usando Microsoft Word COM."""
    print("Iniciando exportación a PDF vía Word COM...")
    abs_docx = os.path.abspath(ruta_docx)
    abs_pdf = os.path.abspath(ruta_pdf)
    
    import win32com.client
    import pythoncom

    pythoncom.CoInitialize()
    word = None
    try:
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False
        doc = word.Documents.Open(abs_docx)
        # 17 = wdExportFormatPDF
        doc.ExportAsFixedFormat(abs_pdf, 17)
        doc.Close(False)
        print(f"[OK] Documento PDF compilado exitosamente: {abs_pdf}")
    except Exception as e:
        print(f"[AVISO] Error al compilar con Word COM: {e}")
    finally:
        if word:
            word.Quit()
        pythoncom.CoUninitialize()


def main():
    parser = argparse.ArgumentParser(description="Generador pericial de entregables Word (.docx). Directriz: Jamás generar PDF.")
    parser.add_argument("--informe", default="informe.md", help="Ruta al borrador Markdown.")
    parser.add_argument("--plantilla", default="plantilla_estudiante_aula40.docx", help="Ruta a plantilla oficial.")
    parser.add_argument("--asignatura", default="ANÁLISIS FORENSE DIGITAL", help="Nombre oficial de la asignatura.")
    parser.add_argument("--semana", default="Semana 7", help="Número de semana (ej: Semana 7).")
    parser.add_argument("--estudiante", default="Security Analyst", help="Nombre del estudiante.")
    parser.add_argument("--carrera", default="Ingeniería en Ciberseguridad", help="Nombre de la carrera.")
    parser.add_argument("--fecha", default=datetime.now().strftime("%d-%m-%Y"), help="Fecha de entrega (DD-MM-YYYY).")
    parser.add_argument("--output-docx", default=None, help="Nombre del archivo Word de salida.")
    args = parser.parse_args()

    if not os.path.exists(args.informe):
        print(f"[ERROR] No existe el archivo de informe: {args.informe}")
        sys.exit(1)

    if not os.path.exists(args.plantilla):
        print(f"[ERROR] No existe la plantilla institucional: {args.plantilla}")
        sys.exit(1)

    # Nombres de salida por defecto
    m_sem = re.search(r"\d+", args.semana)
    num_sem = m_sem.group(0) if m_sem else "X"
    
    out_docx = args.output_docx or f"Juan_Araneda_Tarea{num_sem}.docx"

    print("=" * 80)
    print("GENERADOR PERICIAL DE ENTREGABLES ACADÉMICOS (ANTIGRAVITY)")
    print("Directriz Permanente: Formato Exclusivo Word (.docx) — Prohibido Generar PDF")
    print("=" * 80)
    print(f"Borrador origen: {args.informe}")
    print(f"Plantilla base:  {args.plantilla}")
    print(f"Asignatura:      {args.asignatura}")
    print(f"Semana:          {args.semana}")
    print(f"Estudiante:      {args.estudiante}")
    print(f"Fecha:           {args.fecha}")
    print(f"Destino Word:    {out_docx}")
    print("-" * 80)

    # 1. Respaldo
    respaldar_plantilla(args.plantilla)

    # 2. Portada
    datos_portada = {
        "asignatura": args.asignatura,
        "semana": args.semana,
        "estudiante": args.estudiante,
        "carrera": args.carrera,
        "fecha": args.fecha
    }
    actualizar_portada_xml(args.plantilla, out_docx, datos_portada)

    # 3. Inyección de contenido
    procesar_markdown_a_docx(args.informe, out_docx)

    print(f"[OK] Documento Word institucional generado exitosamente: {out_docx}")
    print("[INFO] Directriz permanente aplicada: NO se genera PDF. El entregable oficial es .docx.")
    print("=" * 80)
    print("PROCESO DE GENERACIÓN COMPLETADO EXITOSAMENTE")
    print("=" * 80)


if __name__ == "__main__":
    main()
