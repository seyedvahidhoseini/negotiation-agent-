import fitz
import pdfplumber
from pathlib import Path
import re
import json
import uuid


MAX_CHUNK_CHARS = 1200  # مناسب embedding models


def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def detect_heading(line_text, font_size, max_font_size):
    if font_size >= max_font_size * 0.9:
        return "h1"
    elif font_size >= max_font_size * 0.75:
        return "h2"
    elif font_size >= max_font_size * 0.6:
        return "h3"
    return "p"


def extract_text_with_structure(pdf_path):
    doc = fitz.open(pdf_path)
    structured_content = []

    for page_num, page in enumerate(doc, start=1):

        blocks = page.get_text("dict")["blocks"]

        max_font_size = 0
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        max_font_size = max(max_font_size, span["size"])

        current_section = "Unknown"

        for block in blocks:
            if "lines" not in block:
                continue

            for line in block["lines"]:
                text = ""
                font_size = 0

                for span in line["spans"]:
                    text += span["text"]
                    font_size = max(font_size, span["size"])

                text = clean_text(text)
                if not text:
                    continue

                tag = detect_heading(text, font_size, max_font_size)

                if tag in ["h1", "h2", "h3"]:
                    current_section = text

                structured_content.append({
                    "page": page_num,
                    "section": current_section,
                    "type": tag,
                    "content": text
                })

    doc.close()
    return structured_content


def extract_tables(pdf_path):
    table_data = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()

            for table in tables:
                rows = []
                for row in table:
                    row = [cell if cell else "" for cell in row]
                    row = [cell.replace("\n", " ") for cell in row]
                    rows.append(" | ".join(row))

                table_text = "\n".join(rows)

                table_data.append({
                    "page": page_num,
                    "section": "Table",
                    "type": "table",
                    "content": table_text
                })

    return table_data


def chunk_for_rag(structured_content):
    chunks = []
    buffer = ""
    current_meta = None

    for item in structured_content:

        text = item["content"]

        if not current_meta:
            current_meta = item

        if len(buffer) + len(text) < MAX_CHUNK_CHARS:
            buffer += " " + text
        else:
            chunks.append({
                "id": str(uuid.uuid4()),
                "content": buffer.strip(),
                "metadata": {
                    "page": current_meta["page"],
                    "section": current_meta["section"],
                    "type": current_meta["type"]
                }
            })
            buffer = text
            current_meta = item

    if buffer:
        chunks.append({
            "id": str(uuid.uuid4()),
            "content": buffer.strip(),
            "metadata": {
                "page": current_meta["page"],
                "section": current_meta["section"],
                "type": current_meta["type"]
            }
        })

    return chunks


def pdf_to_rag_json(pdf_path, output_path):

    text_data = extract_text_with_structure(pdf_path)
    table_data = extract_tables(pdf_path)

    all_data = text_data + table_data

    rag_chunks = chunk_for_rag(all_data)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rag_chunks, f, ensure_ascii=False, indent=2)

    print(f"✅ فایل JSON آماده RAG ساخته شد: {output_path}")


if __name__ == "__main__":
    input_pdf = "skills.pdf"
    output_json = "output/rag_ready.json"

    pdf_to_rag_json(input_pdf, output_json)
