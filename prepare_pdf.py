import json
import re
from pathlib import Path

from pypdf import PdfReader


def clean_text(text):
    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        # 把同一行里连续的空格和制表符压成一个空格。
        cleaned_lines.append(re.sub(r"[ \t\u3000]+", " ", line).strip())

    text = "\n".join(cleaned_lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def split_text(text, size=400, overlap=60):
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])

        if end == len(text):
            break

        start = end - overlap

    return chunks


def main():
    folder = Path(__file__).resolve().parent
    pdf_path = folder / "policy.pdf"

    if not pdf_path.is_file():
        raise SystemExit(f"找不到文件：{pdf_path}")

    reader = PdfReader(pdf_path)
    clean_pages = []
    records = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")

        if not text:
            print(f"第 {page_number} 页没有提取到文字，请检查原页。")
            continue

        clean_pages.append(f"=== PDF 第 {page_number} 页 ===\n{text}")

        for index, chunk in enumerate(split_text(text), start=1):
            records.append({
                "id": f"{pdf_path.stem}-p{page_number}-{index}",
                "source": pdf_path.name,
                "page": page_number,
                "text": chunk,
            })

    (folder / "policy_clean.txt").write_text(
        "\n\n".join(clean_pages), encoding="utf-8"
    )
    (folder / "policy_chunks.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"提取了 {len(clean_pages)} 页，切出 {len(records)} 个片段。")


if __name__ == "__main__":
    main()