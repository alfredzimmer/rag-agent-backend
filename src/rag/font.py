import fitz  # PyMuPDF

doc = fitz.open("src/rag/public/IEEE Std 739-1995-166-168.pdf")

for page_number, page in enumerate(doc, start=1):
    blocks = page.get_text("dict")["blocks"]

    for b in blocks:
        if "lines" not in b:
            continue

        for line in b["lines"]:
            for span in line["spans"]:
                text = span["text"]
                font_size = span["size"]
                font_name = span["font"]
                color = span["color"]
                bold = (span['flags'] & 16) or "bold" in font_name.lower()

                print(f"Page {page_number} | Size: {font_size:<5} | Font: {font_name:<15} | Bold: {bold:<5} | Text: {text}")