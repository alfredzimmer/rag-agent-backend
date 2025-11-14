import fitz  # PyMuPDF

doc = fitz.open("public/IEEE Blue Book Std 1015-2006-13-30.pdf")

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

                print(f"Page {page_number} | Size: {font_size:<5} | Font: {font_name:<15} | Text: {text}")