import pymupdf4llm
import pathlib

FILE_PATH = "/Users/Wayne/Documents/ZiyutecFall25/pyapi/pymupdf/public/IEEE_sample_30.pdf"

md_text = pymupdf4llm.to_markdown(FILE_PATH)

pathlib.Path("output.md").write_bytes(md_text.encode())

