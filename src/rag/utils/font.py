import fitz  # PyMuPDF
import argparse


def extract_font_info(pdf_path: str, start_page: int = 1, end_page: int | None = None, max_pages: int | None = None):
    """
    Extract font information from a PDF document.
    
    Args:
        pdf_path: Path to the PDF file
        start_page: Starting page number (1-indexed, default: 1)
        end_page: Ending page number (1-indexed, inclusive, default: None for last page)
        max_pages: Maximum number of pages to process from start_page (default: None for all pages)
    """
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    
    # Determine the actual end page
    if max_pages is not None:
        calculated_end = start_page + max_pages - 1
        end_page = min(calculated_end, end_page if end_page else total_pages)
    elif end_page is None:
        end_page = total_pages
    
    # Validate page range
    start_page = max(1, start_page)
    end_page = min(end_page, total_pages)
    
    if start_page > end_page:
        print(f"Error: start_page ({start_page}) is greater than end_page ({end_page})")
        return
    
    print(f"Processing pages {start_page} to {end_page} (total: {end_page - start_page + 1} pages)\n")
    
    # Iterate through the specified page range (convert to 0-indexed for fitz)
    for page_number in range(start_page, end_page + 1):
        page = doc[page_number - 1]  # fitz uses 0-indexed pages
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
    
    doc.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract font information from PDF pages")
    parser.add_argument("pdf_path", nargs="?", default="/home/wayne_hao/pyapi/src/codes_and_standards/IEEE/IEEE_Std_739-1995.pdf",
                        help="Path to the PDF file")
    parser.add_argument("--start-page", type=int, default=1,
                        help="Starting page number (1-indexed, default: 1)")
    parser.add_argument("--end-page", type=int, default=None,
                        help="Ending page number (1-indexed, inclusive, default: last page)")
    parser.add_argument("--max-pages", type=int, default=None,
                        help="Maximum number of pages to process from start-page (default: all pages)")
    
    args = parser.parse_args()
    
    extract_font_info(args.pdf_path, args.start_page, args.end_page, args.max_pages)