import re

class IEEEHeaderDetector:
    """
    A custom object that implements the 'get_header_id' method, which
    is required by the 'hdr_info' parameter in to_markdown().
    
    This logic is tailored to IEEE Std series documents,
    prioritizing text patterns (regex) and font style (bold) 
    over simple font size.
    """
    
    def __init__(self):
        # Regex to find numbered sections like "2.1" or "3.3.4"
        self.number_pattern = re.compile(r"^\d+(\.\d+)+\s+")
        
        # Regex to find chapter titles like "Chapter 2"
        self.chapter_pattern = re.compile(r"^Chapter \d+", re.IGNORECASE)

    def get_header_id(self, span, page=None):
        """
        This is the method PyMuPDF will call for every text span.
        It must return "" (not a header) or "# ", "## ", etc.
        """
        text = span['text'].strip()
        if not text:
            return ""  # Not a header if it's empty

        # Get Span Properties
        font = span['font'].lower()
        size = round(span['size'])
        is_bold = (span['flags'] & 16) or "bold" in font

        if page:
            page_height = page.rect.height
            span_y_bottom = span['bbox'][3] # y2 coordinate

            # Exclude anything in the bottom ~12% of the page (e.g., footers)
            if span_y_bottom > page_height * 0.88:
                if "Copyright" in text or text.isdigit():
                    return ""
            
            # Exclude small text at the top (e.g., page headers)
            # Your "CHAPTER 3" header text was size 8.0
            if span_y_bottom < page_height * 0.12 and size < 10:
                 return ""

        # Inclusion Rules 
        
        # LEVEL 1: Chapter Title (e.g., "Chapter 2" or "Operating diagrams")
        if size == 14 and is_bold:
            return "# "

        # LEVEL 2: Main Heading (e.g., "2.1 Introduction", "3.4 Power factor")
        if size == 12 and is_bold and self.number_pattern.match(text):
             # Check how many dots to be sure
             if text.count('.') == 1:
                return "## "

        # LEVEL 3: Subheading (e.g., "3.3.3 Congested...", "3.3.4 Operating")
        if size == 10 and is_bold and self.number_pattern.match(text):
            if text.count('.') == 2:
                return "### "

        if size == 10 and is_bold and self.number_pattern.match(text):
            if text.count('.') >= 3:
                return "#### "
                
        return ""

def IEEE_remove_headers_footers(markdown_text: str) -> str:
    """
    Remove common header/footer patterns from markdown text.
    """
    lines = markdown_text.split('\n')
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()

        # Skip common header/footer patterns
        if (
            # Page numbers (standalone or with text)
            re.match(r'^\d+$', stripped) or
            re.match(r'^Page \d+', stripped, re.IGNORECASE) or
            re.match(r'^\d+ of \d+$', stripped) or

            # Copyright notices (IEEE specific and general)
            re.match(r'^Copyright.*IEEE.*All rights reserved\.?\s*\d*$', stripped) or
            '©' in stripped and 'IEEE' in stripped or
            'Copyright' in stripped and len(stripped) < 150 or

            # IEEE standard headers/footers
            stripped == 'IEEE' or
            re.match(r'^IEEE\s*$', stripped) or
            re.match(r'^Std \d+-\d+', stripped) or
            re.match(r'.*Std \d+-\d+ CHAPTER \d+$', stripped) or
            re.match(r'^OPERATING DIAGRAMS Std \d+-\d+$', stripped) or

            # Generic repeated headers
            stripped in ['CHAPTER', 'Authorized licensed use limited to'] or

            # Very short lines (likely artifacts), but preserve markdown headers
            (len(stripped) < 3 and stripped not in ['#', '##', '###', '####'])
        ):
            continue

        cleaned_lines.append(line)

    # Remove excessive blank lines (more than 2 consecutive)
    result = '\n'.join(cleaned_lines)
    result = re.sub(r'\n{3,}', '\n\n', result)

    return result