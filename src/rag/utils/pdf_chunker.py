import pymupdf
import os
import re
# import pymupdf.layout
import pymupdf4llm
import pathlib
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ============================================================================
# CONFIGURATION SECTION - Modify these dictionaries to customize parsing
# ============================================================================

HEADER_CONFIGS = {
    "ieee": {
        "rules": [
            # Level 1: Chapter titles (e.g., "Chapter 2" or "Operating diagrams")
            {
                "size_range": (13.5, 16),
                "is_bold": True,
                "level": 1
            },
            # Level 2: Main headings with one dot (e.g., "2.1 Introduction")
            {
                "size_range": (10, 14),
                "pattern": r"^\d+\.\d+",
                "dot_count": 1,
                "level": 2,
                "truncate_at": ":"  # Only include text before colon
            },
            # Level 3: Subheadings with two dots (e.g., "2.1.18 ground-fault delay")
            {
                "size_range": (9.5, 12),
                "pattern": r"^\d+\.\d+\.\d+",
                "dot_count": 2,
                "level": 3,
                "truncate_at": ":"  # Only include text before colon
            },
            # Level 4: Three dots (e.g., "4.3.4.2")
            {
                "size_range": (9.5, 12),
                "pattern": r"^\d+\.\d+\.\d+\.\d+",
                "dot_count": 3,
                "level": 4,
                "truncate_at": ":"
            },
            # Level 5: Four dots (e.g., "5.1.2.2.2")
            {
                "size_range": (9.5, 12),
                "pattern": r"^\d+\.\d+\.\d+\.\d+\.\d+",
                "dot_count": 4,
                "level": 5,
                "truncate_at": ":"
            },
            # Level 6: Five dots (e.g., "5.1.2.2.2.1")
            {
                "size_range": (9.5, 12),
                "pattern": r"^\d+\.\d+\.\d+\.\d+\.\d+\.\d+",
                "dot_count": 5,
                "level": 6,
                "truncate_at": ":"
            }
        ],
        "exclusions": {
            "footer_zone": 0.88,  # Bottom 12% of page
            "header_zone": 0.08,  # Top 8% of page
            "min_header_size": 9
        },
        "margins": {
            "top": 0.08,     # Exclude top 8% of page from content
            "bottom": 0.08   # Exclude bottom 8% of page from content
        }
    },
    "nfpa": {
        "rules": [
            # Level 1: Chapter titles (e.g., "Chapter 2")
            {
                "size_range": (18, 22),
                "font_name": "arial-black",
                "level": 1
            },
            # Level 2: Main headings
            {
                "size_range": (13, 18),
                "font_name": "arial-black",
                "level": 2
            },
            # Level 3: Subheadings
            {
                "size_range": (10.5, 13),
                "font_name": "arial-black",
                "level": 3
            },
            # Level 4: Minor subheadings
            {
                "size_range": (9.5, 10),
                "font_name": "arial-black",
                "level": 4
            },
            # Level 5: Smallest headers
            {
                "size_range": (9.5, 10),
                "font_name": "arial-black",
                "level": 5
            }
        ],
        "exclusions": {
            "footer_zone": 0.88,
            "header_zone": 0.12,
            "min_header_size": 9
        },
        "margins": {
            "top": 0.05,     # Exclude top 5% of page from content
            "bottom": 0.08   # Exclude bottom 8% of page from content
        }
    }
}

FOOTER_CONFIGS = {
    "ieee": {
        "patterns": [
            # r"^\d+$",  # Page numbers
            r"^Page \d+",
            r"^\d+ of \d+$",
            r"^Copyright.*IEEE.*All rights reserved\.?\s*\d*$",
            r"^IEEE\s*$",
            r"^Std \d+-\d+",
            r".*Std \d+-\d+ CHAPTER \d+$",
            r"^OPERATING DIAGRAMS Std \d+-\d+$"
        ],
        "keywords": ["IEEE", "CHAPTER", "Authorized licensed use limited to"],
        "copyright_keywords": ["©", "IEEE", "Copyright"],
        "min_line_length": 3,
        "max_copyright_length": 150
    },
    "nfpa": {
        "patterns": [
            # r"^\d+$",  # Page numbers
            r"^Page \d+",
            r"^\d+ of \d+$",
            r"^Copyright.*NFPA",
            r"^NFPA\s*$"
        ],
        "keywords": ["NFPA", "CHAPTER"],
        "copyright_keywords": ["©", "NFPA", "Copyright"],
        "min_line_length": 3,
        "max_copyright_length": 150
    }
}

# ============================================================================
# CONFIGURABLE CLASSES AND FUNCTIONS
# ============================================================================

class ConfigurableHeaderDetector:
    """
    A configurable header detector that uses rules from HEADER_CONFIGS.
    Implements the 'get_header_id' method required by pymupdf4llm's to_markdown().
    """
    
    def __init__(self, config_name: str = "ieee"):
        """
        Initialize with a configuration name.
        
        Args:
            config_name: Name of the configuration to use from HEADER_CONFIGS
        """
        if config_name not in HEADER_CONFIGS:
            raise ValueError(f"Unknown config: {config_name}. Available: {list(HEADER_CONFIGS.keys())}")
        
        self.config = HEADER_CONFIGS[config_name]
        self.rules = self.config["rules"]
        self.exclusions = self.config["exclusions"]
        
        # Compile regex patterns for efficiency
        self.compiled_patterns = {}
        for i, rule in enumerate(self.rules):
            if "pattern" in rule:
                self.compiled_patterns[i] = re.compile(rule["pattern"])

    def get_header_id(self, span, page=None):
        """
        Determine if a text span is a header and return its markdown level.
        
        Args:
            span: Text span dictionary from PyMuPDF
            page: Page object (optional)
            
        Returns:
            String: "" (not a header) or "# ", "## ", etc.
        """
        text = span['text'].strip()
        if not text:
            return ""
        
        # Get span properties
        font = span['font'].lower()
        size = span['size']
        is_bold = (span['flags'] & 16) or "bold" in font
        
        # Apply exclusion rules if page info is available
        if page:
            page_height = page.rect.height
            span_y_bottom = span['bbox'][3]  # y2 coordinate
            
            # Exclude footers (bottom zone)
            if span_y_bottom > page_height * self.exclusions["footer_zone"]:
                if "Copyright" in text or text.isdigit():
                    return ""
            
            # Exclude small text at the top (headers)
            if (span_y_bottom < page_height * self.exclusions["header_zone"] and 
                size < self.exclusions["min_header_size"]):
                return ""
        
        # Check each rule in order
        for i, rule in enumerate(self.rules):
            # Check size range
            min_size, max_size = rule["size_range"]
            if not (min_size <= size <= max_size):
                continue
            
            # Check font name if specified
            if "font_name" in rule:
                if font != rule["font_name"]:
                    continue
            
            # Check bold flag if specified
            if "is_bold" in rule:
                if rule["is_bold"] and not is_bold:
                    continue
            
            # Check pattern if specified
            if "pattern" in rule:
                if i not in self.compiled_patterns:
                    continue
                if not self.compiled_patterns[i].match(text):
                    continue
                
                # Check dot count if specified
                if "dot_count" in rule:
                    if text.count('.') != rule["dot_count"]:
                        continue
            
            # All conditions met - return the header level
            level = rule["level"]
            return "#" * level + " "
        
        return ""


def remove_headers_footers_configurable(markdown_text: str, config_name: str = "ieee") -> str:
    """
    Remove common header/footer patterns from markdown text using configuration.
    
    Args:
        markdown_text: The markdown text to clean
        config_name: Name of the configuration to use from FOOTER_CONFIGS
        
    Returns:
        Cleaned markdown text
    """
    if config_name not in FOOTER_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(FOOTER_CONFIGS.keys())}")
    
    config = FOOTER_CONFIGS[config_name]
    lines = markdown_text.split('\n')
    cleaned_lines = []
    
    # Compile patterns for efficiency
    compiled_patterns = [re.compile(pattern) for pattern in config["patterns"]]
    
    for line in lines:
        stripped = line.strip()
        
        # Check if line should be skipped
        should_skip = False
        
        # Check against regex patterns
        for pattern in compiled_patterns:
            if pattern.match(stripped):
                should_skip = True
                break
        
        if should_skip:
            continue
        
        # Check against keywords
        if stripped in config["keywords"]:
            continue
        
        # Check for copyright with keywords
        copyright_kw = config["copyright_keywords"]
        if (any(kw in stripped for kw in copyright_kw) and 
            len(stripped) < config["max_copyright_length"]):
            continue
        
        # Skip very short lines (likely artifacts), but preserve markdown headers
        # if (len(stripped) < config["min_line_length"] and 
        #     stripped not in ['#', '##', '###', '####', '#####']):
        #     continue
        # DISABLED BECAUSE IT REMOVES CONSTANTS IN FRACTIONS
        
        cleaned_lines.append(line)
    
    # Remove excessive blank lines (more than 2 consecutive)
    result = '\n'.join(cleaned_lines)
    result = re.sub(r'\n{3,}', '\n\n', result)
    
    return result


def truncate_headers_at_colon(markdown_text: str) -> str:
    """
    Truncate markdown headers at colon symbol, moving content after ':' to the next line.
    
    This preserves the content after the colon (e.g., definitions) while keeping
    headers clean and concise.
    
    For example:
        "### 2.1.18 ground-fault protection of equipment: A system intended..."
        becomes:
        "### 2.1.18 ground-fault protection of equipment"
        "A system intended..."
    
    Args:
        markdown_text: The markdown text with headers
        
    Returns:
        Markdown text with truncated headers and preserved content
    """
    lines = markdown_text.split('\n')
    processed_lines = []
    
    for line in lines:
        # Check if line is a markdown header
        if line.startswith('#'):
            # Find where the header markup ends
            header_level = 0
            for char in line:
                if char == '#':
                    header_level += 1
                else:
                    break
            
            # Extract the header text (after the # symbols and space)
            if len(line) > header_level and line[header_level] == ' ':
                header_text = line[header_level + 1:]
                
                # Truncate at colon if present and preserve the content after it
                if ':' in header_text:
                    parts = header_text.split(':', 1)
                    header_part = parts[0].strip()
                    content_part = parts[1].strip() if len(parts) > 1 else ""
                    
                    # Add the truncated header
                    processed_lines.append('#' * header_level + ' ' + header_part)
                    
                    # Add the content after colon as a new line (if not empty)
                    if content_part:
                        processed_lines.append(content_part)
                    
                    continue  # Skip the normal append at the end
        
        processed_lines.append(line)
    
    return '\n'.join(processed_lines)



# ============================================================================
# MAIN PDF PROCESSING FUNCTIONS
# ============================================================================

def load_pdf_as_markdown(file: str, config_name: str = "ieee") -> str:
    """
    Extract text from a PDF file using configurable header detection.
    
    Args:
        file: Path to the PDF file
        config_name: Name of the configuration to use (e.g., "ieee", "nfpa")
        
    Returns:
        Markdown-formatted text
    """
    if config_name not in HEADER_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(HEADER_CONFIGS.keys())}")
    
    config = HEADER_CONFIGS[config_name]
    margin_config = config.get("margins", {"top": 0, "bottom": 0})
    
    with pymupdf.open(file) as doc:
        # Get typical page height from first page to calculate absolute margins
        first_page = doc[0]
        page_height = first_page.rect.height
        
        # Calculate absolute margins in points
        top_margin = page_height * margin_config.get("top", 0)
        bottom_margin = page_height * margin_config.get("bottom", 0)
        
        # Pass margins as (left, top, right, bottom) to exclude header/footer zones
        chunks = pymupdf4llm.to_markdown(
            doc,
            hdr_info=ConfigurableHeaderDetector(config_name),
            margins=(0, top_margin, 0, bottom_margin),
            show_progress=True
        )
        
        # Remove headers and footers
        chunks = remove_headers_footers_configurable(chunks, config_name)
        
        # Truncate headers at colon (keep only text before ':')
        chunks = truncate_headers_at_colon(chunks)
        
        return chunks


def split_pdf(file: str, config_name: str = "ieee") -> list[Document]:
    """
    Split the markdown conversion of a PDF file into chunks with size requirements.
    
    Args:
        file: Path to the PDF file
        config_name: Name of the configuration to use (e.g., "ieee", "nfpa")
        
    Returns:
        List of Document objects with chunked content
    """
    markdown = load_pdf_as_markdown(file, config_name)
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), 
                             ("###", "Header 3"), ("####", "Header 4"),
                             ("#####", "Header 5"), ("######", "Header 6")]
    )
    markdown_splits = markdown_splitter.split_text(markdown)
    
    filename = os.path.basename(file)          # "{filename}.pdf"
    name_without_ext = os.path.splitext(filename)[0] # "{filename}"
    
    # Save markdown file (ensure directory exists)
    md_output = pathlib.Path(f"src/rag/outputs/{name_without_ext}.md").resolve()
    md_output.parent.mkdir(parents=True, exist_ok=True)
    md_output.write_bytes(markdown.encode())
    
    # split raw_splits again with chunk size constraints
    chunk_size = 1024
    chunk_overlap = 256
    sized_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    
    sized_splits = sized_splitter.split_documents(markdown_splits)
    
    return sized_splits


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_splits_as_list(splits, additional_metadata=None, document_name=None) -> list[dict]:
    """
    Format the markdown header splits into a well-structured list of dictionaries.
    
    Args:
        splits: List of Document objects from MarkdownHeaderTextSplitter
        additional_metadata: Optional additional metadata to include
        document_name: Optional document name to include in to_append field
        
    Returns:
        List of dictionaries with to_append, metadata, content, and char_count
    """
    formatted_splits = []
    
    for i, doc in enumerate(splits):
        # Create to_append field with document name and headers
        to_append = {}
        if document_name:
            to_append["name"] = document_name
        to_append.update(doc.metadata)
        
        # Create metadata field with document name and headers
        metadata = {}
        if document_name:
            metadata["name"] = document_name
        metadata.update(doc.metadata)
        
        split_data = {
            "chunk_id": i,
            "to_append": to_append,
            "metadata": metadata,
            "content": doc.page_content.strip(),
            "char_count": len(doc.page_content)
        }
        
        if additional_metadata:
            split_data.update(additional_metadata)
        
        formatted_splits.append(split_data)
    
    return formatted_splits


# ============================================================================
# DEBUG TESTING
# ============================================================================

if __name__ == "__main__":
    # Example: Process NFPA document
    FILE_PATH = "src/rag/public/IEEE Blue Book Std 1015-2006-13-30.pdf"
    CONFIG = "ieee"  # Change to "ieee" for IEEE documents
    
    print(f"Processing {FILE_PATH} with config: {CONFIG}")
    chunks = split_pdf(FILE_PATH, CONFIG)
    
    # Get filename for document name
    filename = os.path.basename(FILE_PATH)
    name_without_ext = os.path.splitext(filename)[0]
    
    formatted_list = format_splits_as_list(chunks, document_name=name_without_ext) # can pass additional_metadata here
    
    # Save to JSON file
    import json
    output_file = pathlib.Path(f"src/rag/outputs/{name_without_ext}.json")
    output_file.write_text(json.dumps(formatted_list, indent=2, ensure_ascii=False))
    print(f"\nSaved {len(formatted_list)} chunks to {output_file}")
    
    # Print available configurations
    print(f"\nAvailable header configs: {list(HEADER_CONFIGS.keys())}")
    print(f"Available footer configs: {list(FOOTER_CONFIGS.keys())}")
