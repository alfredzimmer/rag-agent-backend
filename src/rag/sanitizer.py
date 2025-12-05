import json
import pathlib
from typing import List, Dict, Any
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from tqdm import tqdm

# Initialize LLM once for reuse
llm = ChatOllama(model="qwen3:30b-instruct", temperature=0, reasoning=False, num_ctx=32768)

SYSTEM_PROMPT = """You are a data cleaning specialist for text document chunks.
TASK: Clean and standardize raw text chunks while preserving their core content and meaning.
CLEANING OPERATIONS:
1. Content Normalization:
   - Remove timestamps, dates, and time markers
   - Remove greetings, salutations, small talk and sign-offs
   - Remove metadata (file paths, page numbers, headers, footers)
2. Speaker Label Standardization:
   - Remove speaker labels (i.e. 说话人)
3. Data Quality:
   - Standardize formatting inconsistencies (quotes, dashes, spacing)
4. Content Preservation Rules:
   - DO NOT remove substantive content or data points
   - DO NOT paraphrase or summarize
   - DO NOT alter technical terms, numbers, or domain-specific language
   - DO maintain document structure and logical flow
OUTPUT FORMAT:
Return ONLY the cleaned text content. No explanations, comments, or status messages.
If the chunk is completely empty or non-substantive after cleaning, return: "[EMPTY_CHUNK]"
"""

def sanitize_content(content: str) -> str:
    """Sanitize a single content string."""
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=content)
    ]
    response = llm.invoke(messages)
    return response.content

def sanitize_batch(contents: List[str], batch_size: int = 10) -> List[str]:
    """Sanitize multiple contents in parallel using LangChain's batch method."""
    # Prepare messages for batch processing
    batch_messages = []
    for content in contents:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=content)
        ]
        batch_messages.append(messages)
    
    # Process in batches for better memory management
    sanitized_contents = []
    for i in tqdm(range(0, len(batch_messages), batch_size), desc="Sanitizing contents"):
        batch = batch_messages[i:i + batch_size]
        responses = llm.batch(batch)
        sanitized_contents.extend([response.content for response in responses])
    
    return sanitized_contents

def process_json_file(input_path: pathlib.Path, output_path: pathlib.Path, batch_size: int = 10) -> None:
    """Read, sanitize, and save a single JSON file."""
    print(f"\n{'='*60}")
    print(f"Processing: {input_path.name}")
    print(f"{'='*60}")
    
    # Read JSON file
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        print(f"Warning: {input_path.name} is not a list, skipping...")
        return
    
    print(f"Found {len(data)} chunks")
    
    contents = [chunk.get('content', '') for chunk in data]
    
    sanitized_contents = sanitize_batch(contents, batch_size=batch_size)
    
    # Update chunks with sanitized content, and remove chunks with '[EMPTY_CHUNK]'
    new_data = []
    for chunk, sanitized_content in zip(data, sanitized_contents):
        if sanitized_content.strip() == '[EMPTY_CHUNK]':
            continue  # Skip this chunk
        chunk['content'] = sanitized_content
        if 'char_count' in chunk:
            chunk['char_count'] = len(sanitized_content)
        new_data.append(chunk)
    data[:] = new_data  # In-place replace data with the filtered list
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Saved sanitized content to {output_path.name}")

def sanitize_all_json_files(inputs_dir: str, outputs_dir: str, batch_size: int = 10):
    """Process all JSON files from inputs directory and save to outputs directory."""
    inputs_path = pathlib.Path(inputs_dir).resolve()
    outputs_path = pathlib.Path(outputs_dir).resolve()
    
    # Find all JSON files in inputs directory
    json_files_inputs = sorted(inputs_path.glob("*.json"))
    
    if not json_files_inputs:
        print(f"No JSON files found in {inputs_path}")
        return
    
    print(f"Found {len(json_files_inputs)} JSON files to process")
    print(f"Files: {[f.name for f in json_files_inputs]}")
    
    # Process each file
    for input_file in json_files_inputs:
        try:
            # Construct output path with same filename
            output_file = outputs_path / input_file.name
            process_json_file(input_file, output_file, batch_size=batch_size)
        except Exception as e:
            print(f"Error processing {input_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("All files processed!")
    print(f"{'='*60}")

sanitize_all_json_files(inputs_dir="src/rag/outputs/2023-2024", outputs_dir="src/rag/outputs/2023-2024/sanitized", batch_size=10)