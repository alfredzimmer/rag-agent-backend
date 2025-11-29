from pathlib import Path
import sys

src_dir = Path('/home/frank_shan/dev/python/pyapi/src')
sys.path.insert(0, str(src_dir))

import json
from pathlib import Path
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from tqdm import tqdm

# Define paths
corpus_dir = Path("src/evaluation/rag-syn-corpus")
output_dir = Path("src/evaluation/rag-eval-goldens")
output_dir.mkdir(exist_ok=True)

print("✓ Imports and paths configured")

class QAPair(BaseModel):
    question: str = Field(description="A specific, answerable question based on the content")
    answer: str = Field(description="A concise, accurate answer derived from the content")

# Initialize Ollama with qwen30b
llm = ChatOllama(model="qwen3:30b-instruct", temperature=0.7)

# Create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at creating evaluation question-answer pairs for RAG systems.
    
Given a text chunk, generate ONE high-quality question-answer pair that:
- The question should be specific and answerable from the content
- The answer should be concise and accurate
- Focus on key facts, concepts, or procedures mentioned in the text
- Avoid yes/no questions; prefer what/how/why questions

Return your response as JSON with 'question' and 'answer' fields."""),
    ("user", "Text chunk:\n\n{content}\n\nGenerate a question-answer pair:")
])

# Create the chain
parser = JsonOutputParser(pydantic_object=QAPair)
chain = prompt | llm | parser

print("✓ LLM and prompt chain initialized")

def generate_qa_pair(chunk):
    """Generate a Q&A pair from a single chunk."""
    try:
        result = chain.invoke({"content": chunk["content"]})
        return {
            "input": result["question"],
            "output": result["answer"]
        }
    except Exception as e:
        print(f"Error processing chunk {chunk.get('chunk_id', 'unknown')}: {e}")
        return None

def process_corpus_file(filepath):
    """Process a single corpus file and generate Q&A pairs."""
    print(f"\nProcessing: {filepath.name}")
    
    # Load corpus chunks
    with open(filepath, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    # Generate Q&A pairs for each chunk
    qa_pairs = []
    for chunk in tqdm(chunks, desc="Generating Q&A pairs"):
        qa_pair = generate_qa_pair(chunk)
        if qa_pair:
            qa_pairs.append(qa_pair)
    
    # Modify output filename: A.json -> A.test.json
    stem = filepath.stem   # 'A'
    output_filename = f"{stem}.test.json"
    output_file = output_dir / output_filename
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, indent=4, ensure_ascii=False)
    
    print(f"✓ Saved {len(qa_pairs)} Q&A pairs to {output_file.name}")
    return len(qa_pairs)

print("✓ Processing functions defined")

corpus_files = sorted(corpus_dir.glob("*.json"))
print(f"Found {len(corpus_files)} corpus files to process\n")
print("=" * 70)

total_pairs = 0
for filepath in corpus_files:
    count = process_corpus_file(filepath)
    total_pairs += count

print("\n" + "=" * 70)
print(f"✓ Complete! Generated {total_pairs} total Q&A pairs")
print(f"✓ Output directory: {output_dir.absolute()}")