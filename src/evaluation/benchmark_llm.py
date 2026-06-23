from dotenv import load_dotenv
# Load environment variables first so libraries like ollama read the correct config on import
load_dotenv()

import os
import sys
import random
import time
import json
import argparse
from pathlib import Path
import fitz  # PyMuPDF
import ollama
from tqdm import tqdm

# Configure paths
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
FILES_DIR = Path(os.getenv("INGESTION_UPLOAD_DIR", ".runtime/uploads")).resolve()
REPORTS_DIR = PROJECT_ROOT / "data" / "eval-reports"

# Ensure reports directory exists
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Set model
config_model_name = os.getenv("RAG_LLM_MODEL", "qwen3.6:latest")

def resolve_model_name(model_name: str) -> str:
    try:
        model_list = ollama.list()
        # ollama.list() returns a ModelResponse or a dict, handle both safely
        available_models = []
        if hasattr(model_list, "models"):
            available_models = [m.model for m in model_list.models]
        elif isinstance(model_list, dict) and "models" in model_list:
            available_models = [m.get("name") if isinstance(m, dict) else getattr(m, "model", "") for m in model_list["models"]]

        # Remove empty strings
        available_models = [m for m in available_models if m]

        if model_name in available_models:
            return model_name
        if f"{model_name}:latest" in available_models:
            return f"{model_name}:latest"
        if model_name.endswith(":latest") and model_name[:-7] in available_models:
            return model_name[:-7]

        # Fallback to a qwen model if available
        qwen_models = [m for m in available_models if "qwen" in m.lower()]
        if qwen_models:
            return qwen_models[0]

        if available_models:
            return available_models[0]
    except Exception as e:
        print(f"Warning during model resolution: {e}")
    return model_name

MODEL_NAME = resolve_model_name(config_model_name)

def scan_files(base_dir: Path) -> list[Path]:
    """Recursively scan base_dir for supported files (.pdf, .txt, .md)."""
    supported_extensions = {".pdf", ".txt", ".md"}
    found_files = []

    if not base_dir.exists():
        print(f"Error: Directory {base_dir} does not exist.")
        return []

    for p in base_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in supported_extensions:
            # Skip hidden files and macOS metadata files
            if not p.name.startswith(".") and not p.name.startswith("._"):
                found_files.append(p)

    return found_files

def extract_snippet(file_path: Path, min_chars: int = 600, max_chars: int = 1500) -> str:
    """Extract a random cohesive text snippet from a file."""
    suffix = file_path.suffix.lower()
    text = ""

    try:
        if suffix == ".pdf":
            doc = fitz.open(file_path)
            num_pages = len(doc)
            if num_pages == 0:
                return ""

            # Select a random starting page
            start_page = random.randint(0, num_pages - 1)
            # Try to get enough text from start_page onwards
            for page_num in range(start_page, num_pages):
                page = doc[page_num]
                text += page.get_text()
                if len(text) >= min_chars:
                    break
            doc.close()

        elif suffix in {".txt", ".md"}:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

    except Exception as e:
        print(f"Warning: Failed to parse {file_path.name}: {e}")
        return ""

    # Clean text and find a cohesive block
    text = " ".join(text.split())
    if len(text) < min_chars:
        return text

    # Randomly slice a block if the file is very large
    max_start = max(0, len(text) - max_chars)
    start_idx = random.randint(0, max_start)
    snippet = text[start_idx:start_idx + max_chars]

    # Clean up ends to align with sentences
    first_period = snippet.find(".")
    last_period = snippet.rfind(".")

    if 0 <= first_period < len(snippet) // 3:
        snippet = snippet[first_period + 1:]
    if last_period > len(snippet) // 2:
        snippet = snippet[:last_period + 1]

    return snippet.strip()

def robust_json_llm_call(prompt: str, max_retries: int = 3) -> dict:
    """Call Ollama and ensure a JSON object is parsed from the response."""
    for attempt in range(max_retries):
        try:
            response = ollama.generate(
                model=MODEL_NAME,
                prompt=prompt,
                options={"temperature": 0.3},
            )
            content = response.get("response", "").strip()

            # Find boundaries of JSON object
            start_idx = content.find("{")
            end_idx = content.rfind("}")
            if start_idx != -1 and end_idx != -1:
                json_str = content[start_idx:end_idx + 1]
                return json.loads(json_str)

            # If no JSON brackets found, try parsing directly
            return json.loads(content)
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(1)

    return {}

def generate_qa_pair(snippet: str, source_name: str) -> dict:
    """Generate a question and answer based on the snippet."""
    prompt = f"""You are a RAG system evaluation test case generator.
Based on the following context snippet from the document "{source_name}", generate a single, specific technical question that can be answered using ONLY this context.
Also, generate the correct, concise answer based strictly on the context.

Context:
{snippet}

Format your response strictly as a JSON object with exactly two keys:
- "question": "your generated question"
- "answer": "the ground truth answer based on the context"

Do not include any other text, markdown formatting (like ```json), explanations, or preamble in your response. Return ONLY the JSON object.
"""
    try:
        qa = robust_json_llm_call(prompt)
        if "question" in qa and "answer" in qa:
            return qa
    except Exception as e:
        print(f"Warning: Failed to generate QA pair: {e}")

    return {}

def run_benchmark(question: str, context: str) -> dict:
    """Query Ollama and measure TTFT, throughput, and latency."""
    prompt = f"""Use the following context to answer the question. Be accurate, concise, and do not hallucinate details outside the context.

Context:
{context}

Question:
{question}
"""

    start_time = time.time()
    ttft = None
    content = ""
    chunks_received = 0

    try:
        stream = ollama.generate(
            model=MODEL_NAME,
            prompt=prompt,
            stream=True,
            options={"temperature": 0.0}  # Deterministic for benchmark
        )

        for chunk in stream:
            chunks_received += 1
            if ttft is None:
                ttft = time.time() - start_time
            content += chunk.get("response", "")

            # Check if this is the final chunk containing statistics
            if chunk.get("done", False):
                total_duration_ns = chunk.get("total_duration")
                eval_count = chunk.get("eval_count")
                eval_duration_ns = chunk.get("eval_duration")
                prompt_eval_count = chunk.get("prompt_eval_count")

                # Convert nanoseconds to seconds
                total_time = total_duration_ns / 1e9 if total_duration_ns else (time.time() - start_time)
                eval_duration = eval_duration_ns / 1e9 if eval_duration_ns else total_time

                # Calculate metrics
                tokens_per_sec = (eval_count / eval_duration) if (eval_count and eval_duration) else (chunks_received / total_time)

                return {
                    "actual_output": content.strip(),
                    "ttft_sec": ttft if ttft is not None else total_time,
                    "total_time_sec": total_time,
                    "tokens_per_sec": tokens_per_sec,
                    "output_tokens": eval_count or chunks_received,
                    "input_tokens": prompt_eval_count or 0,
                    "status": "success"
                }

    except Exception as e:
        print(f"Error during benchmark inference: {e}")
        return {
            "actual_output": f"Error: {e}",
            "ttft_sec": 0.0,
            "total_time_sec": 0.0,
            "tokens_per_sec": 0.0,
            "output_tokens": 0,
            "input_tokens": 0,
            "status": "failed"
        }

    # Fallback in case streaming didn't output statistics in the final chunk
    total_time = time.time() - start_time
    words = len(content.split())
    approx_tokens = int(words * 1.3)

    return {
        "actual_output": content.strip(),
        "ttft_sec": ttft if ttft is not None else total_time,
        "total_time_sec": total_time,
        "tokens_per_sec": approx_tokens / max(0.1, total_time - (ttft or 0)),
        "output_tokens": approx_tokens,
        "input_tokens": 0,
        "status": "success"
    }

def evaluate_quality(context: str, question: str, actual_output: str, expected_output: str) -> dict:
    """Evaluate faithfulness, relevancy, and coherence of the output."""

    # 1. Check Coherence (Local structural heuristic)
    # Check for excessive repetition (loops)
    words = actual_output.split()
    word_count = len(words)
    repetitive = False
    coherence_reason = "Answer is structurally sound."

    if word_count > 10:
        # Check for repeating 3-word n-grams
        ngrams = [" ".join(words[i:i+3]) for i in range(word_count - 2)]
        unique_ngrams = set(ngrams)
        repetition_ratio = 1.0 - (len(unique_ngrams) / len(ngrams))
        if repetition_ratio > 0.4:
            repetitive = True
            coherence_reason = f"Detected high repetition loop (repetition ratio: {repetition_ratio:.2%})."

    # Check if empty or too short
    if word_count < 3 or len(actual_output) < 10:
        repetitive = True
        coherence_reason = "Response is empty or too short to be coherent."

    coherence_score = 0.0 if repetitive else 1.0

    # 2. Faithfulness (Is the answer supported by context? No hallucinations?)
    faithfulness_prompt = f"""You are an LLM evaluation judge. Evaluate if the generated answer is fully supported by the provided context without introducing outside facts or contradictions.

Context:
{context}

Generated Answer:
{actual_output}

Return your response strictly as a JSON object with two keys:
- "faithful": true/false (true if the answer is completely supported, false otherwise)
- "reason": "a brief explanation of your decision"

Return ONLY the JSON object. Do not include markdown headers or other text.
"""

    faith_score = 0.0
    faith_reason = "Failed to run LLM judge."
    try:
        res = robust_json_llm_call(faithfulness_prompt)
        faith_score = 1.0 if res.get("faithful") is True else 0.0
        faith_reason = res.get("reason", "")
    except Exception as e:
        faith_reason = f"LLM judge error: {e}"

    # 3. Answer Relevancy (Does it answer the question?)
    relevancy_prompt = f"""You are an LLM evaluation judge. Evaluate if the generated answer directly and accurately answers the question asked, without being vague, off-topic, or evasive.

Question:
{question}

Generated Answer:
{actual_output}

Return your response strictly as a JSON object with two keys:
- "relevant": true/false (true if the answer is highly relevant, false otherwise)
- "reason": "a brief explanation of your decision"

Return ONLY the JSON object. Do not include markdown headers or other text.
"""

    rel_score = 0.0
    rel_reason = "Failed to run LLM judge."
    try:
        res = robust_json_llm_call(relevancy_prompt)
        rel_score = 1.0 if res.get("relevant") is True else 0.0
        rel_reason = res.get("reason", "")
    except Exception as e:
        rel_reason = f"LLM judge error: {e}"

    return {
        "coherence": coherence_score,
        "coherence_reason": coherence_reason,
        "faithfulness": faith_score,
        "faithfulness_reason": faith_reason,
        "relevancy": rel_score,
        "relevancy_reason": rel_reason
    }

def main():
    parser = argparse.ArgumentParser(description="Ollama LLM Performance and Quality Benchmark Suite")
    parser.add_argument("--samples", type=int, default=5, help="Number of random test cases to generate and run")
    args = parser.parse_args()

    print("=" * 70)
    print(f"STARTING LLM BENCHMARK AND QUALITY EVALUATION")
    print(f"Target Model: {MODEL_NAME}")
    print(f"Files Directory: {FILES_DIR}")
    print(f"Requested Test Cases: {args.samples}")
    print("=" * 70)

    # 1. Discover files
    print("\n[1/5] Discovering documents in files folder...")
    all_files = scan_files(FILES_DIR)
    if not all_files:
        print("Error: No documents (.pdf, .txt, .md) found.")
        sys.exit(1)
    print(f"Found {len(all_files)} files.")

    # 2. Select samples and extract snippets
    print("\n[2/5] Sampling files and extracting text snippets...")
    test_cases = []
    attempts = 0
    max_attempts = args.samples * 3

    # Seed for reproducibility if needed, but random is requested
    random.shuffle(all_files)

    with tqdm(total=args.samples, desc="Extracting Snippets") as pbar:
        while len(test_cases) < args.samples and attempts < len(all_files):
            file = all_files[attempts]
            attempts += 1

            snippet = extract_snippet(file)
            if not snippet or len(snippet) < 400:
                continue

            test_cases.append({
                "source_file": file.name,
                "source_path": str(file),
                "snippet": snippet
            })
            pbar.update(1)

    if len(test_cases) < args.samples:
        print(f"Warning: Only extracted {len(test_cases)} valid text snippets from files.")

    # 3. Generate QA test pairs
    print(f"\n[3/5] Synthesizing QA test cases using {MODEL_NAME}...")
    valid_test_cases = []

    for tc in tqdm(test_cases, desc="Generating QA Cases"):
        qa = generate_qa_pair(tc["snippet"], tc["source_file"])
        if qa:
            tc.update(qa)
            valid_test_cases.append(tc)

    if not valid_test_cases:
        print("Error: Failed to generate any valid QA test cases.")
        sys.exit(1)
    print(f"Successfully generated {len(valid_test_cases)} QA pairs.")

    # 4. Execute Benchmark
    print(f"\n[4/5] Running inference benchmark on {MODEL_NAME}...")
    results = []

    for idx, tc in enumerate(valid_test_cases, 1):
        print(f"\nRunning test case {idx}/{len(valid_test_cases)} ({tc['source_file']})...")
        print(f"Q: {tc['question']}")

        bench = run_benchmark(tc["question"], tc["snippet"])
        tc.update(bench)

        # 5. Evaluate quality
        print(f"Evaluating quality metrics...")
        quality = evaluate_quality(tc["snippet"], tc["question"], tc["actual_output"], tc["answer"])
        tc.update(quality)

        results.append(tc)

        # Immediate console summary of case
        print(f"  Speed: {tc['tokens_per_sec']:.2f} t/s | TTFT: {tc['ttft_sec']:.2f}s | Total: {tc['total_time_sec']:.2f}s")
        print(f"  Scores: Coherence={tc['coherence']} | Faithfulness={tc['faithfulness']} | Relevancy={tc['relevancy']}")

    # Calculate aggregate benchmarks
    total_cases = len(results)
    avg_ttft = sum(r["ttft_sec"] for r in results) / total_cases
    avg_tps = sum(r["tokens_per_sec"] for r in results) / total_cases
    avg_latency = sum(r["total_time_sec"] for r in results) / total_cases

    avg_coherence = sum(r["coherence"] for r in results) / total_cases
    avg_faithfulness = sum(r["faithfulness"] for r in results) / total_cases
    avg_relevancy = sum(r["relevancy"] for r in results) / total_cases

    # Print console summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"Time to First Token (TTFT) avg: {avg_ttft:.2f} seconds")
    print(f"Inference Speed avg:             {avg_tps:.2f} tokens/second")
    print(f"Total Response Latency avg:      {avg_latency:.2f} seconds")
    print("-" * 70)
    print(f"Coherence Score avg:             {avg_coherence:.1%}")
    print(f"Faithfulness Score avg:          {avg_faithfulness:.1%}")
    print(f"Answer Relevancy Score avg:      {avg_relevancy:.1%}")
    print("=" * 70)

    # 6. Generate detailed markdown report
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_file = REPORTS_DIR / f"benchmark_report_{timestamp}.md"
    summary_file = REPORTS_DIR / "benchmark_summary.md"

    report_content = f"""# LLM Performance and Quality Benchmark Report

- **Date / Time:** {time.strftime("%Y-%m-%d %H:%M:%S")}
- **Target Model:** `{MODEL_NAME}`
- **Test Set size:** {total_cases} random document snippets from the configured ingestion upload directory

## Executive Summary

| Metric | Average Value | Description |
| :--- | :--- | :--- |
| **TTFT (Latency)** | **{avg_ttft:.3f} s** | Average time to generate the first token (Model loading/prompt evaluation) |
| **Throughput (Speed)** | **{avg_tps:.2f} t/s** | Average tokens generated per second during response generation |
| **Response Latency** | **{avg_latency:.3f} s** | Average total time to generate the full response |
| **Coherence Score** | **{avg_coherence:.1%}** | Percentage of answers that did not experience structural loops/errors |
| **Faithfulness Score** | **{avg_faithfulness:.1%}** | Percentage of answers completely grounded in context (no hallucination) |
| **Answer Relevancy** | **{avg_relevancy:.1%}** | Percentage of answers directly addressing the user's question |

---

## Detailed Test Cases

"""

    for idx, r in enumerate(results, 1):
        report_content += f"""### Test Case {idx}: {r['source_file']}
- **Tokens/sec:** {r['tokens_per_sec']:.2f} t/s | **TTFT:** {r['ttft_sec']:.3f}s | **Total Latency:** {r['total_time_sec']:.3f}s
- **Token Counts:** {r['input_tokens']} input / {r['output_tokens']} output

#### Context Snippet
```text
... {r['snippet'][:500]} ...
```

#### Generated QA Pair
- **Synthesized Question:** {r['question']}
- **Ground Truth Answer:** {r['answer']}

#### Model Response
- **Status:** `{r['status']}`
- **Actual Response:**
{r['actual_output']}

#### Quality Evaluation
- **Coherence:** `{r['coherence']}` (Score: {r['coherence']:.1f})
  - *Reason:* {r['coherence_reason']}
- **Faithfulness:** `{r['faithfulness']}` (Score: {r['faithfulness']:.1f})
  - *Reason:* {r['faithfulness_reason']}
- **Relevancy:** `{r['relevancy']}` (Score: {r['relevancy']:.1f})
  - *Reason:* {r['relevancy_reason']}

---
"""

    # Save timestamped report
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_content)

    # Save as latest summary file
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"\n✓ Saved detailed benchmark report to {report_file}")
    print(f"✓ Saved latest summary report to {summary_file}")

if __name__ == "__main__":
    main()
