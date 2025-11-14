"""
Module for synthesizing datasets from PDF documents.

This module provides functionality to:
1. Ingest PDFs from the documents folder
2. Process and chunk the PDFs
3. Generate clustered contexts for synthetic data generation
4. Generate synthetic instruction-response pairs using LLM (4-stage process):
   - Stage 1: Instruction generation from contexts
   - Stage 2: Instruction quality evaluation
   - Stage 3: Response generation with dual personas
   - Stage 4: Instruction-response pair quality evaluation
5. Save high-quality synthetic datasets to JSON files

Complete implementation following the methodology described in the research paper.
"""

import pathlib
import json
from tkinter import N
from typing import Optional, Literal
from datetime import datetime
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from pdf_chunker import split_pdf
from generate_context import generate_clustered_contexts
from IEEE_utils import IEEEHeaderDetector, IEEE_remove_headers_footers
from openai import OpenAI

# Load environment variables
load_dotenv()

class EvaluationResponse(BaseModel):
    score: int
    explanation: str

class DatasetSynthesizer:
    """
    Handles PDF ingestion and clustered context generation for synthetic dataset creation.
    """

    def __init__(
        self,
        documents_dir: str = "documents",
        outputs_dir: str = "outputs",
        header_detector=IEEEHeaderDetector,
        remove_headers_footers_func=IEEE_remove_headers_footers,
        chunking: bool = False,
        chunk_size: int = 512,
        chunk_overlap: int = 0,
    ):
        """
        Initialize the DatasetSynthesizer.

        Args:
            documents_dir: Path to directory containing PDF documents
            outputs_dir: Path to directory for saving outputs
            header_detector: Function/class to detect headers in PDFs
            remove_headers_footers_func: Function to remove headers/footers
            chunking: Whether to apply recursive chunking to markdown splits
            chunk_size: Size of text chunks for splitting (if chunking is True)
            chunk_overlap: Overlap between consecutive chunks (if chunking is True)
        """
        self.documents_dir = pathlib.Path(documents_dir)
        self.outputs_dir = pathlib.Path(outputs_dir)
        self.header_detector = header_detector
        self.remove_headers_footers_func = remove_headers_footers_func
        self.chunking = chunking
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Create output directory if it doesn't exist
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

        # Initialize OpenAI client (lazy initialization)
        self._llm_client = None

    def get_pdf_files(self) -> list[pathlib.Path]:
        """
        Get all PDF files from the documents directory.

        Returns:
            List of paths to PDF files
        """
        if not self.documents_dir.exists():
            raise ValueError(f"Documents directory not found: {self.documents_dir}")

        pdf_files = list(self.documents_dir.glob("*.pdf"))
        return sorted(pdf_files)

    def _get_llm_client(self) -> OpenAI:
        """Get or create OpenAI client (lazy initialization)."""
        if self._llm_client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            self._llm_client = OpenAI(api_key=api_key)
        return self._llm_client

    def _call_llm(self, prompt: str, temperature: float = 0.7, format: str = "text", model: str = "gpt-4o-mini") -> str:
        """
        Call the LLM with a prompt.

        Args:
            prompt: The prompt to send
            temperature: Sampling temperature
            format: Format of the response
            model: Model to use

        Returns:
            Generated text response
        """
        client = self._get_llm_client()

        output = None
        
        if format == "eval":
            parsed_response = client.responses.parse(
                model=model,
                input=[{"role": "user", "content": prompt}],
                temperature=temperature,
                text_format=EvaluationResponse
            )
            output = parsed_response.output
        else:
            response = client.responses.create(
                model=model,
                input=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            output = response.output[0].content.text

        return parsed_response.output

    def generate_instructions(self, context: str, num_samples: int = 5, icl_question: str = None) -> list[str]:
        """
        Stage 1: Generate diverse questions from a context.

        Args:
            context: The context to generate questions from
            num_samples: Number of questions to generate
            icl_question: In-context learning example question

        Returns:
            List of generated questions
        """
        if icl_question is None:
            icl_question = "What are the key design considerations for a low-noise amplifier in RF circuits?"

        prompt = f"""You are asked to come up with a set of {num_samples} diverse questions on Electrical Engineering based on the provided context.
Please follow these guiding principles when generating responses:
* Use proper grammar and punctuation.
* Always generate safe and respectful content. Do not generate content that is harmful, abusive, or offensive.
* Always generate content that is factually accurate and relevant to the prompt.
* The questions should be clear and human-like.
* The questions should be diverse and cover a wide range of topics.
* The questions should not be template-based or generic, it should be very diverse.
* Simply return the questions, do not return any answers or explanations.
* Strictly adhere to the prompt and generate responses in the same style and format as the example.

To better assist you with this task, here is an example:
### Question:
1. {icl_question}

Context:
{context}

Now generate {num_samples} such questions, remember to follow the principles mentioned above
and use the same format as the examples. Remember to use the same style and format as the example
above. Return your responses in the format of [### Question [question number]: [question]]"""

        response = self._call_llm(prompt, temperature=0.8)

        # Parse questions from response
        questions = []
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('### Question') or line[0].isdigit():
                # Remove numbering and "### Question" prefix
                question = line.split(':', 1)[-1].strip()
                if question:
                    questions.append(question)

        return questions

    def evaluate_instruction(self, question: str) -> dict:
        """
        Stage 2: Evaluate if a question meets quality criteria.

        Args:
            question: The question to evaluate

        Returns:
            Dictionary with 'passed' (bool) and 'reason' (str)
        """
        prompt = f"""Please evaluate whether the following question is suitable for an AI training dataset.

Question: {question}

Evaluation criteria:
1. Is it relevant to Electrical Engineering domain?
2. Is it safe and respectful (not harmful, abusive, or offensive)?
3. Can it be answered by a language model with domain knowledge?
4. Is it clear and well-formed?

Respond with ONLY "PASS" or "FAIL" followed by a brief reason (one sentence).
Format: PASS: reason OR FAIL: reason"""

        response = self._call_llm(prompt, temperature=0.3)

        if response.startswith("PASS"):
            return {"passed": True, "reason": response.split(':', 1)[1].strip()}
        else:
            return {"passed": False, "reason": response.split(':', 1)[1].strip() if ':' in response else "Failed evaluation"}

    def generate_response(self, question: str, context: str, creative: bool = False) -> str:
        """
        Stage 3: Generate a response to a question.

        Args:
            question: The question to answer
            context: Context to use for answering
            creative: If True, use creative persona; if False, use precise persona

        Returns:
            Generated response
        """
        if creative:
            persona = "You are a creative and engaging expert in Electrical Engineering. Provide detailed, insightful answers that include examples and analogies to help understanding."
        else:
            persona = "You are a precise and technical expert in Electrical Engineering. Provide accurate, detailed answers with technical depth and clarity."

        prompt = f"""{persona}

Context:
{context}

Question: {question}

Please provide a comprehensive answer to the question based on the context provided. Your answer should be informative, well-structured, and demonstrate expert knowledge."""

        temperature = 0.8 if creative else 0.5
        return self._call_llm(prompt, temperature=temperature)

    def evaluate_instruction_response_pair(self, question: str, answer: str) -> dict:
        """
        Stage 4: Evaluate the quality of a question-answer pair.

        Args:
            question: The question
            answer: The generated answer

        Returns:
            Dictionary with 'score' (1-3) and 'explanation' (str)
        """
        prompt = f"""Please act as an impartial judge and evaluate the quality of the answer provided by an AI assistant
to the questions displayed below. Evaluate whether or not the answer is a good example of how AI
Assistant should respond to the user's instruction. Please assign a score using the following 3-point
scale:

1: It means the answer is incorrect, irrelevant, unsafe or provides incomplete and garbage information.
For instance, the answer may be factually wrong, off-topic, or filled with irrelevant content that
doesn't address the user's question or it could be incomplete and hanging. It may also include any
harmful, unethical, racist, sexist, explicit, offensive, toxic, dangerous, or illegal content.

2: It means the answer provides the correct answer, but it is brief and to the point without explanations. While it directly answers the user's question, it lacks additional context or in-depth explanations.

3: It means the answer is a perfect answer from an AI Assistant. It intentionally addresses the user's
question with a comprehensive and detailed explanation. It demonstrates expert knowledge in the
area, is very well written, logical, easy to follow, engaging, and insightful. And the answer is safe and
does not include any harmful content.

Question: {question}

Answer: {answer}

Begin your evaluation by providing a short explanation. Be as objective as possible. After providing
your explanation, you must rate the answer on a scale of 1 to 3 as mentioned above.

Format your response as:
Explanation: [your explanation]
Score: [1, 2, or 3]"""

        response = self._call_llm(prompt, temperature=0.3)

        # Parse score and explanation
        explanation = ""
        score = 2  # Default score

        lines = response.strip().split('\n')
        for line in lines:
            if line.startswith("Explanation:"):
                explanation = line.split(':', 1)[1].strip()
            elif line.startswith("Score:"):
                score_text = line.split(':', 1)[1].strip()
                try:
                    score = int(score_text[0])  # Get first digit
                except:
                    score = 2

        return {"score": score, "explanation": explanation}

    def synthesize_dataset(
        self,
        contexts: list[dict],
        questions_per_context: int = 5,
        creative_responses: bool = False,
        min_quality_score: int = 2
    ) -> list[dict]:
        """
        Complete pipeline: Generate synthetic instruction-response pairs from contexts.

        Args:
            contexts: List of context dictionaries from process_pdf
            questions_per_context: Number of questions to generate per context
            creative_responses: Use creative persona for responses
            min_quality_score: Minimum quality score (1-3) to include in dataset

        Returns:
            List of high-quality instruction-response pairs
        """
        dataset = []

        print(f"\nStarting synthetic dataset generation...")
        print(f"Total contexts to process: {len(contexts)}")

        for ctx_idx, context_dict in enumerate(contexts):
            print(f"\nProcessing context {ctx_idx + 1}/{len(contexts)} (cluster {context_dict['cluster_id']})")

            # Combine chunks in context
            context_text = "\n\n".join([chunk['content'] for chunk in context_dict['chunks']])

            # Stage 1: Generate instructions
            print(f"  - Generating {questions_per_context} questions...")
            questions = self.generate_instructions(context_text, questions_per_context)
            print(f"  - Generated {len(questions)} questions")

            # Stage 2: Evaluate instructions
            print(f"  - Evaluating questions...")
            valid_questions = []
            for q in questions:
                eval_result = self.evaluate_instruction(q)
                if eval_result['passed']:
                    valid_questions.append(q)
            print(f"  - {len(valid_questions)} questions passed evaluation")

            # Stage 3 & 4: Generate and evaluate responses
            print(f"  - Generating and evaluating responses...")
            for question in valid_questions:
                # Generate response
                answer = self.generate_response(question, context_text, creative_responses)

                # Evaluate pair
                evaluation = self.evaluate_instruction_response_pair(question, answer)

                # Only include if meets quality threshold
                if evaluation['score'] >= min_quality_score:
                    dataset.append({
                        'instruction': question,
                        'response': answer,
                        'context': context_text[:500] + "..." if len(context_text) > 500 else context_text,
                        'cluster_id': context_dict['cluster_id'],
                        'quality_score': evaluation['score'],
                        'quality_explanation': evaluation['explanation']
                    })

            print(f"  - Added {len([d for d in dataset if d['cluster_id'] == context_dict['cluster_id']])} high-quality pairs from this context")

        print(f"\nDataset generation complete!")
        print(f"Total high-quality pairs generated: {len(dataset)}")

        return dataset

    def process_pdf(
        self,
        pdf_path: pathlib.Path,
        save_chunks: bool = False,
        n_clusters: Optional[int] = None,
        method: Literal["kmeans", "none"] = "kmeans",
        chunks_per_cluster: int = 5,
        diversity_weight: float = 0.3
    ) -> dict:
        """
        Process a single PDF file: extract, chunk, and generate clusters.

        Args:
            pdf_path: Path to the PDF file
            save_chunks: Whether to save raw chunks for debugging
            n_clusters: Number of clusters (None for auto-determination with kmeans)
            method: Clustering method ('kmeans' or 'none')
            chunks_per_cluster: Number of representative chunks per cluster
            diversity_weight: Balance between centrality and diversity

        Returns:
            Dictionary containing processing results
        """
        print(f"\nProcessing: {pdf_path.name}")
        print("  - Extracting and chunking PDF...")
        chunks = split_pdf(
            str(pdf_path),
            self.header_detector,
            self.remove_headers_footers_func,
            chunking=self.chunking,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        print(f"  - Extracted {len(chunks)} chunks")

        if save_chunks and chunks:
            self._save_raw_chunks(pdf_path, chunks)

        # Generate clustered contexts
        print("  - Generating contexts...")
        contexts = generate_clustered_contexts(
            chunks,
            n_clusters=n_clusters,
            method=method,
            chunks_per_cluster=chunks_per_cluster,
            diversity_weight=diversity_weight
        )
        print(f"  - Generated {len(contexts)} contexts")

        # Compile results
        result = {
            "pdf_name": pdf_path.name,
            "pdf_path": str(pdf_path),
            "processed_at": datetime.now().isoformat(),
            "total_chunks": len(chunks),
            "num_clusters": len(contexts),
            "contexts": contexts,
            "raw_chunks_count": len(chunks)
        }

        return result

    def process_all_pdfs(
        self,
        n_clusters: Optional[int] = None,
        method: Literal["kmeans", "none"] = "kmeans",
        chunks_per_cluster: int = 5,
        diversity_weight: float = 0.3,
        save_individual: bool = True,
        save_combined: bool = True,
        save_chunks: bool = False
    ) -> list[dict]:
        """
        Process all PDFs in the documents directory and generate clustered contexts.

        Args:
            n_clusters: Number of clusters (None for auto-determination with kmeans)
            method: Clustering method ('kmeans' or 'none'). 'none' returns original chunks without clustering
            chunks_per_cluster: Number of representative chunks per cluster (ignored if method='none')
            diversity_weight: Balance between centrality and diversity (ignored if method='none')
            save_individual: Save results for each PDF separately
            save_combined: Save combined results for all PDFs
            save_chunks: Save raw chunks for debugging

        Returns:
            List of processing results for each PDF
        """
        pdf_files = self.get_pdf_files()

        if not pdf_files:
            print(f"No PDF files found in {self.documents_dir}")
            return []

        print(f"\nFound {len(pdf_files)} PDF file(s) to process:")
        for pdf in pdf_files:
            print(f"  - {pdf.name}")

        all_results = []

        # Process each PDF
        for pdf_path in pdf_files:
            try:
                result = self.process_pdf(
                    pdf_path,
                    save_chunks=save_chunks,
                    n_clusters=n_clusters,
                    method=method,
                    chunks_per_cluster=chunks_per_cluster,
                    diversity_weight=diversity_weight
                )
                all_results.append(result)

                # Save individual PDF results if requested
                if save_individual:
                    self._save_pdf_result(result)

            except Exception as e:
                print(f"  ERROR processing {pdf_path.name}: {str(e)}")
                # Continue with other PDFs even if one fails
                continue

        # Save combined results if requested
        if save_combined and all_results:
            self._save_combined_results(all_results)

        return all_results

    def _save_raw_chunks(self, pdf_path: pathlib.Path, chunks: list) -> None:
        """Save raw chunks for debugging visualization."""
        pdf_name = pdf_path.stem
        output_file = self.outputs_dir / f"{pdf_name}_raw_chunks.json"

        # Format chunks for easy visualization
        formatted_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_data = {
                "chunk_id": i,
                "content": chunk.page_content,
                "metadata": chunk.metadata,
                "char_count": len(chunk.page_content),
                "word_count": len(chunk.page_content.split())
            }
            formatted_chunks.append(chunk_data)

        output = {
            "pdf_name": pdf_path.name,
            "total_chunks": len(formatted_chunks),
            "chunks": formatted_chunks
        }

        output_file.write_text(json.dumps(output, indent=2, ensure_ascii=False))
        print(f"  - Raw chunks saved to: {output_file}")

    def _save_pdf_result(self, result: dict) -> None:
        """Save processing result for a single PDF."""
        pdf_name = pathlib.Path(result["pdf_name"]).stem
        output_file = self.outputs_dir / f"{pdf_name}_clustered_contexts.json"

        # Create a simplified version for debugging
        debug_output = {
            "pdf_name": result["pdf_name"],
            "processed_at": result["processed_at"],
            "total_chunks": result["total_chunks"],
            "num_clusters": result["num_clusters"],
            "clusters": [
                {
                    "cluster_id": ctx["cluster_id"],
                    "size": ctx["size"],
                    "coherence": ctx["coherence"],
                    "representative_chunks": [
                        {
                            "content": chunk["content"],
                            "metadata": chunk["metadata"]
                        }
                        for chunk in ctx["chunks"]
                    ]
                }
                for ctx in result["contexts"]
            ]
        }

        output_file.write_text(json.dumps(debug_output, indent=2, ensure_ascii=False))
        print(f"  - Saved to: {output_file}")

    def _save_combined_results(self, all_results: list[dict]) -> None:
        """Save combined results from all PDFs."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.outputs_dir / f"all_pdfs_clustered_contexts_{timestamp}.json"

        combined_output = {
            "generated_at": datetime.now().isoformat(),
            "total_pdfs_processed": len(all_results),
            "pdfs": [
                {
                    "pdf_name": result["pdf_name"],
                    "total_chunks": result["total_chunks"],
                    "num_clusters": result["num_clusters"],
                }
                for result in all_results
            ],
            "total_clusters": sum(r["num_clusters"] for r in all_results),
            "detailed_results": all_results
        }

        output_file.write_text(json.dumps(combined_output, indent=2, ensure_ascii=False))
        print(f"\nCombined results saved to: {output_file}")

    def _save_synthetic_dataset(self, dataset: list[dict], filename: str = None) -> pathlib.Path:
        """
        Save synthetic dataset to JSON file.

        Args:
            dataset: List of instruction-response pairs
            filename: Optional custom filename (default: auto-generated with timestamp)

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"synthetic_dataset_{timestamp}.json"

        output_file = self.outputs_dir / filename

        output_data = {
            "generated_at": datetime.now().isoformat(),
            "total_samples": len(dataset),
            "quality_distribution": {
                "score_1": len([d for d in dataset if d['quality_score'] == 1]),
                "score_2": len([d for d in dataset if d['quality_score'] == 2]),
                "score_3": len([d for d in dataset if d['quality_score'] == 3])
            },
            "dataset": dataset
        }

        output_file.write_text(json.dumps(output_data, indent=2, ensure_ascii=False))
        print(f"\nSynthetic dataset saved to: {output_file}")
        print(f"Total samples: {len(dataset)}")
        print(f"Quality distribution: {output_data['quality_distribution']}")

        return output_file

    def process_and_synthesize(
        self,
        n_clusters: Optional[int] = None,
        method: Literal["kmeans", "none"] = "kmeans",
        chunks_per_cluster: int = 5,
        diversity_weight: float = 0.3,
        questions_per_context: int = 5,
        creative_responses: bool = False,
        min_quality_score: int = 2,
        save_intermediate: bool = True
    ) -> tuple[list[dict], list[dict]]:
        """
        Complete pipeline: Process PDFs and generate synthetic dataset.

        Args:
            n_clusters: Number of clusters for context generation
            method: Clustering method
            chunks_per_cluster: Chunks per cluster
            diversity_weight: Diversity weight for chunk selection
            questions_per_context: Questions to generate per context
            creative_responses: Use creative persona for responses
            min_quality_score: Minimum quality score for inclusion
            save_intermediate: Save intermediate results (contexts)

        Returns:
            Tuple of (processing_results, synthetic_dataset)
        """
        # Process all PDFs and generate contexts
        print("=" * 70)
        print("STAGE 1: PDF Processing and Context Generation")
        print("=" * 70)

        results = self.process_all_pdfs(
            n_clusters=n_clusters,
            method=method,
            chunks_per_cluster=chunks_per_cluster,
            diversity_weight=diversity_weight,
            save_individual=save_intermediate,
            save_combined=save_intermediate,
            save_chunks=False
        )

        if not results:
            print("No PDFs processed. Exiting.")
            return [], []

        # Collect all contexts from all PDFs
        all_contexts = []
        for result in results:
            all_contexts.extend(result['contexts'])

        print(f"\nTotal contexts collected: {len(all_contexts)}")

        # Generate synthetic dataset
        print("\n" + "=" * 70)
        print("STAGE 2: Synthetic Dataset Generation")
        print("=" * 70)

        dataset = self.synthesize_dataset(
            all_contexts,
            questions_per_context=questions_per_context,
            creative_responses=creative_responses,
            min_quality_score=min_quality_score
        )

        # Save synthetic dataset
        print("\n" + "=" * 70)
        print("STAGE 3: Saving Synthetic Dataset")
        print("=" * 70)

        self._save_synthetic_dataset(dataset)

        return results, dataset


# Future enhancements:
# - Add support for different document types (not just PDFs)
# - Add configurable chunking strategies
# - Add support for custom clustering parameters per document
# - Add multi-processing for faster PDF processing and LLM calls
# - Add support for different LLM providers (e.g., Anthropic, local models)
# - Add few-shot learning with custom example questions
# - Add support for multi-turn conversations in synthetic data


def main():
    """
    Main function demonstrating the complete synthetic dataset generation pipeline.

    This will:
    1. Process PDFs from the documents directory
    2. Generate clustered contexts
    3. Generate synthetic instruction-response pairs using LLM
    4. Evaluate and filter high-quality pairs
    5. Save the final dataset to JSON
    """
    print("=" * 70)
    print("Synthetic Dataset Generator for Electrical Engineering")
    print("=" * 70)

    # Initialize synthesizer
    synthesizer = DatasetSynthesizer(
        documents_dir="src/extraction/documents",
        outputs_dir="src/extraction/outputs",
    )

    # Run complete pipeline
    try:
        results, dataset = synthesizer.process_and_synthesize(
            n_clusters=None,  # Auto-determine optimal clusters
            method="kmeans",  # Use K-means clustering
            chunks_per_cluster=5,  # 5 representative chunks per cluster
            diversity_weight=0.3,  # Balance centrality vs diversity
            questions_per_context=5,  # Generate 5 questions per context
            creative_responses=False,  # Use precise/technical persona
            min_quality_score=2,  # Only include score 2+ pairs
            save_intermediate=True  # Save intermediate results
        )

        # Print final summary
        print("\n" + "=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)
        print(f"PDFs processed: {len(results)}")
        print(f"Total contexts generated: {sum(r['num_clusters'] for r in results)}")
        print(f"High-quality instruction-response pairs: {len(dataset)}")

        if dataset:
            avg_score = sum(d['quality_score'] for d in dataset) / len(dataset)
            print(f"Average quality score: {avg_score:.2f}")

        print("\nOutputs saved to: src/extraction/outputs/")
        print("  - synthetic_dataset_*.json: Final synthetic dataset")
        print("  - *_clustered_contexts.json: Intermediate contexts")
        print("=" * 70)

    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    main()
