"""
Module for synthesizing datasets from PDF documents.

This module provides functionality to:
1. Ingest PDFs from the documents folder
2. Process and chunk the PDFs
3. Generate clustered contexts for synthetic data generation
4. Generate synthetic instruction-response pairs using LLM (4-stage process):
   - Stage 1: Instruction generation from contexts
   - Stage 2: Instruction quality evaluation (optional)
   - Stage 3: Response generation with dual personas
   - Stage 4: Instruction-response pair quality evaluation
5. Save high-quality synthetic datasets to JSON files

Features:
- Multiprocessing support for parallel context processing across worker processes
- Async/await for concurrent LLM API calls within each worker
- Configurable instruction evaluation (can be disabled)
- Each worker processes contexts independently for better scalability
- Checkpoint system for resuming interrupted dataset generation runs
- Intermediate results saved separately (clustered contexts, raw chunks, combined results)

Directory Structure:
- documents/: Input PDF files
- outputs/: Final synthetic datasets
- intermediate/: All intermediate results including:
  - Raw chunks from PDF processing
  - Clustered contexts
  - Combined results from all PDFs
  - Checkpoints during dataset generation (every N samples, default 100)

Checkpoint Usage:
To resume from an interrupted run, use the session_id from the checkpoint:
  synthesizer.process_and_synthesize(resume_from_checkpoint="20250114_123456")

Complete implementation following the methodology described in the research paper.
"""

import pathlib
import json
from typing import Optional, Literal
from datetime import datetime
import os
import asyncio
import multiprocessing as mp
from functools import partial
from dotenv import load_dotenv
from pdf_chunker import split_pdf
from generate_context import generate_clustered_contexts
from IEEE_utils import IEEEHeaderDetector, IEEE_remove_headers_footers
from llm_worker import LLMWorker

# Load environment variables
load_dotenv()


def _process_context_chunk(
    context_chunk: list[dict],
    instruction_model: str,
    response_model: str,
    instruction_eval_model: str,
    response_eval_model: str,
    evaluate_instructions: bool,
    questions_per_context: int,
    creative_responses: bool,
    min_quality_score: int
) -> list[dict]:
    """
    Process a chunk of contexts in a separate process.
    This is a module-level function so it can be pickled for multiprocessing.
    """
    # Create worker instance
    worker = LLMWorker(
        instruction_model=instruction_model,
        response_model=response_model,
        instruction_eval_model=instruction_eval_model,
        response_eval_model=response_eval_model,
        evaluate_instructions=evaluate_instructions
    )

    # Process all contexts in this chunk asynchronously
    async def process_chunk():
        results = []
        for ctx_idx, context_dict in enumerate(context_chunk):
            print(f"  Worker processing context {ctx_idx + 1}/{len(context_chunk)} (cluster {context_dict['cluster_id']})")
            dataset_entries = await worker.process_context(
                context_dict,
                questions_per_context,
                creative_responses,
                min_quality_score
            )
            results.extend(dataset_entries)
            print(f"  - Generated {len(dataset_entries)} pairs for cluster {context_dict['cluster_id']}")
        return results

    # Run async event loop in this process
    return asyncio.run(process_chunk())


class DatasetSynthesizer:
    """
    Handles PDF ingestion and clustered context generation for synthetic dataset creation.
    """

    def __init__(
        self,
        documents_dir: str = "documents",
        outputs_dir: str = "outputs",
        intermediate_dir: str = "intermediate",
        header_detector=IEEEHeaderDetector,
        remove_headers_footers_func=IEEE_remove_headers_footers,
        chunking: bool = False,
        chunk_size: int = 512,
        chunk_overlap: int = 0,
        instruction_model: str = "gpt-4o-mini",
        response_model: str = "gpt-4o-mini",
        instruction_eval_model: str = "gpt-4o-mini",
        response_eval_model: str = "gpt-4o-mini",
        evaluate_instructions: bool = True,
        num_workers: int = 4,
        checkpoint_interval: int = 100,
    ):
        """
        Initialize the DatasetSynthesizer.

        Args:
            documents_dir: Path to directory containing PDF documents
            outputs_dir: Path to directory for saving outputs
            intermediate_dir: Path to directory for saving intermediate results
            header_detector: Function/class to detect headers in PDFs
            remove_headers_footers_func: Function to remove headers/footers
            chunking: Whether to apply recursive chunking to markdown splits
            chunk_size: Size of text chunks for splitting (if chunking is True)
            chunk_overlap: Overlap between consecutive chunks (if chunking is True)
            instruction_model: Model for generating instructions
            response_model: Model for generating responses
            instruction_eval_model: Model for evaluating instructions
            response_eval_model: Model for evaluating responses
            evaluate_instructions: Whether to evaluate instructions (Stage 2)
            num_workers: Number of worker processes for parallel synthesis
            checkpoint_interval: Number of samples to generate before saving checkpoint
        """
        self.documents_dir = pathlib.Path(documents_dir)
        self.outputs_dir = pathlib.Path(outputs_dir)
        self.intermediate_dir = pathlib.Path(intermediate_dir)
        self.header_detector = header_detector
        self.remove_headers_footers_func = remove_headers_footers_func
        self.chunking = chunking
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.instruction_model = instruction_model
        self.response_model = response_model
        self.instruction_eval_model = instruction_eval_model
        self.response_eval_model = response_eval_model
        self.evaluate_instructions = evaluate_instructions
        self.num_workers = num_workers
        self.checkpoint_interval = checkpoint_interval
        # Create output and intermediate directories if they don't exist
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.intermediate_dir.mkdir(parents=True, exist_ok=True)

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


    def synthesize_dataset(
        self,
        contexts: list[dict],
        questions_per_context: int = 5,
        creative_responses: bool = False,
        min_quality_score: int = 2,
        enable_checkpoints: bool = True,
        resume_from_checkpoint: str | None = None
    ) -> list[dict]:
        """
        Complete pipeline: Generate synthetic instruction-response pairs from contexts.
        Uses multiprocessing to parallelize context processing across multiple workers.
        Supports checkpointing to resume interrupted runs.

        Args:
            contexts: List of context dictionaries from process_pdf
            questions_per_context: Number of questions to generate per context
            creative_responses: Use creative persona for responses
            min_quality_score: Minimum quality score (1-3) to include in dataset
            enable_checkpoints: Whether to save checkpoints during generation
            resume_from_checkpoint: Optional session_id to resume from

        Returns:
            List of high-quality instruction-response pairs
        """
        # Initialize or resume session
        if resume_from_checkpoint:
            session_id = resume_from_checkpoint
            dataset, checkpoint_num = self._load_checkpoint(session_id)
            print(f"\nResuming from checkpoint {checkpoint_num}")
            print(f"Already processed: {len(dataset)} samples")
        else:
            session_id = self._get_session_id()
            dataset = []
            checkpoint_num = 0

        print(f"\nStarting synthetic dataset generation...")
        print(f"Session ID: {session_id}")
        print(f"Total contexts to process: {len(contexts)}")
        print(f"Using {self.num_workers} worker processes")
        print(f"Checkpoint interval: {self.checkpoint_interval} samples")

        # Split contexts into chunks for each worker
        chunk_size = max(1, len(contexts) // self.num_workers)
        context_chunks = [
            contexts[i:i + chunk_size]
            for i in range(0, len(contexts), chunk_size)
        ]

        print(f"Split into {len(context_chunks)} chunks")

        # Create a partial function with the instance attributes bound
        # This allows the module-level function to be pickled for multiprocessing
        worker_func = partial(
            _process_context_chunk,
            instruction_model=self.instruction_model,
            response_model=self.response_model,
            instruction_eval_model=self.instruction_eval_model,
            response_eval_model=self.response_eval_model,
            evaluate_instructions=self.evaluate_instructions,
            questions_per_context=questions_per_context,
            creative_responses=creative_responses,
            min_quality_score=min_quality_score
        )

        # Process chunks with checkpointing
        try:
            with mp.Pool(processes=self.num_workers) as pool:
                results_chunks = pool.map(worker_func, context_chunks)

            # Flatten results and save checkpoints
            for chunk_idx, chunk_results in enumerate(results_chunks):
                dataset.extend(chunk_results)

                # Save checkpoint if interval reached
                if enable_checkpoints and len(dataset) >= (checkpoint_num + 1) * self.checkpoint_interval:
                    checkpoint_num += 1
                    self._save_checkpoint(dataset, checkpoint_num, session_id)

        except KeyboardInterrupt:
            print("\n\nInterrupted! Saving checkpoint...")
            if enable_checkpoints:
                self._save_checkpoint(dataset, checkpoint_num + 1, session_id)
            print(f"Progress saved. Resume with session_id: {session_id}")
            raise

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
                    self._save_clustered_contexts_result(result)

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
        output_file = self.intermediate_dir / f"{pdf_name}_raw_chunks.json"

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

    def _save_clustered_contexts_result(self, result: dict) -> None:
        """Save processing result for a single PDF."""
        pdf_name = pathlib.Path(result["pdf_name"]).stem
        output_file = self.intermediate_dir / f"{pdf_name}_clustered_contexts.json"

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
        output_file = self.intermediate_dir / f"all_pdfs_clustered_contexts_{timestamp}.json"

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

    def _save_checkpoint(self, dataset: list[dict], checkpoint_num: int, session_id: str) -> pathlib.Path:
        """
        Save a checkpoint of the synthetic dataset.

        Args:
            dataset: List of instruction-response pairs so far
            checkpoint_num: Checkpoint number
            session_id: Unique session identifier

        Returns:
            Path to saved checkpoint file
        """
        checkpoint_file = self.intermediate_dir / f"checkpoint_{session_id}_{checkpoint_num}.json"

        checkpoint_data = {
            "session_id": session_id,
            "checkpoint_num": checkpoint_num,
            "saved_at": datetime.now().isoformat(),
            "total_samples": len(dataset),
            "dataset": dataset
        }

        checkpoint_file.write_text(json.dumps(checkpoint_data, indent=2, ensure_ascii=False))
        print(f"  Checkpoint saved: {checkpoint_file.name} ({len(dataset)} samples)")

        return checkpoint_file

    def _load_checkpoint(self, session_id: str) -> tuple[list[dict], int]:
        """
        Load the latest checkpoint for a session.

        Args:
            session_id: Unique session identifier

        Returns:
            Tuple of (dataset, checkpoint_num)
        """
        # Find all checkpoints for this session
        checkpoint_pattern = f"checkpoint_{session_id}_*.json"
        checkpoint_files = list(self.intermediate_dir.glob(checkpoint_pattern))

        if not checkpoint_files:
            return [], 0

        # Get the latest checkpoint
        latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)

        print(f"Loading checkpoint: {latest_checkpoint.name}")
        checkpoint_data = json.loads(latest_checkpoint.read_text())

        return checkpoint_data["dataset"], checkpoint_data["checkpoint_num"]

    def _get_session_id(self) -> str:
        """Generate a unique session ID for checkpointing."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _save_synthetic_dataset(self, dataset: list[dict], filename: str | None = None) -> pathlib.Path:
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
        save_intermediate: bool = True,
        enable_checkpoints: bool = True,
        resume_from_checkpoint: str | None = None
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
            enable_checkpoints: Whether to save checkpoints during synthesis
            resume_from_checkpoint: Optional session_id to resume from

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
            min_quality_score=min_quality_score,
            enable_checkpoints=enable_checkpoints,
            resume_from_checkpoint=resume_from_checkpoint
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
# - Add support for different LLM providers (e.g., Anthropic, local models)
# - Add few-shot learning with custom example questions
# - Add support for multi-turn conversations in synthetic data
# - Add rate limiting and retry logic for API calls


def main():
    """
    Main function demonstrating the complete synthetic dataset generation pipeline.

    This will:
    1. Process PDFs from the documents directory
    2. Generate clustered contexts
    3. Generate synthetic instruction-response pairs using LLM (with async and multiprocessing)
    4. Evaluate and filter high-quality pairs
    5. Save the final dataset to JSON
    """
    print("=" * 70)
    print("Synthetic Dataset Generator for Electrical Engineering")
    print("=" * 70)

    # Initialize synthesizer with multiprocessing support
    synthesizer = DatasetSynthesizer(
        documents_dir="src/extraction/documents",
        outputs_dir="src/extraction/outputs",
        intermediate_dir="src/extraction/intermediate",
        chunking=False,
        evaluate_instructions=False,  # Enable instruction evaluation
        num_workers=4,  # Number of parallel worker processes
        checkpoint_interval=100  # Save checkpoint every 100 samples
    )

    # Run complete pipeline
    try:
        results, dataset = synthesizer.process_and_synthesize(
            n_clusters=None,  # Auto-determine optimal clusters
            method="none",  # No clustering
            chunks_per_cluster=5,  # 5 representative chunks per cluster
            diversity_weight=0.3,  # Balance centrality vs diversity
            questions_per_context=5,  # Generate 5 questions per context
            creative_responses=False,  # Use precise/technical persona
            min_quality_score=2,  # Only include score 2+ pairs
            save_intermediate=True,  # Save intermediate results
            enable_checkpoints=True    
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

        print("\nOutputs saved to:")
        print("  - src/extraction/outputs/synthetic_dataset_*.json: Final synthetic dataset")
        print("  - src/extraction/intermediate/: All intermediate results and checkpoints")
        print("=" * 70)

    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    main()
