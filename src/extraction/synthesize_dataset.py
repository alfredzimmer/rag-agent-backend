"""
Module for synthesizing datasets from PDF documents.

This module provides functionality to:
1. Ingest PDFs from the documents folder
2. Process and chunk the PDFs
3. Generate clustered contexts for synthetic data generation
4. Save outputs for debugging and analysis

Currently partially implemented for testing purposes.
"""

import pathlib
import json
from typing import Optional, Literal
from datetime import datetime

from pdf_chunker import split_pdf
from generate_context import generate_clustered_contexts
from IEEE_utils import IEEEHeaderDetector, IEEE_remove_headers_footers


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
    ):
        """
        Initialize the DatasetSynthesizer.

        Args:
            documents_dir: Path to directory containing PDF documents
            outputs_dir: Path to directory for saving outputs
            header_detector: Function/class to detect headers in PDFs
            remove_headers_footers_func: Function to remove headers/footers
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between consecutive chunks
        """
        self.documents_dir = pathlib.Path(documents_dir)
        self.outputs_dir = pathlib.Path(outputs_dir)
        self.header_detector = header_detector
        self.remove_headers_footers_func = remove_headers_footers_func

        # Create output directory if it doesn't exist
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

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

    def process_pdf(self, pdf_path: pathlib.Path, save_chunks: bool = False) -> dict:
        """
        Process a single PDF file: extract, chunk, and generate clusters.

        Args:
            pdf_path: Path to the PDF file
            save_chunks: Whether to save raw chunks for debugging

        Returns:
            Dictionary containing processing results
        """
        print(f"\nProcessing: {pdf_path.name}")

        # Step 1: Split PDF into chunks
        print("  - Extracting and chunking PDF...")
        chunks = split_pdf(
            str(pdf_path),
            self.header_detector,
            self.remove_headers_footers_func,
        )
        print(f"  - Extracted {len(chunks)} chunks")

        # Save raw chunks if requested for debugging
        if save_chunks and chunks:
            self._save_raw_chunks(pdf_path, chunks)

        # Step 2: Generate clustered contexts
        # TODO: Add more clustering methods and parameters in future
        print("  - Generating clustered contexts...")
        contexts = generate_clustered_contexts(
            chunks,
            n_clusters=None,  # Auto-determine optimal clusters
            method="kmeans",
            chunks_per_cluster=5,
            diversity_weight=0.3
        )
        print(f"  - Generated {len(contexts)} clustered contexts")

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
        method: Literal["kmeans", "dbscan"] = "kmeans",
        chunks_per_cluster: int = 5,
        diversity_weight: float = 0.3,
        save_individual: bool = True,
        save_combined: bool = True,
        save_chunks: bool = False
    ) -> list[dict]:
        """
        Process all PDFs in the documents directory and generate clustered contexts.

        Args:
            n_clusters: Number of clusters (None for auto-determination)
            method: Clustering method ('kmeans' or 'dbscan')
            chunks_per_cluster: Number of representative chunks per cluster
            diversity_weight: Balance between centrality and diversity
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
                result = self.process_pdf(pdf_path, save_chunks=save_chunks)
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


# TODO: Future enhancements
# - Add synthetic query generation from clustered contexts
# - Add quality metrics for generated contexts
# - Add support for different document types (not just PDFs)
# - Add configurable chunking strategies
# - Add support for custom clustering parameters per document
# - Add multi-processing for faster PDF processing


def main():
    """
    Main function for testing the synthesize_dataset module.
    Includes debug output for split chunks visualization.
    """
    print("=" * 70)
    print("Dataset Synthesizer - Testing Mode (with Debug Output)")
    print("=" * 70)

    # Initialize synthesizer with configurable chunk parameters
    synthesizer = DatasetSynthesizer(
        documents_dir="src/extraction/documents",
        outputs_dir="src/extraction/outputs",
    )

    # Process all PDFs and generate clustered contexts
    # Set save_chunks=True to output raw chunks for debugging
    results = synthesizer.process_all_pdfs(
        n_clusters=None,  # Auto-determine
        method="kmeans",
        chunks_per_cluster=5,
        diversity_weight=0.3,
        save_individual=True,
        save_combined=True,
        save_chunks=True  # Enable debug output for split chunks
    )

    # Print summary
    print("\n" + "=" * 70)
    print("Processing Summary")
    print("=" * 70)
    print(f"Total PDFs processed: {len(results)}")
    print(f"Total clusters generated: {sum(r['num_clusters'] for r in results)}")
    print(f"Total chunks processed: {sum(r['total_chunks'] for r in results)}")
    print("\nOutputs saved to: outputs/")
    print("  - *_raw_chunks.json: Split chunks for debugging/visualization")
    print("  - *_clustered_contexts.json: Clustered contexts")
    print("  - all_pdfs_*.json: Combined results")
    print("=" * 70)


if __name__ == "__main__":
    main()
