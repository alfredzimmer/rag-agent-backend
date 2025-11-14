"""
Example script demonstrating how to use the DatasetSynthesizer.

This script shows:
1. Basic usage with default parameters
2. Advanced usage with custom parameters
3. How to load and inspect the generated dataset
"""

from synthesize_dataset import DatasetSynthesizer
import json


def basic_example():
    """
    Basic usage: Generate synthetic dataset with default parameters.
    """
    print("=" * 70)
    print("BASIC EXAMPLE: Default Parameters")
    print("=" * 70)

    # Initialize synthesizer
    synthesizer = DatasetSynthesizer(
        documents_dir="src/extraction/documents",
        outputs_dir="src/extraction/outputs"
    )

    # Run the complete pipeline
    results, dataset = synthesizer.process_and_synthesize()

    print(f"\nGenerated {len(dataset)} high-quality instruction-response pairs")
    return dataset


def advanced_example():
    """
    Advanced usage: Customize parameters for specific needs.
    """
    print("\n" + "=" * 70)
    print("ADVANCED EXAMPLE: Custom Parameters")
    print("=" * 70)

    # Initialize synthesizer with custom chunking
    synthesizer = DatasetSynthesizer(
        documents_dir="src/extraction/documents",
        outputs_dir="src/extraction/outputs",
        chunking=True,  # Enable recursive chunking
        chunk_size=512,
        chunk_overlap=50
    )

    # Run with custom parameters
    results, dataset = synthesizer.process_and_synthesize(
        n_clusters=10,  # Fixed number of clusters
        method="kmeans",
        chunks_per_cluster=3,  # Fewer chunks per cluster
        diversity_weight=0.5,  # More diversity
        questions_per_context=3,  # Fewer questions per context
        creative_responses=True,  # Use creative persona
        min_quality_score=3,  # Only highest quality
        save_intermediate=True
    )

    print(f"\nGenerated {len(dataset)} highest-quality pairs (score 3 only)")
    return dataset


def inspect_dataset(dataset_file: str):
    """
    Load and inspect a generated synthetic dataset.

    Args:
        dataset_file: Path to the JSON dataset file
    """
    print("\n" + "=" * 70)
    print("INSPECTING DATASET")
    print("=" * 70)

    with open(dataset_file, 'r') as f:
        data = json.load(f)

    print(f"Generated at: {data['generated_at']}")
    print(f"Total samples: {data['total_samples']}")
    print(f"Quality distribution: {data['quality_distribution']}")

    # Show first sample
    if data['dataset']:
        print("\nFirst sample:")
        sample = data['dataset'][0]
        print(f"  Instruction: {sample['instruction'][:100]}...")
        print(f"  Response: {sample['response'][:100]}...")
        print(f"  Quality Score: {sample['quality_score']}")


def generate_from_specific_contexts():
    """
    Generate dataset from already-processed contexts (skip PDF processing).
    """
    print("\n" + "=" * 70)
    print("EXAMPLE: Using Pre-Processed Contexts")
    print("=" * 70)

    synthesizer = DatasetSynthesizer(
        documents_dir="src/extraction/documents",
        outputs_dir="src/extraction/outputs"
    )

    # Load existing contexts from a previous run
    # (In practice, you would load from a saved file)
    # For this example, we'll process PDFs first
    results = synthesizer.process_all_pdfs(
        n_clusters=5,
        method="kmeans",
        save_individual=False,
        save_combined=False
    )

    # Extract contexts
    all_contexts = []
    for result in results:
        all_contexts.extend(result['contexts'])

    # Generate dataset from these contexts
    dataset = synthesizer.synthesize_dataset(
        all_contexts,
        questions_per_context=3,
        creative_responses=False,
        min_quality_score=2
    )

    # Save manually with custom filename
    synthesizer._save_synthetic_dataset(dataset, "custom_dataset.json")

    print(f"\nGenerated {len(dataset)} pairs from {len(all_contexts)} contexts")


if __name__ == "__main__":
    # Run basic example
    dataset = basic_example()

    # Optionally run advanced example (uncomment to use)
    # dataset = advanced_example()

    # Optionally inspect the generated dataset
    # inspect_dataset("src/extraction/outputs/synthetic_dataset_TIMESTAMP.json")

    # Optionally generate from specific contexts
    # generate_from_specific_contexts()
