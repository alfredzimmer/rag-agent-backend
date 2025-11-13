"""
Example usage of the clustered context generation for synthetic data generation.
"""

from langchain_core.documents import Document
from generate_context import generate_clustered_contexts


def example_basic_usage():
    """Basic usage with auto-determined clusters."""
    # Sample documents
    chunks = [
        Document(page_content="Machine learning models require training data.", metadata={"source": "doc1", "page": 1}),
        Document(page_content="Neural networks learn patterns from examples.", metadata={"source": "doc1", "page": 2}),
        Document(page_content="Python is a popular programming language.", metadata={"source": "doc2", "page": 1}),
        Document(page_content="JavaScript runs in web browsers.", metadata={"source": "doc2", "page": 2}),
        Document(page_content="Deep learning uses multiple layers.", metadata={"source": "doc3", "page": 1}),
        Document(page_content="Variables store data in programs.", metadata={"source": "doc4", "page": 1}),
    ]

    # Generate contexts with automatic cluster determination
    contexts = generate_clustered_contexts(chunks)

    print("Generated Contexts:")
    for ctx in contexts:
        print(f"\nCluster {ctx['cluster_id']}:")
        print(f"  Size: {ctx['size']} chunks")
        print(f"  Coherence: {ctx['coherence']:.3f}")
        print(f"  Representative chunks:")
        for chunk in ctx['chunks']:
            print(f"    - {chunk['content'][:50]}...")


def example_fixed_clusters():
    """Usage with fixed number of clusters."""
    chunks = [
        Document(page_content=f"Document about topic {i % 3}", metadata={"id": i})
        for i in range(20)
    ]

    # Force 3 clusters
    contexts = generate_clustered_contexts(
        chunks,
        n_clusters=3,
        chunks_per_cluster=3
    )

    print(f"\nGenerated {len(contexts)} contexts with fixed cluster count")


def example_dbscan_clustering():
    """Usage with DBSCAN for density-based clustering."""
    chunks = [
        Document(page_content=f"Sample text {i}", metadata={"id": i})
        for i in range(15)
    ]

    # Use DBSCAN to find natural clusters
    contexts = generate_clustered_contexts(
        chunks,
        method="dbscan",
        eps=0.3,  # Adjust based on your data
        min_cluster_size=3
    )

    print(f"\nDBSCAN found {len(contexts)} natural clusters")


def example_synthetic_query_generation():
    """Example of using clusters for synthetic query generation."""
    chunks = [
        Document(page_content="The Python programming language is widely used for data science.", metadata={"topic": "python"}),
        Document(page_content="Python has excellent libraries like NumPy and pandas.", metadata={"topic": "python"}),
        Document(page_content="JavaScript is the language of the web browser.", metadata={"topic": "javascript"}),
        Document(page_content="React and Vue are popular JavaScript frameworks.", metadata={"topic": "javascript"}),
        Document(page_content="Machine learning algorithms can predict outcomes.", metadata={"topic": "ml"}),
        Document(page_content="Neural networks mimic the human brain structure.", metadata={"topic": "ml"}),
    ]

    # Generate contexts with high diversity for varied synthetic queries
    contexts = generate_clustered_contexts(
        chunks,
        n_clusters=3,
        chunks_per_cluster=2,
        diversity_weight=0.5  # Higher diversity for more varied context
    )

    print("\nContexts for Synthetic Query Generation:")
    for ctx in contexts:
        print(f"\nCluster {ctx['cluster_id']} (coherence: {ctx['coherence']:.3f}):")
        combined_context = " ".join([c['content'] for c in ctx['chunks']])
        print(f"  Combined context: {combined_context[:100]}...")
        print(f"  Could generate query like: 'What does this context tell us about...'")


if __name__ == "__main__":
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    example_basic_usage()

    print("\n" + "=" * 60)
    print("Example 2: Fixed Clusters")
    print("=" * 60)
    example_fixed_clusters()

    print("\n" + "=" * 60)
    print("Example 3: DBSCAN Clustering")
    print("=" * 60)
    example_dbscan_clustering()

    print("\n" + "=" * 60)
    print("Example 4: Synthetic Query Generation")
    print("=" * 60)
    example_synthetic_query_generation()
