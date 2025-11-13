from typing import Optional, Literal
from langchain_core.documents import Document
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

from embeddings import embed_chunks


def _determine_optimal_clusters(embeddings: np.ndarray, max_clusters: int = 10, min_clusters: int = 2) -> int:
    """
    Determine optimal number of clusters using silhouette score.

    Args:
        embeddings: Array of embeddings
        max_clusters: Maximum number of clusters to try
        min_clusters: Minimum number of clusters

    Returns:
        Optimal number of clusters
    """
    n_samples = len(embeddings)
    max_clusters = min(max_clusters, n_samples - 1)

    if n_samples < min_clusters:
        return 1

    best_score = -1
    best_n_clusters = min_clusters

    for n_clusters in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)

        if score > best_score:
            best_score = score
            best_n_clusters = n_clusters

    return best_n_clusters


def _cluster_embeddings(
    embeddings: list[list[float]],
    n_clusters: Optional[int] = None,
    method: Literal["kmeans", "dbscan"] = "kmeans",
    eps: float = 0.3
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cluster embeddings using specified method.

    Args:
        embeddings: List of embedding vectors
        n_clusters: Number of clusters (if None, auto-determine for kmeans)
        method: Clustering method ('kmeans' or 'dbscan')
        eps: Epsilon parameter for DBSCAN

    Returns:
        Tuple of (cluster_labels, cluster_centers or None)
    """
    embeddings_array = np.array(embeddings)

    if method == "kmeans":
        if n_clusters is None:
            n_clusters = _determine_optimal_clusters(embeddings_array)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings_array)
        centers = kmeans.cluster_centers_
        return labels, centers

    elif method == "dbscan":
        dbscan = DBSCAN(eps=eps, min_samples=2, metric='cosine')
        labels = dbscan.fit_predict(embeddings_array)
        # For DBSCAN, compute centers manually
        unique_labels = set(labels)
        centers = []
        for label in unique_labels:
            if label == -1:  # Noise points
                continue
            cluster_points = embeddings_array[labels == label]
            centers.append(cluster_points.mean(axis=0))
        return labels, np.array(centers) if centers else None

    else:
        raise ValueError(f"Unknown clustering method: {method}")


def _get_representative_chunks_from_cluster(
    cluster_indices: list[int],
    embeddings: np.ndarray,
    center: np.ndarray,
    content: list[str],
    metadata: list[dict],
    n_samples: int = 5,
    diversity_weight: float = 0.3
) -> list[dict]:
    """
    Get representative chunks from a cluster balancing centrality and diversity.

    Args:
        cluster_indices: Indices of chunks in this cluster
        embeddings: All embeddings
        center: Cluster center
        content: All chunk contents
        metadata: All chunk metadata
        n_samples: Number of samples to retrieve
        diversity_weight: Weight for diversity vs centrality (0-1)

    Returns:
        List of representative chunks with metadata
    """
    if not cluster_indices:
        return []

    cluster_embeddings = embeddings[cluster_indices]
    n_samples = min(n_samples, len(cluster_indices))

    # Calculate distances to center
    distances_to_center = np.linalg.norm(cluster_embeddings - center, axis=1)

    # Select most central chunk first
    selected_indices = [cluster_indices[np.argmin(distances_to_center)]]
    selected_local_indices = [np.argmin(distances_to_center)]

    # Greedily select remaining chunks balancing centrality and diversity
    for _ in range(n_samples - 1):
        scores = []
        for i, global_idx in enumerate(cluster_indices):
            if i in selected_local_indices:
                scores.append(-np.inf)
                continue

            # Centrality score (lower distance is better)
            centrality_score = 1 / (1 + distances_to_center[i])

            # Diversity score (higher min distance to selected is better)
            if selected_indices:
                distances_to_selected = [
                    np.linalg.norm(cluster_embeddings[i] - embeddings[sel_idx])
                    for sel_idx in selected_indices
                ]
                diversity_score = min(distances_to_selected)
            else:
                diversity_score = 0

            # Combined score
            score = (1 - diversity_weight) * centrality_score + diversity_weight * diversity_score
            scores.append(score)

        best_local_idx = np.argmax(scores)
        selected_indices.append(cluster_indices[best_local_idx])
        selected_local_indices.append(best_local_idx)

    # Build result
    results = []
    for idx in selected_indices:
        results.append({
            'content': content[idx],
            'metadata': metadata[idx],
            'index': idx
        })

    return results


def generate_clustered_contexts(
    chunks: list[Document],
    n_clusters: Optional[int] = None,
    method: Literal["kmeans", "dbscan"] = "kmeans",
    chunks_per_cluster: int = 5,
    diversity_weight: float = 0.3,
    min_cluster_size: int = 2,
    eps: float = 0.3
) -> list[dict]:
    """
    Generate contexts by clustering document embeddings and selecting representative chunks.

    This creates diverse contexts for synthetic data generation by:
    1. Clustering similar document chunks together
    2. Selecting representative samples from each cluster
    3. Balancing centrality (representativeness) and diversity within clusters

    Args:
        chunks: List of Document chunks to cluster
        n_clusters: Number of clusters (None for auto-determination with kmeans)
        method: Clustering method ('kmeans' or 'dbscan')
        chunks_per_cluster: Number of chunks to select per cluster
        diversity_weight: Balance between centrality (0) and diversity (1)
        min_cluster_size: Minimum cluster size to include
        eps: Epsilon parameter for DBSCAN clustering

    Returns:
        List of contexts, each containing:
            - cluster_id: Cluster identifier
            - chunks: Representative chunks from the cluster
            - size: Total number of chunks in cluster
            - coherence: Average similarity within cluster
    """
    if not chunks:
        return []

    # Get embeddings
    content, metadata, embeddings = embed_chunks(chunks)
    embeddings_array = np.array(embeddings)

    # Cluster embeddings
    labels, centers = _cluster_embeddings(embeddings, n_clusters, method, eps)

    # Generate contexts for each cluster
    contexts = []
    unique_labels = sorted(set(labels))

    for label in unique_labels:
        # Skip noise points in DBSCAN
        if label == -1:
            continue

        # Get cluster indices
        cluster_indices = [i for i, l in enumerate(labels) if l == label]

        # Skip small clusters
        if len(cluster_indices) < min_cluster_size:
            continue

        # Get cluster center
        if centers is not None and label < len(centers):
            center = centers[label]
        else:
            # Compute center for this cluster
            center = embeddings_array[cluster_indices].mean(axis=0)

        # Get representative chunks
        representative_chunks = _get_representative_chunks_from_cluster(
            cluster_indices,
            embeddings_array,
            center,
            content,
            metadata,
            chunks_per_cluster,
            diversity_weight
        )

        # Calculate cluster coherence (average pairwise similarity)
        cluster_embeddings = embeddings_array[cluster_indices]
        if len(cluster_embeddings) > 1:
            similarity_matrix = cosine_similarity(cluster_embeddings)
            # Average of upper triangle (excluding diagonal)
            coherence = (similarity_matrix.sum() - len(cluster_embeddings)) / (len(cluster_embeddings) * (len(cluster_embeddings) - 1))
        else:
            coherence = 1.0

        contexts.append({
            'cluster_id': int(label),
            'chunks': representative_chunks,
            'size': len(cluster_indices),
            'coherence': float(coherence)
        })

    # Sort by cluster size (largest first)
    contexts.sort(key=lambda x: x['size'], reverse=True)

    return contexts




