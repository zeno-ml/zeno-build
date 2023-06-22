"""Features related to clustering of text data."""
import math

import numpy as np
from pandas import DataFrame
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from zeno import DistillReturn, ZenoOptions, distill


def _cluster_docs(
    documents: list[str],
    num_clusters: int,
    model_name: str,
) -> np.ndarray:
    # Load the model. Models are downloaded automatically if not present
    model = SentenceTransformer(model_name)

    # Encode the documents to get their embeddings
    document_embeddings = model.encode(documents)

    # Perform kmeans clustering
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(document_embeddings)

    # Get the cluster number for each document
    return kmeans.labels_


@distill
def data_clusters(df: DataFrame, ops: ZenoOptions) -> DistillReturn:
    """Cluster the labels together to find similar sentences.

    Args:
        df: Zeno DataFrame
        ops: Zeno options

    Returns:
        DistillReturn: Number of digits in the output
    """
    documents = [str(x) for x in df[ops.data_column]]
    num_clusters = int(math.sqrt(len(documents)))
    document_clusters = _cluster_docs(documents, num_clusters, "all-MiniLM-L6-v2")
    return DistillReturn(distill_output=document_clusters)


@distill
def label_clusters(df: DataFrame, ops: ZenoOptions) -> DistillReturn:
    """Cluster the labels together to find similar sentences.

    Args:
        df: Zeno DataFrame
        ops: Zeno options

    Returns:
        DistillReturn: Number of digits in the output
    """
    documents = [str(x) for x in df[ops.label_column]]
    num_clusters = int(math.sqrt(len(documents)))
    document_clusters = _cluster_docs(documents, num_clusters, "all-MiniLM-L6-v2")
    return DistillReturn(distill_output=document_clusters)
