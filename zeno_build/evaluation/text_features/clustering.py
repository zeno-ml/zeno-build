"""Features related to clustering of text data."""
from pandas import DataFrame
from sklearn.cluster import KMeans
from zeno import DistillReturn, ZenoOptions, distill


def _cluster_docs(
    documents: list[str],
    num_clusters: int,
    model_name: str,
) -> list[str]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "Please install sentence-transformers to use clustering features."
        )

    # Load the model. Models are downloaded automatically if not present
    model = SentenceTransformer(model_name)

    # Encode the documents to get their embeddings
    document_embeddings = model.encode(documents)

    # Perform kmeans clustering
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(document_embeddings)

    # Get the cluster number for each document
    return [f"C{x}" for x in kmeans.labels_.tolist()]


@distill
def data_clusters(df: DataFrame, ops: ZenoOptions) -> DistillReturn:
    """Cluster the labels together to find similar sentences.

    Args:
        df: Zeno DataFrame
        ops: Zeno options

    Returns:
        DistillReturn: Cluster IDs of each of the input data points
    """
    documents = [str(x) for x in df[ops.data_column]]
    # TODO(gneubig): having something like the sqrt of the number of documents seems
    #   like a good number of clusters, but more than 20 are not supported in Zeno:
    #   https://github.com/zeno-ml/zeno/issues/873
    # num_clusters = int(math.sqrt(len(documents)))
    num_clusters = 20
    document_clusters = _cluster_docs(documents, num_clusters, "all-MiniLM-L6-v2")
    return DistillReturn(distill_output=document_clusters)


@distill
def label_clusters(df: DataFrame, ops: ZenoOptions) -> DistillReturn:
    """Cluster the labels together to find similar sentences.

    Args:
        df: Zeno DataFrame
        ops: Zeno options

    Returns:
        DistillReturn: Cluster IDs of each of the labels
    """
    documents = [str(x) for x in df[ops.label_column]]
    # TODO(gneubig): having something like the sqrt of the number of documents seems
    #   like a good number of clusters, but more than 20 are not supported in Zeno:
    #   https://github.com/zeno-ml/zeno/issues/873
    # num_clusters = int(math.sqrt(len(documents)))
    num_clusters = 20
    document_clusters = _cluster_docs(documents, num_clusters, "all-MiniLM-L6-v2")
    return DistillReturn(distill_output=document_clusters)
