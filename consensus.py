import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# 🔹 LOAD EMBEDDING MODEL (ONCE)
# =========================================================
_embedding_model = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


# =========================================================
# 🔹 SEMANTIC CONSENSUS
# =========================================================
def semantic_consensus(captions: dict) -> dict:
    """
    Compute a semantic consensus among multiple captions.

    Args:
        captions (dict): {
            "BLIP Base": "...",
            "ViT-GPT2": "...",
            "GIT": "..."
        }

    Returns:
        dict: {
            "best_model": str,
            "best_caption": str,
            "scores": dict,
            "similarity_matrix": list
        }
    """

    model_names = list(captions.keys())
    texts = list(captions.values())

    model = get_embedding_model()
    embeddings = model.encode(texts)

    # Similarity matrix
    sim_matrix = cosine_similarity(embeddings)

    # Mean similarity score for each caption
    mean_scores = sim_matrix.mean(axis=1)

    best_index = int(np.argmax(mean_scores))

    return {
        "best_model": model_names[best_index],
        "best_caption": texts[best_index],
        "scores": {
            model_names[i]: float(mean_scores[i])
            for i in range(len(model_names))
        },
        "similarity_matrix": sim_matrix.tolist()
    }
