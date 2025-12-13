import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

_embed_model = None

def _get_embed():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def pick_best_tot_candidate(consensus_caption: str, candidates: list[str]) -> dict:
    """
    Choisit le meilleur candidat ToT par similarité sémantique au consensus,
    avec une petite pénalité si le texte est trop long (évite blabla).
    """
    m = _get_embed()
    emb = m.encode([consensus_caption] + candidates)

    base = emb[0:1]
    c_emb = emb[1:]
    sims = cosine_similarity(base, c_emb)[0]

    lengths = np.array([len(c.split()) for c in candidates], dtype=float)
    length_penalty = np.clip((lengths - 20) / 30, 0, 0.2)  # max -0.2

    final_scores = sims - length_penalty
    idx = int(np.argmax(final_scores))

    return {
        "candidates": candidates,
        "similarity_to_consensus": {f"cand_{i+1}": float(sims[i]) for i in range(len(candidates))},
        "final_score": {f"cand_{i+1}": float(final_scores[i]) for i in range(len(candidates))},
        "picked": f"cand_{idx+1}",
        "picked_caption": candidates[idx],
    }
