import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Tuple
from dataclasses import dataclass

@dataclass
class CategoryStats:
    dimension_type: str
    category: str
    support_count: int
    support_ratio: float
    mean_score: float

def match_categories_for_dimension(
    dim_type: str,
    cat_embs: Dict[str, np.ndarray],
    obs_embs: np.ndarray,
    similarity_threshold: float = 0.4,
    min_support_ratio: float = 0.01
) -> Tuple[Dict[str, CategoryStats], np.ndarray]:

    cat_names = list(cat_embs.keys())
    if not cat_names:
        return {}, np.full(len(obs_embs), -1)

    matrix = np.stack([cat_embs[c] for c in cat_names])
    sims = cosine_similarity(obs_embs, matrix)

    best_idx = sims.argmax(axis=1)
    best_scores = sims[np.arange(len(obs_embs)), best_idx]
    mask = best_scores >= similarity_threshold

    N = len(obs_embs)
    stats = {}
    valid_mask = np.zeros(len(cat_names), dtype=bool)

    for i, cname in enumerate(cat_names):
        sel = (best_idx == i) & mask
        count = int(sel.sum())
        ratio = count / N if N else 0

        if ratio >= min_support_ratio:
            stats[cname] = CategoryStats(
                dimension_type=dim_type,
                category=cname,
                support_count=count,
                support_ratio=ratio,
                mean_score=float(best_scores[sel].mean()) if count else 0
            )
            valid_mask[i] = True

    remapped = np.array([
        idx if mask[j] and valid_mask[idx] else -1
        for j, idx in enumerate(best_idx)
    ])

    return stats, remapped
