from typing import Dict, Tuple
import numpy as np
from insight_extraction.categorizer.matching.matcher import match_categories_for_dimension

def match_all_dimensions(
    intent: Dict[str, any],
    obs_embs: np.ndarray,
    dim2cat_embs: Dict[str, Dict[str, np.ndarray]],
    similarity_threshold: float = 0.4,
    min_support_ratio: float = 0.01
) -> Tuple[
    Dict[str, Dict[str, any]],
    Dict[str, np.ndarray]
]:

    all_stats = {}
    all_best = {}

    for dim, cat_embs in dim2cat_embs.items():
        stats, best = match_categories_for_dimension(
            dim, cat_embs, obs_embs,
            similarity_threshold, min_support_ratio
        )
        all_stats[dim] = stats
        all_best[dim] = best

    return all_stats, all_best
