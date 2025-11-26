import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer


def embed_texts(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int = 32
) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True
    )


def build_category_text(cat: Dict[str, Any]) -> str:
    """
    Concatenate name + description + synonyms + examples
    into a single text optimized for embedding.
    """
    name = cat.get("name", "")
    desc = cat.get("description", "")
    syns = ", ".join(cat.get("synonyms", []))
    exs = " | ".join(cat.get("examples", []))

    return (
        f"Category: {name}. "
        f"Description: {desc}. "
        f"Synonyms: {syns}. "
        f"Examples: {exs}."
    ).strip()


def embed_categories(
    model: SentenceTransformer,
    intent: Dict[str, Any],
    expansions: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Improved version:
    - Use expansions to create rich texts
        for each category.
    - Embed those texts instead of using only the category names.
    """
    dim2cat_embs = {}

    for group in intent.get("group_by", []):
        dim = group.get("dimension_type")
        values = list(dict.fromkeys(group.get("values", [])))  # uniq

        if not values:
            continue

        # expansions for this dimension
        exp_for_dim = expansions.get(dim)
        if exp_for_dim is None:
            continue

        # build rich texts
        rich_texts = []
        valid_values = []

        for v in values:
            if v not in exp_for_dim:
                continue

            rich_text = build_category_text(exp_for_dim[v])
            rich_texts.append(rich_text)
            valid_values.append(v)

        if rich_texts:
            vectors = embed_texts(model, rich_texts)

            dim2cat_embs[dim] = {
                v: vectors[i] for i, v in enumerate(valid_values)
            }

    return dim2cat_embs
