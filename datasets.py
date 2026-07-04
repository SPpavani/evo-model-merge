"""
Synthetic domain datasets — standing in for three real specialized corpora,
e.g. how Sakana AI merged a Japanese-language model with a math-reasoning model.

Each domain has a different underlying decision boundary, so a model trained
on domain A genuinely under-performs on domain B unless the merge is good.
"""
import torch
import numpy as np


def make_domain_dataset(domain: str, n_samples: int = 2000, input_dim: int = 20, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, input_dim)).astype(np.float32)

    # Each domain uses a different random (but fixed) projection + nonlinearity
    # to define its class boundaries -> simulates a distinct task/language/skill.
    proj_seed = {"japanese_nlp": 1, "math_reasoning": 2, "code_generation": 3}[domain]
    proj_rng = np.random.default_rng(proj_seed)
    W = proj_rng.normal(size=(input_dim, 4)).astype(np.float32)

    logits = X @ W
    if domain == "math_reasoning":
        logits = np.sin(logits * 2.0)          # nonlinear boundary
    elif domain == "code_generation":
        logits = logits + np.roll(logits, 1, axis=1) * 0.5  # cross-feature interaction

    y = np.argmax(logits, axis=1).astype(np.int64)

    return torch.from_numpy(X), torch.from_numpy(y)


DOMAINS = ["japanese_nlp", "math_reasoning", "code_generation"]
