"""Métricas de ranqueamento com desempate aleatório reprodutível e bootstrap pareado."""

from __future__ import annotations

import numpy as np
import torch


def ranks(scores: torch.Tensor, positives: list[list[int]], tie_break: torch.Tensor) -> list[np.ndarray]:
    """Posição (1 = topo) de cada positivo de cada consulta; empates resolvidos por uma permutação fixa."""
    out = []
    for row, items in zip(scores, positives):
        target = row[items].unsqueeze(1)
        better = (row.unsqueeze(0) > target) | ((row.unsqueeze(0) == target) & (tie_break.unsqueeze(0) < tie_break[items].unsqueeze(1)))
        out.append(better.sum(1).numpy() + 1)
    return out


def query_metrics(position: np.ndarray, k_values=(5, 10)) -> dict[str, float]:
    ideal = sum(1 / np.log2(i + 2) for i in range(min(len(position), 10)))
    values = {f"recall@{k}": float(np.mean(position <= k)) for k in k_values}
    values["ndcg@10"] = float(sum(1 / np.log2(p + 1) for p in position if p <= 10) / ideal)
    values["mrr"] = float(1 / position.min())
    return values


def filtered_pair_ranks(position: np.ndarray) -> np.ndarray:
    """Rank de cada positivo descontando os outros positivos que ficaram à frente dele."""
    order = np.argsort(position)
    filtered = np.empty_like(position)
    filtered[order] = position[order] - np.arange(len(position))
    return filtered


def summarize(positions: list[np.ndarray], disjoint: list[np.ndarray]) -> tuple[dict, dict[str, np.ndarray]]:
    per_query = [query_metrics(p) for p in positions]
    names = per_query[0]
    per_query_arrays = {name: np.asarray([q[name] for q in per_query]) for name in names}
    pairs = np.concatenate([filtered_pair_ranks(p) for p in positions])
    mask = np.concatenate(disjoint)
    summary = {name: float(values.mean()) for name, values in per_query_arrays.items()}
    summary.update(
        {
            "pair_hits@10": float(np.mean(pairs <= 10)),
            "pair_mrr": float(np.mean(1 / pairs)),
            "disjoint_pair_hits@10": float(np.mean(pairs[mask] <= 10)),
            "disjoint_pair_mrr": float(np.mean(1 / pairs[mask])),
            "queries": len(positions),
            "pairs": int(len(pairs)),
            "disjoint_pairs": int(mask.sum()),
        }
    )
    return summary, per_query_arrays


def paired_bootstrap(left: np.ndarray, right: np.ndarray, iterations: int = 10_000, seed: int = 3407) -> dict[str, float]:
    differences = np.asarray(left, float) - np.asarray(right, float)
    rng = np.random.default_rng(seed)
    replicates = np.asarray([differences[rng.integers(0, len(differences), len(differences))].mean() for _ in range(iterations)])
    lower, upper = np.quantile(replicates, [0.025, 0.975])
    return {"difference": float(differences.mean()), "ci95_lower": float(lower), "ci95_upper": float(upper), "n": len(differences)}
