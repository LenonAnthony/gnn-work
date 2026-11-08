"""Treina MLP e GraphSAGE com pares de synset e compara com baselines de grafo e de texto."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import to_undirected

from gnn.data import KINDS, load
from gnn.metrics import paired_bootstrap, ranks, summarize

CONFIG = {
    "hidden": 512,
    "embedding": 256,
    "dropout": 0.2,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "batch_pairs": 256,
    "extra_negatives": 1024,
    "temperature": 0.05,
    "max_epochs": 200,
    "patience": 20,
    "seeds": [3407, 3408, 3409],
    "bootstrap_iterations": 10_000,
}
TRAINED = {
    "mlp": "MLP (texto, sem grafo)",
    "graphsage": "GraphSAGE (texto + KCRG)",
    "graphsage_residual": "GraphSAGE residual (texto + KCRG)",
}
EXAMPLES = ("abelha", "tossir", "escola", "cachorro", "água", "dentista", "chorar", "futebol")


class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, CONFIG["hidden"]),
            nn.ReLU(),
            nn.Dropout(CONFIG["dropout"]),
            nn.Linear(CONFIG["hidden"], CONFIG["embedding"]),
        )

    def forward(self, x, edge_index):
        return F.normalize(self.net(x), dim=-1)


class GraphSAGE(nn.Module):
    def __init__(self, dim: int, residual: bool = False):
        super().__init__()
        self.input = nn.Linear(dim, CONFIG["hidden"])
        self.conv1 = SAGEConv(CONFIG["hidden"], CONFIG["hidden"])
        self.conv2 = SAGEConv(CONFIG["hidden"], CONFIG["embedding"])
        self.skip = nn.Linear(CONFIG["hidden"], CONFIG["embedding"]) if residual else None
        self.dropout = nn.Dropout(CONFIG["dropout"])

    def forward(self, x, edge_index):
        h0 = self.dropout(F.relu(self.input(x)))
        h1 = self.dropout(F.relu(self.conv1(h0, edge_index)))
        out = self.conv2(h1, edge_index)
        if self.skip is not None:
            out = out + self.skip(h0)
        return F.normalize(out, dim=-1)


class Setup:
    def __init__(self):
        graph, features = load()
        self.graph = graph
        self.labels = graph["labels"]
        kinds = graph["kinds"]
        self.anchors = torch.tensor([i for i, k in enumerate(kinds) if k == "anchor"])
        self.position = {int(node): i for i, node in enumerate(self.anchors)}
        one_hot = F.one_hot(torch.tensor([KINDS.index(k) for k in kinds]), len(KINDS)).float()
        self.text = features
        self.x = torch.cat([features["labse"], features["serafim"], one_hot], dim=1)
        self.edge_index = to_undirected(torch.tensor(graph["edges"]).T)
        self.partners = {}
        for pair in graph["positives"]:
            for a, b in ((pair["left"], pair["right"]), (pair["right"], pair["left"])):
                self.partners.setdefault(a, []).append((b, pair["lexically_disjoint"]))
        splits = graph["splits"]
        self.queries = {
            split: sorted(n for n in self.partners if splits[n] == split) for split in ("development", "test")
        }
        self.train_pairs = torch.tensor(
            [[p["left"], p["right"]] for p in graph["positives"] if splits[p["left"]] == splits[p["right"]] == "optimization"]
        )
        self.tie_break = torch.from_numpy(np.random.default_rng(3407).permutation(len(self.anchors))).float()

    def evaluate(self, scores_for, split):
        """scores_for(query_nodes) -> tensor [queries, anchors]; o próprio nó recebe -inf."""
        queries = self.queries[split]
        positions, disjoint = [], []
        for start in range(0, len(queries), 512):
            batch = queries[start : start + 512]
            scores = scores_for(torch.tensor(batch)).float()
            scores[torch.arange(len(batch)), torch.tensor([self.position[q] for q in batch])] = -torch.inf
            items = [[self.position[p] for p, _ in self.partners[q]] for q in batch]
            positions.extend(ranks(scores, items, self.tie_break))
            disjoint.extend(np.asarray([d for _, d in self.partners[q]]) for q in batch)
        return summarize(positions, disjoint)


def baselines(setup: Setup) -> dict:
    rows = {}
    anchors = setup.anchors
    rng = torch.Generator().manual_seed(3407)
    random_scores = torch.rand(len(anchors), generator=rng)
    rows["Aleatório"] = lambda q: random_scores.expand(len(q), -1).clone()

    n = len(setup.labels)
    edge_index = setup.edge_index
    degree = torch.bincount(edge_index[0], minlength=n).float()
    rows["Popularidade (grau no KCRG)"] = lambda q: degree[anchors].expand(len(q), -1).clone()

    adjacency = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1]), (n, n)).coalesce()
    inverse_log = torch.where(degree > 1, 1 / torch.log(degree.clamp(min=2)), torch.zeros_like(degree))
    weighted = (adjacency.to_dense()[anchors] * inverse_log).to_sparse()
    dense_anchor = adjacency.to_dense()[anchors]

    def adamic_adar(q):
        rows_q = dense_anchor[[setup.position[int(i)] for i in q]]
        return (weighted @ rows_q.T).T.to_dense()

    rows["Adamic-Adar (KCRG, 2 saltos)"] = adamic_adar
    for name, label in (("labse", "Cosseno LaBSE"), ("serafim", "Cosseno Serafim")):
        matrix = setup.text[name]
        rows[label] = lambda q, m=matrix: m[q] @ m[anchors].T
    return rows


def train(setup: Setup, kind: str, seed: int, log: list) -> nn.Module:
    torch.manual_seed(seed)
    dim = setup.x.shape[1]
    model = MLP(dim) if kind == "mlp" else GraphSAGE(dim, residual=kind == "graphsage_residual")
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    generator = torch.Generator().manual_seed(seed)
    best, best_state, waited = -1.0, None, 0
    pairs = setup.train_pairs
    for epoch in range(1, CONFIG["max_epochs"] + 1):
        model.train()
        order = torch.randperm(len(pairs), generator=generator)
        losses = []
        for start in range(0, len(pairs), CONFIG["batch_pairs"]):
            batch = pairs[order[start : start + CONFIG["batch_pairs"]]]
            flip = torch.rand(len(batch), generator=generator) < 0.5
            left = torch.where(flip, batch[:, 1], batch[:, 0])
            right = torch.where(flip, batch[:, 0], batch[:, 1])
            extra = setup.anchors[torch.randint(len(setup.anchors), (CONFIG["extra_negatives"],), generator=generator)]
            z = model(setup.x, setup.edge_index)
            logits = z[left] @ torch.cat([z[right], z[extra]]).T / CONFIG["temperature"]
            loss = F.cross_entropy(logits, torch.arange(len(batch)))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            z = model(setup.x, setup.edge_index)
        dev, _ = setup.evaluate(lambda q: z[q] @ z[setup.anchors].T, "development")
        log.append({"model": kind, "seed": seed, "epoch": epoch, "loss": float(np.mean(losses)), "dev_mrr": dev["mrr"]})
        if dev["mrr"] > best:
            best, waited = dev["mrr"], 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= CONFIG["patience"]:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


def examples(setup: Setup, scorers: dict) -> dict:
    by_label = {}
    for node in setup.anchors.tolist():
        by_label.setdefault(setup.labels[node], node)
    out = {}
    for label in EXAMPLES:
        if label not in by_label:
            continue
        node = by_label[label]
        out[label] = {"split": setup.graph["splits"][node], "synset_partners": [setup.labels[p] for p, _ in setup.partners.get(node, [])]}
        for name, scorer in scorers.items():
            scores = scorer(torch.tensor([node]))[0].float()
            scores[setup.position[node]] = -torch.inf
            top = torch.topk(scores, 5).indices
            out[label][name] = [setup.labels[int(setup.anchors[i])] for i in top]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--seeds", type=int, nargs="+", default=CONFIG["seeds"])
    parser.add_argument("--max-epochs", type=int, default=CONFIG["max_epochs"])
    args = parser.parse_args()
    CONFIG["seeds"], CONFIG["max_epochs"] = args.seeds, args.max_epochs
    torch.set_num_threads(8)
    args.output.mkdir(parents=True, exist_ok=True)
    setup = Setup()

    results, per_query, log = {}, {}, []
    scorers = baselines(setup)
    for name, scorer in scorers.items():
        results[name] = {split: setup.evaluate(scorer, split)[0] for split in ("development", "test")}
        per_query[name] = setup.evaluate(scorer, "test")[1]
        print(name, json.dumps(results[name]["test"]), flush=True)

    for kind, label in TRAINED.items():
        runs = []
        for seed in CONFIG["seeds"]:
            started = time.perf_counter()
            model = train(setup, kind, seed, log)
            with torch.no_grad():
                z = model(setup.x, setup.edge_index)
            scorer = lambda q, z=z: z[q] @ z[setup.anchors].T
            dev, _ = setup.evaluate(scorer, "development")
            test, arrays = setup.evaluate(scorer, "test")
            runs.append({"seed": seed, "development": dev, "test": test, "seconds": time.perf_counter() - started})
            if seed == CONFIG["seeds"][0]:
                per_query[label], scorers[label] = arrays, scorer
                torch.save(model.state_dict(), args.output / f"{kind}_seed{seed}.pt")
            print(label, seed, json.dumps(test), flush=True)
        metrics = runs[0]["test"].keys()
        results[label] = {
            split: {
                m: float(np.mean([r[split][m] for r in runs])) for m in metrics
            }
            | {f"{m}_sd": float(np.std([r[split][m] for r in runs], ddof=1)) for m in metrics if len(runs) > 1}
            for split in ("development", "test")
        }
        results[label]["runs"] = runs

    gnn = max((TRAINED["graphsage"], TRAINED["graphsage_residual"]), key=lambda name: results[name]["development"]["mrr"])
    comparisons = {
        f"{gnn} - {other}": {m: paired_bootstrap(per_query[gnn][m], per_query[other][m], CONFIG["bootstrap_iterations"]) for m in ("mrr", "recall@10", "ndcg@10")}
        for other in per_query
        if other != gnn
    }
    report = {
        "config": CONFIG,
        "train_pairs": len(setup.train_pairs),
        "queries": {split: len(q) for split, q in setup.queries.items()},
        "candidates": len(setup.anchors),
        "selected_gnn_by_development_mrr": gnn,
        "results": results,
        "bootstrap_test_seed3407": comparisons,
        "note": "Relação-alvo silver (synset WordNet compartilhado na ARASAAC); não é julgamento profissional de CAA.",
    }
    (args.output / "results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    (args.output / "training_log.json").write_text(json.dumps(log, indent=1))
    (args.output / "examples.json").write_text(json.dumps(examples(setup, scorers), indent=2, ensure_ascii=False))
    print(json.dumps(comparisons, indent=1))


if __name__ == "__main__":
    main()
