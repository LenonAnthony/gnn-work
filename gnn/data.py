"""Monta o grafo KCRG, os rótulos de synset e as features de texto a partir do PictoInstruct (somente leitura)."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import torch

PICTOINSTRUCT = Path(os.environ.get("PICTOINSTRUCT_ROOT", "/Documents/livox-ai/livox-data-science/PictoInstruct"))
SOURCES = {
    "kcrg": "data/processed/offline/graphs/G_kcrg.json",
    "seeds": "data/processed/offline/graphs/seed_units.jsonl",
    "concepts": "data/interim/offline/grouped/generation_concepts.jsonl",
    "splits": "data/processed/offline/dataset_offline_silver_v1/split_assignments.json",
}
ENCODERS = {
    "labse": ("sentence-transformers/LaBSE", "836121a0533e5664b21c7aacc5d22951f2b8b25b"),
    "serafim": ("PORTULAN/serafim-335m-portuguese-pt-sentence-encoder", "a01887015444f7599669c509447c5bdbce958916"),
}
KINDS = ("anchor", "category", "tag")
MAX_SYNSET_SIZE = 50
STOPWORDS = set("a o as os e de da do das dos em no na nos nas com para por um uma ao à".split())
DATA = Path("data")


def tokens(label: str) -> set[str]:
    return {t for t in re.findall(r"\w+", label.casefold()) if t not in STOPWORDS}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build() -> dict:
    paths = {name: PICTOINSTRUCT / relative for name, relative in SOURCES.items()}
    kcrg = json.loads(paths["kcrg"].read_text())
    nodes = sorted(kcrg["nodes"], key=lambda node: node["id"])
    index = {node["id"]: i for i, node in enumerate(nodes)}

    anchor_of_pictogram = {
        seed["pictogram_id"]: next(k["kc_id"] for k in seed["key_concepts"] if k["kind"] == "anchor")
        for seed in read_jsonl(paths["seeds"])
    }
    split_of = {row["concept_id"]: row["split"] for row in json.loads(paths["splits"].read_text())["assignments"]}
    concepts = read_jsonl(paths["concepts"])
    node_of_concept = {}
    for concept in concepts:
        anchors = {anchor_of_pictogram[p] for p in concept["pictogram_ids"] if p in anchor_of_pictogram}
        if len(anchors) != 1:
            raise ValueError(f"{concept['concept_id']}: expected one KCRG anchor, got {len(anchors)}")
        node_of_concept[concept["concept_id"]] = index[anchors.pop()]
    if len(set(node_of_concept.values())) != len(node_of_concept):
        raise ValueError("two concepts map to the same anchor")

    split = ["none"] * len(nodes)
    concept_id = [None] * len(nodes)
    for cid, node in node_of_concept.items():
        split[node] = split_of.get(cid, "none")
        concept_id[node] = cid

    members = collections.defaultdict(set)
    for concept in concepts:
        for synset in concept["synsets"]:
            members[synset].add(node_of_concept[concept["concept_id"]])
    pairs = {}
    for synset, group in sorted(members.items()):
        if 1 < len(group) <= MAX_SYNSET_SIZE:
            ordered = sorted(group)
            for i, left in enumerate(ordered):
                for right in ordered[i + 1 :]:
                    pairs.setdefault((left, right), []).append(synset)
    labels = [node["label_ptbr"] for node in nodes]
    positives = [
        {"left": left, "right": right, "synsets": synsets, "lexically_disjoint": not (tokens(labels[left]) & tokens(labels[right]))}
        for (left, right), synsets in sorted(pairs.items())
    ]

    edges = np.asarray([[index[e["source"]], index[e["target"]]] for e in kcrg["edges"]], dtype=np.int64)
    weights = np.asarray([e["weight"] for e in kcrg["edges"]], dtype=np.float32)
    graph = {
        "labels": labels,
        "kinds": [node["kind"] for node in nodes],
        "kc_ids": [node["id"] for node in nodes],
        "concept_ids": concept_id,
        "splits": split,
        "edges": edges.tolist(),
        "edge_weights": weights.tolist(),
        "positives": positives,
    }
    manifest = {
        "version": "gnn_work_kcrg_synset_v1",
        "sources": {name: {"path": str(path.relative_to(PICTOINSTRUCT)), "sha256": sha256(path)} for name, path in paths.items()},
        "nodes": len(nodes),
        "nodes_by_kind": collections.Counter(graph["kinds"]),
        "kcrg_edges": len(edges),
        "anchors_by_split": collections.Counter(s for s, k in zip(split, graph["kinds"]) if k == "anchor"),
        "max_synset_size": MAX_SYNSET_SIZE,
        "excluded_hub_synsets": sorted(s for s, g in members.items() if len(g) > MAX_SYNSET_SIZE),
        "positive_pairs": len(positives),
        "lexically_disjoint_pairs": sum(p["lexically_disjoint"] for p in positives),
        "anchors_with_partner": len({n for p in positives for n in (p["left"], p["right"])}),
        "target_relation": "dois conceitos compartilham ao menos um synset WordNet anotado na ARASAAC; essa relação não entra no KCRG",
    }
    return graph, manifest


def encode(labels: list[str]) -> dict[str, np.ndarray]:
    from sentence_transformers import SentenceTransformer

    out = {}
    for name, (model_id, revision) in ENCODERS.items():
        model = SentenceTransformer(model_id, revision=revision, device="cpu")
        out[name] = np.asarray(
            model.encode(labels, batch_size=256, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False),
            dtype=np.float32,
        )
    return out


def load() -> tuple[dict, dict[str, torch.Tensor]]:
    graph = json.loads((DATA / "graph.json").read_text())
    features = {name: torch.from_numpy(np.load(DATA / f"features_{name}.npy")) for name in ENCODERS}
    return graph, features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-features", action="store_true")
    args = parser.parse_args()
    DATA.mkdir(exist_ok=True)
    graph, manifest = build()
    (DATA / "graph.json").write_text(json.dumps(graph, ensure_ascii=False))
    if not args.skip_features:
        for name, matrix in encode(graph["labels"]).items():
            np.save(DATA / f"features_{name}.npy", matrix)
    manifest["outputs"] = {path.name: sha256(path) for path in sorted(DATA.glob("*")) if path.name != "manifest.json"}
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("sources", "outputs", "excluded_hub_synsets")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
