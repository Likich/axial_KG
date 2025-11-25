#!/usr/bin/env python3
"""Reproducible helpers for the Voronoi diagrams prepared in the notebooks.

Two sub-commands are provided:
    diagram  – run the node2vec → UMAP → Voronoi pipeline and save a PNG.
    layers   – load a GEXF with fuzzy Louvain info, auto-pick seeds and export
               both CSV + GEXF with Voronoi-based shells.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

# Headless rendering when invoked from the CLI.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patheffects as patheffects  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy.cluster.hierarchy import fcluster, linkage  # noqa: E402
from scipy.spatial import Voronoi  # noqa: E402
from scipy.spatial.distance import cdist  # noqa: E402
import scipy.linalg as sla  # noqa: E402

if not hasattr(sla, "triu"):
    from numpy import triu as _np_triu

    sla.triu = _np_triu

try:
    from node2vec import Node2Vec  # noqa: E402
except ImportError:
    Node2Vec = None

try:
    from umap import UMAP  # noqa: E402
except ImportError:
    UMAP = None

try:
    from openai import OpenAI  # noqa: E402
except ImportError:
    OpenAI = None


def _load_graph(path: Path) -> nx.Graph:
    if not path.exists():
        raise FileNotFoundError(path)
    return nx.read_gexf(path)


def _ensure_coordinates(G: nx.Graph, seed: int = 42) -> pd.DataFrame:
    """Return a DataFrame with columns (node, x, y). Compute layout if needed."""
    missing_xy = [
        n
        for n, attrs in G.nodes(data=True)
        if "x" not in attrs or "y" not in attrs
    ]
    if missing_xy:
        pos = nx.spring_layout(G, weight="weight", dim=2, seed=seed)
        for n, (x, y) in pos.items():
            G.nodes[n]["x"] = float(x)
            G.nodes[n]["y"] = float(y)

    coords = pd.DataFrame(
        [(n, float(attrs["x"]), float(attrs["y"])) for n, attrs in G.nodes(data=True)],
        columns=["node", "x", "y"],
    ).set_index("node")
    return coords


def run_node2vec_umap(
    G: nx.Graph,
    *,
    dimensions: int = 64,
    walk_length: int = 30,
    num_walks: int = 200,
    workers: int = 1,
    window: int = 10,
    min_count: int = 1,
    n_neighbors: int = 20,
    min_dist: float = 0.3,
    metric: str = "cosine",
    random_state: int = 42,
) -> pd.DataFrame:
    """Return a 2-D embedding using node2vec followed by UMAP."""
    if Node2Vec is None:
        raise ImportError("node2vec is required for the 'diagram' command.")
    if UMAP is None:
        raise ImportError("umap-learn is required for the 'diagram' command.")
    node2vec = Node2Vec(
        G,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        workers=workers,
        seed=random_state,
    )
    model = node2vec.fit(window=window, min_count=min_count)
    nodes = list(G.nodes())
    emb = pd.DataFrame([model.wv[n] for n in nodes], index=nodes)

    um = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    coords = pd.DataFrame(
        um.fit_transform(emb),
        index=emb.index,
        columns=["x", "y"],
    )
    return coords


def top_pagerank_seeds(
    G: nx.Graph, coords: pd.DataFrame, count: int = 10
) -> pd.DataFrame:
    """Pick the PageRank top-k nodes as Voronoi seeds."""
    pr = nx.pagerank(G, weight="weight")
    order = sorted(pr, key=pr.get, reverse=True)[:count]
    missing = [n for n in order if n not in coords.index]
    if missing:
        raise ValueError(f"Coordinates missing for: {missing}")
    return coords.loc[order].copy()


def plot_voronoi(coords: pd.DataFrame, seeds: pd.DataFrame, out_path: Path) -> None:
    """Render the Voronoi diagram with all nodes and highlighted seeds."""
    points = coords[["x", "y"]].values
    vor = Voronoi(points)

    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    from scipy.spatial import voronoi_plot_2d

    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors="gray", line_alpha=0.3)
    ax.scatter(coords["x"], coords["y"], s=5, color="black")
    ax.scatter(seeds["x"], seeds["y"], s=80, color="red")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title("Voronoi shells on UMAP projection")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_louvain_voronoi(
    vor: Voronoi,
    coords: pd.DataFrame,
    centroids_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Replicate the exact layered visualization from the notebook."""
    from scipy.spatial import voronoi_plot_2d

    palette = {"core": "red", "mid": "orange", "outer": "lightblue"}
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    sns.set_style("white")

    voronoi_plot_2d(
        vor,
        ax=ax,
        show_vertices=False,
        show_points=False,
        line_colors="lightgray",
        line_alpha=0.3,
        line_width=0.5,
    )

    sns.scatterplot(
        data=coords,
        x="x",
        y="y",
        hue="layer",
        palette=palette,
        s=12,
        alpha=0.7,
        ax=ax,
        linewidth=0,
    )

    if not centroids_df.empty:
        ax.scatter(
            centroids_df["x"],
            centroids_df["y"],
            s=280,
            marker="*",
            edgecolors="black",
            linewidths=0.8,
            color="yellow",
            label="centroid",
        )
        # label each centroid with its node name for easier reading
        for node, row in centroids_df.iterrows():
            ax.text(
                row["x"],
                row["y"] + 0.15,
                str(node),
                fontsize=8,
                weight="bold",
                ha="center",
                va="bottom",
                color="black",
                path_effects=[patheffects.withStroke(linewidth=2, foreground="white")],
            )

    ax.set_title("Voronoi Tessellation Around Louvain Cluster Centroids\n(core/mid/outer shells)")
    ax.legend(title="layer")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def annotate_layers(coords: pd.DataFrame, seeds: pd.DataFrame) -> pd.DataFrame:
    """Assign each point to the nearest seed and label distance quantiles."""
    seed_coords = seeds[["x", "y"]].values
    dist = cdist(coords[["x", "y"]], seed_coords)
    assign = dist.argmin(axis=1)
    coords = coords.copy()
    coords["seed"] = [seeds.index[i] for i in assign]
    coords["dist_to_seed"] = dist.min(axis=1)
    coords["layer"] = pd.qcut(
        coords["dist_to_seed"],
        q=3,
        labels=["core", "mid", "outer"],
    )
    return coords


def save_coords(coords: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    coords.reset_index().rename(columns={"index": "node"}).to_csv(out_path, index=False)


# ----- Coding helpers -----


def sample_cluster_entities(
    coords: pd.DataFrame,
    comm_id: int,
    *,
    n_core: int = 40,
    n_mid: int = 25,
    n_outer: int = 15,
) -> Dict[str, List[str]]:
    sub = coords[coords["nearest_comm"] == comm_id]

    def pick(layer: str, n: int) -> List[str]:
        layer_df = sub[sub["layer"] == layer].sort_values("pagerank", ascending=False)
        return layer_df.index.tolist()[:n]

    return {
        "core": pick("core", n_core),
        "mid": pick("mid", n_mid),
        "outer": pick("outer", n_outer),
    }


def _bullet_list(items: Sequence[str], limit: int) -> str:
    vals = list(items)[:limit]
    if not vals:
        return "- (none)"
    return "\n".join(f"- {v}" for v in vals)


def parse_json_from_llm(raw: str, where: str = "") -> Optional[Dict]:
    """Robust JSON parsing for LLM responses."""
    if not raw:
        print(f"[WARN] Empty LLM response for {where}")
        return None

    text = raw.strip()
    fenced = re.search(r"```json(.*?)```", text, flags=re.S | re.I)
    if fenced:
        text = fenced.group(1).strip()
    else:
        block = re.search(r"\{.*\}", text, flags=re.S)
        if block:
            text = block.group(0).strip()

    def _repair_json(s: str) -> str:
        s = re.sub(r'"cluster_id"\s*:\s*([A-Za-z0-9_]+)', r'"cluster_id": "\1"', s)
        s = re.sub(r'"cluster_id"\s*:\s*"\\?"?([A-Za-z0-9_]+)\\?"?"', r'"cluster_id": "\1"', s)
        s = re.sub(r'}\s*{', '},{', s)
        s = re.sub(r',\s*([}\]])', r'\1', s)
        return s

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        cleaned = _repair_json(text)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            try:
                from json_repair import repair_json

                repaired = repair_json(cleaned)
                return json.loads(repaired)
            except Exception as exc:
                print(f"[JSON ERROR] {where}: {exc}")
                print("----------- RAW LLM OUTPUT (truncated) -----------")
                print(cleaned[:1000])
                print("--------------------------------------------------")
                return None


def build_openai_client(api_key: str, api_base: str) -> "OpenAI":
    if OpenAI is None:
        raise ImportError("openai package is required for coding commands.")
    if not api_key:
        raise ValueError("API key required for coding commands (set OPENAI_API_KEY).")
    return OpenAI(api_key=api_key, base_url=api_base or None)


OPEN_CODE_PROMPT = """
You are helping with qualitative open coding of clusters of related entities.
Each cluster comes from a Voronoi-layered knowledge graph. Use the core/mid/outer
examples to infer a concise conceptual label and supporting subcodes.

Cluster ID: {comm_id}

- CORE ENTITIES (most central / prototypical):
{core_list}

- MID ENTITIES (related concepts):
{mid_list}

- OUTER ENTITIES (peripheral or contextual signals):
{outer_list}

TASK:
1. Propose ONE short, human-readable CODE LABEL (3–6 words) that captures the shared idea.
2. List 3–6 SUBCODES (short phrases) that describe distinct aspects. Include short descriptions.
3. Attach 2–5 example entities for each subcode.
4. Give a 2–3 sentence explanation of the overall theme.

Respond with STRICT JSON matching:
{{
  "code_label": "...",
  "subcodes": [
    {{
      "name": "...",
      "description": "...",
      "example_entities": ["...", "..."]
    }}
  ],
  "explanation": "..."
}}
""".strip()


def open_code_cluster(
    client: "OpenAI",
    model: str,
    comm_id: int,
    ents: Dict[str, Sequence[str]],
    *,
    max_examples: int = 25,
) -> Optional[Dict]:
    prompt = OPEN_CODE_PROMPT.format(
        comm_id=comm_id,
        core_list=_bullet_list(ents.get("core", []), max_examples),
        mid_list=_bullet_list(ents.get("mid", []), max_examples),
        outer_list=_bullet_list(ents.get("outer", []), max_examples),
    )
    resp = client.chat.completions.create(
        model=model,
        temperature=0.35,
        max_tokens=500,
        messages=[
            {
                "role": "system",
                "content": "You are a rigorous qualitative researcher. Output ONLY valid JSON.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    raw = (resp.choices[0].message.content or "").strip()
    return parse_json_from_llm(raw, where=f"open_code_cluster {comm_id}")


AXIAL_PROMPT = """
You are performing axial coding by grouping several open-coded clusters.
Derive a higher-level category that links the provided cluster codes and entities.

Axial Group ID: {ax_id}

Included clusters and their open codes:
{cluster_block}

Representative entities (mix of core/mid/outer snippets):
{entity_block}

TASK:
1. Provide an AXIAL LABEL summarizing the shared storyline (4–8 words).
2. Briefly explain the overarching theme.
3. List 3–5 salient PROPERTIES/DRIVERS (short phrases).
4. Describe each cluster's role in this axial category.

IMPORTANT OUTPUT RULES:
- cluster_id values MUST be JSON strings (e.g., "5_core", not bare identifiers).
- Return strictly valid JSON (no comments, no trailing commas, no explanations outside JSON).

Return JSON in this structure:
{{
  "axial_label": "...",
  "summary": "...",
  "properties": ["...", "..."],
  "cluster_roles": [
    {{
      "cluster_id": 0,
      "role": "...",
      "linkage": "..."  // one sentence on how it connects
    }}
  ]
}}
""".strip()


def _collect_entities_for_axial(
    comms: Sequence[int],
    cluster_entities: Dict[int, Dict[str, Sequence[str]]],
    limit: int,
) -> str:
    samples: List[str] = []
    for cid in comms:
        layers = cluster_entities.get(cid, {})
        for layer_name in ("core", "mid", "outer"):
            layer_items = layers.get(layer_name, [])
            for ent in layer_items[: max(1, limit // 3)]:
                samples.append(f"Cluster {cid} [{layer_name}]: {ent}")
    return _bullet_list(samples, limit)


def axial_label(
    client: "OpenAI",
    model: str,
    ax_id: int,
    comms: Sequence[int],
    open_codes: Dict[int, Optional[Dict]],
    cluster_entities: Dict[int, Dict[str, Sequence[str]]],
    *,
    max_examples: int = 45,
) -> Optional[Dict]:
    cluster_lines = []
    for cid in comms:
        info = open_codes.get(cid) or {}
        label = info.get("code_label", "(unlabeled)")
        explanation = info.get("explanation", "")
        cluster_lines.append(f"- Cluster {cid}: {label} :: {explanation}")
    prompt = AXIAL_PROMPT.format(
        ax_id=ax_id,
        cluster_block="\n".join(cluster_lines) or "- (no open codes yet)",
        entity_block=_collect_entities_for_axial(comms, cluster_entities, max_examples),
    )
    resp = client.chat.completions.create(
        model=model,
        temperature=0.3,
        max_tokens=500,
        messages=[
            {
                "role": "system",
                "content": "You synthesize qualitative clusters into axial codes. Output JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    raw = (resp.choices[0].message.content or "").strip()
    parsed = parse_json_from_llm(raw, where=f"axial_group {ax_id}")
    if parsed and isinstance(parsed.get("cluster_roles"), list):
        for role in parsed["cluster_roles"]:
            if not isinstance(role, dict):
                continue
            cid = role.get("cluster_id", "")
            role["cluster_id"] = str(cid).strip('"')
    return parsed


def save_open_codes(
    open_codes: Dict[int, Optional[Dict]],
    summary_path: Path,
    subcodes_path: Path,
) -> None:
    summary_cols = ["cluster_id", "code_label", "explanation"]
    subcode_cols = ["cluster_id", "subcode_name", "description", "example_entities"]
    summary_rows = []
    subcode_rows = []
    for cluster_id, data in open_codes.items():
        if not data:
            continue
        summary_rows.append(
            {
                "cluster_id": cluster_id,
                "code_label": data.get("code_label", ""),
                "explanation": data.get("explanation", ""),
            }
        )
        for sub in data.get("subcodes", []) or []:
            subcode_rows.append(
                {
                    "cluster_id": cluster_id,
                    "subcode_name": sub.get("name", ""),
                    "description": sub.get("description", ""),
                    "example_entities": "; ".join(sub.get("example_entities", [])),
                }
            )
    pd.DataFrame(summary_rows, columns=summary_cols).to_csv(summary_path, index=False)
    pd.DataFrame(subcode_rows, columns=subcode_cols).to_csv(subcodes_path, index=False)
    print(f"Saved open-code summaries → {summary_path}")
    print(f"Saved open-code subcodes → {subcodes_path}")


def save_axial_codes(axial_codes: Dict[int, Optional[Dict]], out_path: Path) -> None:
    cols = ["axial_id", "axial_label", "summary", "properties", "cluster_roles"]
    rows = []
    for ax_id, data in axial_codes.items():
        if not data:
            continue
        rows.append(
            {
                "axial_id": ax_id,
                "axial_label": data.get("axial_label", ""),
                "summary": data.get("summary", ""),
                "properties": "; ".join(data.get("properties", []) or []),
                "cluster_roles": json.dumps(data.get("cluster_roles", [])),
            }
        )
    pd.DataFrame(rows, columns=cols).to_csv(out_path, index=False)
    print(f"Saved axial codes → {out_path}")


def save_axial_memberships(axial_map: Dict[int, List[int]], out_path: Path) -> None:
    cols = ["axial_id", "cluster_id"]
    rows = []
    for ax_id, comms in axial_map.items():
        for cid in comms:
            rows.append({"axial_id": ax_id, "cluster_id": cid})
    pd.DataFrame(rows, columns=cols).to_csv(out_path, index=False)
    print(f"Saved axial memberships → {out_path}")


# ----- Louvain-centric helpers -----
# ----- Louvain-centric helpers -----


def load_memberships_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "node" not in df.columns or "hard_comm" not in df.columns:
        raise ValueError("Membership CSV must contain 'node' and 'hard_comm' columns.")
    df = df.set_index("node")
    df["hard_comm"] = df["hard_comm"].astype(int)
    return df


def attach_hard_labels(coords: pd.DataFrame, memberships: pd.DataFrame) -> pd.DataFrame:
    coords = coords.join(memberships["hard_comm"], how="inner")
    coords = coords.dropna(subset=["hard_comm"])
    coords["hard_comm"] = coords["hard_comm"].astype(int)
    if coords.empty:
        raise ValueError("No overlap between graph nodes and membership CSV.")
    return coords


def compute_centroids(coords: pd.DataFrame) -> pd.DataFrame:
    centroids = []
    for comm in sorted(coords["hard_comm"].unique()):
        nodes_c = coords[coords["hard_comm"] == comm]
        if nodes_c.empty:
            continue
        mu = nodes_c[["x", "y"]].mean()
        idx = ((nodes_c[["x", "y"]] - mu) ** 2).sum(axis=1).idxmin()
        centroids.append(
            {
                "comm": comm,
                "node": idx,
                "x": coords.loc[idx, "x"],
                "y": coords.loc[idx, "y"],
            }
        )
    if not centroids:
        raise ValueError("Unable to derive centroids for any community.")
    return pd.DataFrame(centroids).set_index("node")


def assign_layers_per_comm(coords: pd.DataFrame, centroids_df: pd.DataFrame) -> pd.DataFrame:
    coords = coords.copy()
    seed_coords = centroids_df[["x", "y"]].values
    D = cdist(coords[["x", "y"]], seed_coords)
    nearest_idx = D.argmin(axis=1)
    coords["nearest_comm"] = centroids_df["comm"].iloc[nearest_idx].values
    coords["dist_to_seed"] = D[np.arange(len(coords)), nearest_idx]
    coords["layer"] = "outer"
    for comm, grp in coords.groupby("nearest_comm"):
        if len(grp) < 3:
            continue
        q1, q2 = grp["dist_to_seed"].quantile([1 / 3, 2 / 3]).values
        coords.loc[grp.index[grp["dist_to_seed"] <= q1], "layer"] = "core"
        coords.loc[
            grp.index[
                (grp["dist_to_seed"] > q1) & (grp["dist_to_seed"] <= q2)
            ],
            "layer",
        ] = "mid"
    return coords


# ----- Auto Voronoi shells from fuzzy Louvain -----


def _hard_labels(G: nx.Graph, attr: str) -> Dict[str, int]:
    labels = {}
    missing = []
    for n, attrs in G.nodes(data=True):
        if attr not in attrs:
            missing.append(n)
        else:
            labels[n] = int(attrs[attr])
    if missing:
        raise ValueError(f"Nodes missing '{attr}': {len(missing)}")
    return labels


def _soft_membership_attrs(G: nx.Graph, prefix: str = "p_comm_") -> List[str]:
    for _, attrs in G.nodes(data=True):
        cols = [k for k in attrs.keys() if str(k).startswith(prefix)]
        if cols:
            return sorted(cols)
    return []


def _auto_seed_counts(group_sizes: Dict[int, int], nodes_per_seed: int) -> Dict[int, int]:
    return {
        comm: max(1, math.ceil(size / float(nodes_per_seed)))
        for comm, size in group_sizes.items()
    }


def _pick_seeds_by_pagerank(
    groups: Dict[int, List[str]],
    pr: Dict[str, float],
    per_comm: Dict[int, int],
) -> List[str]:
    seeds = []
    for comm, members in groups.items():
        n_seeds = per_comm[comm]
        top = sorted(members, key=lambda n: pr.get(n, 0.0), reverse=True)[:n_seeds]
        seeds.extend(top)
    return seeds


def _bridge_entropy(row: pd.Series, attrs: Dict[str, Dict[str, float]], soft_cols: List[str]) -> float:
    if not soft_cols:
        return float("nan")
    vals = np.array([float(attrs[row.name].get(col, 0.0)) for col in soft_cols], dtype=float)
    total = vals.sum()
    if not total:
        return 0.0
    vals /= total
    nz = vals[vals > 0]
    return float(-(nz * np.log(nz)).sum())


def run_auto_layers(
    gexf_path: Path,
    out_prefix: Path,
    *,
    hard_attr: str = "hard_comm",
    nodes_per_seed: int = 250,
) -> None:
    G = _load_graph(gexf_path)
    coords = _ensure_coordinates(G)

    hard_labels = _hard_labels(G, hard_attr)
    groups: Dict[int, List[str]] = {}
    for node, label in hard_labels.items():
        groups.setdefault(label, []).append(node)

    pr = nx.pagerank(G, weight="weight")
    counts = _auto_seed_counts({c: len(m) for c, m in groups.items()}, nodes_per_seed)
    seed_nodes = _pick_seeds_by_pagerank(groups, pr, counts)
    seeds_df = coords.loc[seed_nodes].copy()
    seeds_df["cluster_id"] = [hard_labels[n] for n in seed_nodes]

    X = coords[["x", "y"]].values
    S = seeds_df[["x", "y"]].values
    D = cdist(X, S)
    closest = D.argmin(axis=1)
    dist_min = D[np.arange(len(X)), closest]

    assign_df = coords.copy()
    assign_df["seed_node"] = [seed_nodes[i] for i in closest]
    assign_df["seed_cluster"] = [hard_labels[s] for s in assign_df["seed_node"]]
    assign_df["dist_to_seed"] = dist_min
    q1, q2 = np.quantile(dist_min, [0.33, 0.66])

    def _layer(d: float) -> str:
        if d <= q1:
            return "core"
        if d <= q2:
            return "mid"
        return "outer"

    assign_df["layer"] = assign_df["dist_to_seed"].apply(_layer)

    soft_cols = _soft_membership_attrs(G)
    attrs = {n: dict(data) for n, data in G.nodes(data=True)}
    assign_df["bridge_entropy"] = assign_df.apply(
        lambda row: _bridge_entropy(row, attrs, soft_cols), axis=1
    )

    csv_path = out_prefix.with_suffix(".csv")
    assign_df.reset_index().rename(columns={"index": "node"}).to_csv(csv_path, index=False)

    for n, row in assign_df.iterrows():
        G.nodes[n]["voronoi_seed"] = str(row["seed_node"])
        G.nodes[n]["cluster_id"] = int(row["seed_cluster"])
        G.nodes[n]["layer"] = str(row["layer"])
        G.nodes[n]["dist_to_seed"] = float(row["dist_to_seed"])
        if not math.isnan(row["bridge_entropy"]):
            G.nodes[n]["bridge_entropy"] = float(row["bridge_entropy"])

    gexf_out = out_prefix.with_suffix(".gexf")
    nx.write_gexf(G, gexf_out)
    print(f"Saved {csv_path} and {gexf_out}")


def build_axial_groups(centroids_df: pd.DataFrame, axial_groups: int) -> Dict[int, List[str]]:
    coords = centroids_df[["x", "y"]].values
    if "unit_id" in centroids_df.columns:
        unit_ids = centroids_df["unit_id"].tolist()
    else:
        unit_ids = centroids_df.index.tolist()
    n_clusters = len(unit_ids)
    if n_clusters == 0:
        return {}
    if n_clusters == 1:
        return {1: [unit_ids[0]]}
    axial_groups = max(1, min(axial_groups, n_clusters))
    Z = linkage(coords, method="ward")
    labels = fcluster(Z, t=axial_groups, criterion="maxclust")
    groups: Dict[int, List[str]] = {}
    for label, unit in zip(labels, unit_ids):
        groups.setdefault(int(label), []).append(str(unit))
    return {ax_id: sorted(comms, key=str) for ax_id, comms in sorted(groups.items())}


# ----- CLI -----


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    diag = sub.add_parser(
        "diagram",
        help="Reproduce the notebook Voronoi chart using node2vec + UMAP.",
    )
    diag.add_argument("graph", type=Path, help="Input GEXF graph.")
    diag.add_argument(
        "--seed-count",
        type=int,
        default=10,
        help="Number of top PageRank nodes to highlight as seeds.",
    )
    diag.add_argument(
        "--output",
        type=Path,
        default=Path("voronoi_diagram.png"),
        help="PNG path for the generated figure.",
    )
    diag.add_argument(
        "--coords-out",
        type=Path,
        default=None,
        help="Optional CSV to store coordinates + layers.",
    )
    diag.add_argument(
        "--memberships",
        type=Path,
        default=None,
        help="CSV with fuzzy Louvain memberships (must include node + hard_comm).",
    )
    diag.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers for node2vec random walks (1 avoids loky issues).",
    )

    layers = sub.add_parser(
        "layers",
        help="Auto-pick seeds per community when hard_comm + p_comm_* attrs exist.",
    )
    layers.add_argument("graph", type=Path, help="Input GEXF graph.")
    layers.add_argument(
        "--out-prefix",
        type=Path,
        default=Path("voronoi_layers"),
        help="Prefix used for CSV and GEXF outputs.",
    )
    layers.add_argument(
        "--hard-attr",
        type=str,
        default="hard_comm",
        help="Node attribute with fuzzy Louvain hard assignment.",
    )
    layers.add_argument(
        "--nodes-per-seed",
        type=int,
        default=250,
        help="Target number of nodes represented by each seed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "diagram":
        G = _load_graph(args.graph)
        memberships = load_memberships_csv(args.memberships) if args.memberships else None
        coords = run_node2vec_umap(G, workers=args.workers)
        if memberships is not None:
            coords = attach_hard_labels(coords, memberships)
            centroids_df = compute_centroids(coords)
            coords = assign_layers_per_comm(coords, centroids_df)
            vor = Voronoi(coords[["x", "y"]].values)
            plot_louvain_voronoi(vor, coords, centroids_df, args.output)
            annotated = coords
        else:
            seeds = top_pagerank_seeds(G, coords, count=args.seed_count)
            plot_voronoi(coords, seeds, args.output)
            annotated = annotate_layers(coords, seeds)
        if args.coords_out:
            save_coords(annotated, args.coords_out)
        print(f"Voronoi diagram stored at {args.output}")
    elif args.command == "layers":
        run_auto_layers(
            args.graph,
            args.out_prefix,
            hard_attr=args.hard_attr,
            nodes_per_seed=args.nodes_per_seed,
        )
    else:
        raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
