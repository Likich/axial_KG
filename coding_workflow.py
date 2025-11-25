#!/usr/bin/env python3
"""Command-line helper to run open + axial coding on Voronoi communities."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

import networkx as nx

from voronoi_workflow import (
    _load_graph,
    axial_label,
    build_axial_groups,
    build_openai_client,
    open_code_cluster,
    run_node2vec_umap,
    save_axial_codes,
    save_axial_memberships,
    save_coords,
    save_open_codes,
    sample_cluster_entities,
    attach_hard_labels,
    assign_layers_per_comm,
    compute_centroids,
    load_memberships_csv,
)


def build_layer_units(
    cluster_entities: dict,
    coords: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    """Create unit dictionaries + centroids for each (cluster, layer)."""

    units = {}
    centroids = []
    for cid, layers in cluster_entities.items():
        for layer in ("core", "mid", "outer"):
            focus = layers.get(layer, []) or []
            if not focus:
                continue
            unit_id = f"{cid}_{layer}"
            units[unit_id] = {
                "core": focus,
                "mid": layers.get("mid" if layer == "core" else "core", []),
                "outer": layers.get("outer", []),
            }
            mask = (coords["nearest_comm"] == cid) & (coords["layer"] == layer)
            subset = coords.loc[mask, ["x", "y"]]
            if subset.empty:
                continue
            centroids.append(
                {
                    "unit_id": unit_id,
                    "base_comm": cid,
                    "layer": layer,
                    "x": subset["x"].mean(),
                    "y": subset["y"].mean(),
                }
            )
    if not centroids:
        raise ValueError("No layer units available for axial coding.")
    return units, pd.DataFrame(centroids).set_index("unit_id")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph", type=Path, help="Input GEXF graph.")
    parser.add_argument(
        "--memberships",
        type=Path,
        required=True,
        help="CSV with fuzzy Louvain memberships (needs node + hard_comm columns).",
    )
    parser.add_argument(
        "--open-summary-csv",
        type=Path,
        default=Path("open_codes_summary.csv"),
        help="Where to store cluster-level open code summaries.",
    )
    parser.add_argument(
        "--open-subcodes-csv",
        type=Path,
        default=Path("open_codes_subcodes.csv"),
        help="Where to store open-code subcodes (long format).",
    )
    parser.add_argument(
        "--axial-csv",
        type=Path,
        default=Path("axial_codes.csv"),
        help="Where to store axial coding results.",
    )
    parser.add_argument(
        "--axial-memberships",
        type=Path,
        default=Path("axial_memberships.csv"),
        help="Mapping between axial IDs and cluster IDs.",
    )
    parser.add_argument(
        "--coords-out",
        type=Path,
        default=None,
        help="Optional CSV snapshot of coords + layers used for coding.",
    )
    parser.add_argument(
        "--axial-groups",
        type=int,
        default=6,
        help="Target number of axial categories (hierarchical clustering).",
    )
    parser.add_argument(
        "--axial-by-layer",
        action="store_true",
        help="Treat each cluster's core/mid/outer layers as separate units for open + axial coding.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers for node2vec random walks.",
    )
    parser.add_argument("--sample-core", type=int, default=40, help="Core entities per cluster.")
    parser.add_argument("--sample-mid", type=int, default=25, help="Mid entities per cluster.")
    parser.add_argument("--sample-outer", type=int, default=15, help="Outer entities per cluster.")
    parser.add_argument(
        "--prompt-max",
        type=int,
        default=30,
        help="Maximum entities per layer included in prompts.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("LLM_MODEL", "mistralai/Mixtral-8x7B-Instruct-v0.1"),
        help="Model ID for the Mixtral endpoint.",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
        help="Base URL for the OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("OPENAI_API_KEY", ""),
        help="API key for the endpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    G = _load_graph(args.graph)
    memberships = load_memberships_csv(args.memberships)
    coords = run_node2vec_umap(G, workers=args.workers)
    coords = attach_hard_labels(coords, memberships)
    centroids_df = compute_centroids(coords)
    coords = assign_layers_per_comm(coords, centroids_df)
    pr = nx.pagerank(G, weight="weight")
    coords["pagerank"] = coords.index.map(pr).fillna(0.0)
    if args.coords_out:
        save_coords(coords, args.coords_out)

    cluster_ids = sorted(coords["nearest_comm"].unique())
    cluster_entities = {}
    for cid in cluster_ids:
        cluster_entities[cid] = sample_cluster_entities(
            coords,
            cid,
            n_core=args.sample_core,
            n_mid=args.sample_mid,
            n_outer=args.sample_outer,
        )

    if args.axial_by_layer:
        coding_units, axial_centroids = build_layer_units(cluster_entities, coords)
    else:
        coding_units = cluster_entities
        axial_centroids = (
            centroids_df.reset_index()
            .rename(columns={"index": "node"})
            .set_index("comm")
        )

    client = build_openai_client(args.api_key, args.api_base)

    open_codes = {}
    for unit_id in sorted(coding_units.keys(), key=str):
        try:
            open_codes[unit_id] = open_code_cluster(
                client,
                args.model,
                unit_id,
                coding_units[unit_id],
                max_examples=args.prompt_max,
            )
        except Exception as exc:
            print(f"[WARN] Open coding failed for cluster {unit_id}: {exc}")
            open_codes[unit_id] = None
    save_open_codes(open_codes, args.open_summary_csv, args.open_subcodes_csv)

    axial_map = build_axial_groups(axial_centroids, args.axial_groups)
    axial_codes = {}
    for ax_id, comms in axial_map.items():
        try:
            axial_codes[ax_id] = axial_label(
                client,
                args.model,
                ax_id,
                comms,
                open_codes,
                coding_units,
                max_examples=args.prompt_max * 2,
            )
        except Exception as exc:
            print(f"[WARN] Axial coding failed for group {ax_id}: {exc}")
            axial_codes[ax_id] = None
    save_axial_codes(axial_codes, args.axial_csv)
    save_axial_memberships(axial_map, args.axial_memberships)
    print("Open & axial coding complete.")


if __name__ == "__main__":
    main()
