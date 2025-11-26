from pathlib import Path
import ast
from collections import Counter, defaultdict
import pandas as pd
import plotly.graph_objects as go

# Paths
chunks_path = Path("Entities_chunk.csv")
coords_path = Path("coords_top3.csv")
open_summary_path = Path("open_codes_summary.csv")
axial_codes_path = Path("axial_codes.csv")
axial_memberships_path = Path("axial_memberships.csv")
out_html = Path("diagrams/codes_chunks_sankey.html")

# Load data
chunks = pd.read_csv(chunks_path)
coords = pd.read_csv(coords_path).set_index("node")
open_summary = pd.read_csv(open_summary_path)
axial_codes = pd.read_csv(axial_codes_path)
axial_map = pd.read_csv(axial_memberships_path)

# Maps
axial_lookup = axial_codes.set_index("axial_id")["axial_label"].to_dict()
cluster_to_axial = {}
for ax_id, cluster_id in axial_map.itertuples(index=False):
    cluster_to_axial[str(cluster_id)] = axial_lookup.get(ax_id, f"Axial {ax_id}")
open_lookup = open_summary.set_index("cluster_id")["code_label"].to_dict()

# Node indexing
node_index: dict[str, int] = {}
node_labels: list[str] = []
node_color: list[str] = []

palette = [
    "#5aa9e6", "#f97306", "#9b59b6", "#2ecc71", "#e74c3c", "#f1c40f",
    "#1abc9c", "#e67e22", "#34495e", "#7f8c8d",
]

def add_node(key: str, label: str, color: str) -> int:
    if key in node_index:
        return node_index[key]
    node_index[key] = len(node_labels)
    node_labels.append(label)
    node_color.append(color)
    return node_index[key]

links_src = []
links_tgt = []
links_val = []
links_hover = []

# Build links: axial -> open -> chunk -> entity
for _, row in chunks.iterrows():
    try:
        ents = ast.literal_eval(row.get("entities_grounded", "[]"))
    except Exception:
        ents = []
    ents = [e.strip() for e in ents if str(e).strip()]
    matched = coords.loc[coords.index.intersection(ents)]
    if matched.empty:
        axial_label = "Unmapped"
        open_code = "Unmapped"
        matched_list = []
        base_comm = None
    else:
        base_comm = matched["hard_comm"].mode().iat[0]
        axial_label = matched["hard_comm"].map(lambda c: cluster_to_axial.get(str(int(c)), "Unmapped")).mode().iat[0]
        open_labels = []
        for _, r2 in matched.iterrows():
            key = f"{int(r2['hard_comm'])}_{r2['layer']}"
            open_labels.append(open_lookup.get(key, ""))
        open_code = Counter([o for o in open_labels if o]).most_common(1)
        open_code = open_code[0][0] if open_code else "Unmapped"
        matched_list = matched.index.tolist()

    axial_key = f"axial::{axial_label}"
    open_key = f"open::{open_code}::{axial_label}"
    chunk_key = f"chunk::{row['file']}:{row['page_number']}:{row['chunk_number']}"
    chunk_text_full = str(row['chunk']).strip().replace('\n', ' ')
    chunk_text = chunk_text_full
    if len(chunk_text) > 120:
        chunk_text = chunk_text[:117] + '...'
    chunk_label = f"{row['file']} p{row['page_number']} #{row['chunk_number']}"

    color = palette[(base_comm or 0) % len(palette)]
    a_idx = add_node(axial_key, axial_label, color)
    o_idx = add_node(open_key, f"{open_code}", color)
    c_idx = add_node(chunk_key, chunk_label, "#bdc3c7")

    links_src.extend([a_idx, o_idx])
    links_tgt.extend([o_idx, c_idx])
    links_val.extend([1, 1])
    links_hover.extend([
        f"Axial → Open: {axial_label} → {open_code}",
        f"Open → Chunk: {open_code} → {chunk_label}<br><br>{chunk_text_full[:240]}" + ("..." if len(chunk_text_full) > 240 else ""),
    ])

    # link chunk to a few entities (limit to 3 for readability)
    for ent in matched_list[:3]:
        e_key = f"ent::{ent}"
        e_idx = add_node(e_key, ent, "#95a5a6")
        links_src.append(c_idx)
        links_tgt.append(e_idx)
        links_val.append(1)
        links_hover.append(f"Chunk → Entity: {ent}")

fig = go.Figure(
    go.Sankey(
        arrangement="snap",
        node=dict(label=node_labels, color=node_color, pad=12, thickness=14),
        link=dict(
            source=links_src,
            target=links_tgt,
            value=links_val,
            hovertemplate="%{customdata}<extra></extra>",
            customdata=links_hover,
            color="rgba(0,0,0,0.2)",
        ),
    )
)
fig.update_layout(
    title="Axial → Open → Chunks → Entities (Sankey)",
    font=dict(size=11),
    margin=dict(l=10, r=10, t=40, b=10),
)

out_html.parent.mkdir(parents=True, exist_ok=True)
fig.write_html(out_html, include_plotlyjs="cdn")
print(f"Saved chunk/entity sankey → {out_html}")
