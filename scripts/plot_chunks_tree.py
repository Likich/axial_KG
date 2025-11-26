from pathlib import Path
import ast
from collections import Counter
import pandas as pd
import plotly.graph_objects as go

# Paths
chunks_path = Path("Entities_chunk.csv")
coords_path = Path("coords_top3.csv")
open_summary_path = Path("open_codes_summary.csv")
axial_codes_path = Path("axial_codes.csv")
axial_memberships_path = Path("axial_memberships.csv")
out_html = Path("diagrams/codes_chunks_tree.html")

# Load data
chunks = pd.read_csv(chunks_path)
coords = pd.read_csv(coords_path)
coords.index = coords['node'].map(lambda x: str(x).strip().lower())
coords = coords.set_index(coords.index)
open_summary = pd.read_csv(open_summary_path)
axial_codes = pd.read_csv(axial_codes_path)
axial_map = pd.read_csv(axial_memberships_path)

# Maps
axial_lookup = axial_codes.set_index("axial_id")["axial_label"].to_dict()
cluster_to_axial = {}
for ax_id, cluster_id in axial_map.itertuples(index=False):
    cluster_to_axial[str(cluster_id)] = axial_lookup.get(ax_id, f"Axial {ax_id}")

# open codes lookup supports both base cluster_id and cluster_id_layer keys
open_lookup = {}
for cid, code in open_summary.set_index("cluster_id")["code_label"].items():
    cid_str = str(cid)
    open_lookup[cid_str] = code
    for layer in ["core", "mid", "outer"]:
        open_lookup[f"{cid_str}_{layer}"] = code

# Build hierarchy lists for Plotly sunburst/tree
labels = []
parents = []
hover_texts = []
values = []

root_label = "Codes"
labels.append(root_label)
parents.append("")
hover_texts.append("Axial > Open > Chunk")
values.append(0)

# track unique nodes
node_ids = {root_label: 0}

# helpers to add node

def add_node(label, parent, hover, value=1):
    labels.append(label)
    parents.append(parent)
    hover_texts.append(hover)
    values.append(value)
    node_ids[label] = len(labels) - 1

for _, row in chunks.iterrows():
    try:
        ents = ast.literal_eval(row.get("entities_grounded", "[]"))
    except Exception:
        ents = []
    ents = [str(e).strip().lower() for e in ents if str(e).strip()]
    matched = coords.loc[coords.index.intersection(ents)]
    if matched.empty:
        axial_label = "Unmapped"
        open_code = "Unmapped"
        matched_list = []
    else:
        axial_label = matched["hard_comm"].map(lambda c: cluster_to_axial.get(str(int(c)), "Unmapped")).mode().iat[0]
        open_labels = []
        for _, r2 in matched.iterrows():
            layer_norm = str(r2["layer"]).strip().lower()
            cid = int(r2["hard_comm"])
            key = f"{cid}_{layer_norm}"
            open_labels.append(open_lookup.get(key, open_lookup.get(str(cid), "")))
        open_code = Counter([o for o in open_labels if o]).most_common(1)
        open_code = open_code[0][0] if open_code else "Unmapped"
        matched_list = matched.index.tolist()

    axial_node = f"Axial: {axial_label}"
    open_node = f"Open: {open_code}"
    chunk_node = f"Chunk: {row['file']} p{row['page_number']} c{row['chunk_number']}"

    if axial_node not in node_ids:
        add_node(axial_node, root_label, axial_label, 0)
    if open_node not in node_ids:
        add_node(open_node, axial_node, open_code, 0)

    hover = (
        f"<b>Axial:</b> {axial_label}<br><b>Open:</b> {open_code}<br>"
        f"<b>Entities matched:</b> {', '.join(matched_list) if matched_list else 'None'}<br><br>"
        f"{row['chunk']}"
    )
    add_node(chunk_node, open_node, hover, max(len(matched_list), 1))

fig = go.Figure(
    go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        hovertext=hover_texts,
        hoverinfo="text",
        branchvalues="total",
    )
)
fig.update_layout(
    title="Axial → Open → Chunks (Sunburst)",
    margin=dict(l=10, r=10, t=40, b=10),
)

out_html.parent.mkdir(parents=True, exist_ok=True)
fig.write_html(out_html, include_plotlyjs="cdn")
print(f"Saved chunk tree → {out_html}")
