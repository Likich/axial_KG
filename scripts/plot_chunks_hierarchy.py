from pathlib import Path
import ast
from collections import Counter
import pandas as pd
import plotly.express as px

# Paths
chunks_path = Path("Entities_chunk.csv")
coords_path = Path("coords_top3.csv")
open_summary_path = Path("open_codes_summary.csv")
axial_codes_path = Path("axial_codes.csv")
axial_memberships_path = Path("axial_memberships.csv")
out_html = Path("diagrams/codes_chunks_treemap.html")

# Load base data
chunks = pd.read_csv(chunks_path)
coords = pd.read_csv(coords_path).set_index("node")
open_summary = pd.read_csv(open_summary_path)
axial_codes = pd.read_csv(axial_codes_path)
axial_map = pd.read_csv(axial_memberships_path)

# Map cluster -> axial label
axial_lookup = axial_codes.set_index("axial_id")["axial_label"].to_dict()
cluster_to_axial = {}
for ax_id, cluster_id in axial_map.itertuples(index=False):
    cluster_to_axial[str(cluster_id)] = axial_lookup.get(ax_id, f"Axial {ax_id}")

# Map cluster_layer -> open code label
open_lookup = open_summary.set_index("cluster_id")["code_label"].to_dict()

rows = []
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
    else:
        axial_label = matched["hard_comm"].map(lambda c: cluster_to_axial.get(str(int(c)), "Unmapped")).mode().iat[0]
        open_labels = []
        for _, r2 in matched.iterrows():
            key = f"{int(r2['hard_comm'])}_{r2['layer']}"
            open_labels.append(open_lookup.get(key, ""))
        open_code = Counter([o for o in open_labels if o]).most_common(1)
        open_code = open_code[0][0] if open_code else "Unmapped"
        matched_list = matched.index.tolist()

    val = max(len(matched_list), 1)
    hover = (
        f"<b>File:</b> {row['file']} p{row['page_number']} chunk {row['chunk_number']}<br>"
        f"<b>Axial:</b> {axial_label}<br><b>Open:</b> {open_code}<br>"
        f"<b>Entities matched:</b> {', '.join(matched_list) if matched_list else 'None'}<br><br>"
        f"{row['chunk']}"
    )
    rows.append(
        {
            "axial": axial_label,
            "open_code": open_code,
            "chunk_id": f"{row['file']}:{row['page_number']}:{row['chunk_number']}",
            "value": val,
            "hover": hover,
        }
    )

hier_df = pd.DataFrame(rows)

fig = px.treemap(
    hier_df,
    path=[px.Constant("Codes"), "axial", "open_code", "chunk_id"],
    values="value",
    color="axial",
    hover_data={"hover": True, "value": False},
    color_discrete_sequence=px.colors.qualitative.Pastel,
)
fig.update_traces(root_color="lightgray")
fig.update_layout(
    title="Axial → Open → Chunk (entities) Treemap",
    margin=dict(l=10, r=10, t=40, b=10),
)

out_html.parent.mkdir(parents=True, exist_ok=True)
fig.write_html(out_html, include_plotlyjs="cdn")
print(f"Saved chunk treemap → {out_html}")
