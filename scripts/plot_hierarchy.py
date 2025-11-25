from pathlib import Path
import pandas as pd
import plotly.express as px

# Paths
axial_codes_path = Path("axial_codes.csv")
axial_memberships_path = Path("axial_memberships.csv")
open_summary_path = Path("open_codes_summary.csv")
open_subcodes_path = Path("open_codes_subcodes.csv")
out_html = Path("diagrams/codes_hierarchy.html")

# Load data
axial_codes = pd.read_csv(axial_codes_path)
axial_map = pd.read_csv(axial_memberships_path)
open_summary = pd.read_csv(open_summary_path)
open_sub = pd.read_csv(open_subcodes_path)

# Map cluster -> axial label, cluster -> open label
axial_lookup = axial_codes.set_index("axial_id")["axial_label"].to_dict()
cluster_to_axial = {}
for ax_id, cluster_id in axial_map.itertuples(index=False):
    cluster_to_axial[str(cluster_id)] = axial_lookup.get(ax_id, f"Axial {ax_id}")

open_lookup = open_summary.set_index("cluster_id")["code_label"].to_dict()

rows = []
for cluster_id, open_label in open_lookup.items():
    axial_label = cluster_to_axial.get(str(cluster_id), "(unmapped)")
    subs = open_sub[open_sub["cluster_id"] == cluster_id]
    if subs.empty:
        # create a leaf from the open code itself
        rows.append(
            {
                "axial": axial_label,
                "open_code": open_label,
                "subcode": open_label,
                "value": 1,
                "hover": open_summary.loc[open_summary["cluster_id"] == cluster_id, "explanation"].iat[0],
            }
        )
        continue
    for _, sub in subs.iterrows():
        examples = sub.get("example_entities", "")
        n_examples = len([e for e in str(examples).split(";") if e.strip()]) or 1
        hover = f"{sub['description']}<br><b>Examples:</b> {examples}"
        rows.append(
            {
                "axial": axial_label,
                "open_code": open_label,
                "subcode": sub.get("subcode_name", "(subcode)"),
                "value": n_examples,
                "hover": hover,
            }
        )

hier_df = pd.DataFrame(rows)

fig = px.treemap(
    hier_df,
    path=[px.Constant("Codes"), "axial", "open_code", "subcode"],
    values="value",
    color="axial",
    hover_data={"hover": True, "value": False},
    color_discrete_sequence=px.colors.qualitative.Set3,
)
fig.update_traces(root_color="lightgray")
fig.update_layout(
    title="Axial → Open → Subcode Hierarchy",
    margin=dict(l=10, r=10, t=40, b=10),
)

out_html.parent.mkdir(parents=True, exist_ok=True)
fig.write_html(out_html, include_plotlyjs="cdn")
print(f"Saved hierarchy treemap → {out_html}")
