from pathlib import Path
import pandas as pd
import numpy as np
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go

# Paths
coords_path = Path('coords_top3.csv')
axial_codes_path = Path('axial_codes.csv')
axial_memberships_path = Path('axial_memberships.csv')
open_codes_path = Path('open_codes_summary.csv')
gexf_path = Path('qlora_outputs/top3_cooc_clusters.gexf')
out_html = Path('diagrams/graph_axial_interactive.html')

# Load data
coords = pd.read_csv(coords_path)
coords = coords.set_index('node')
axial_codes = pd.read_csv(axial_codes_path)
axial_map = pd.read_csv(axial_memberships_path)
open_codes = pd.read_csv(open_codes_path)

# Map clusters to axial labels
label_lookup = axial_codes.set_index('axial_id')['axial_label'].to_dict()
cluster_to_label = {}
for ax_id, unit_id in axial_map.itertuples(index=False):
    base_comm = int(str(unit_id).split('_')[0])
    cluster_to_label[base_comm] = label_lookup.get(ax_id, f'Axial {ax_id}')
coords['axial_label'] = coords['hard_comm'].map(cluster_to_label)

# Map open code labels per (cluster, layer)
open_label_lookup = open_codes.set_index('cluster_id')['code_label'].to_dict()
coords['open_code'] = coords.apply(
    lambda r: open_label_lookup.get(f"{int(r['hard_comm'])}_{r['layer']}", ''), axis=1
)

# Attach degree if available
graph = nx.read_gexf(gexf_path)
degree = dict(graph.degree(weight='weight'))
coords['degree'] = coords.index.map(degree).fillna(0)

# Build Plotly figure
fig = px.scatter(
    coords.reset_index(),
    x='x', y='y',
    color='axial_label',
    hover_data={
        'node': True,
        'axial_label': True,
        'hard_comm': True,
        'layer': True,
        'open_code': True,
        'degree': ':.0f',
        'x': ':.2f',
        'y': ':.2f',
    },
    template='plotly_white',
    color_discrete_sequence=px.colors.qualitative.Set2,
)
fig.update_traces(marker=dict(size=6, opacity=0.8, line=dict(width=0)))
fig.update_layout(
    title='Interactive Axial-Labeled Knowledge Graph (Top-3 Clusters)',
    legend_title='Axial label',
    margin=dict(l=20, r=20, t=60, b=20),
)

# Optionally draw transparent edges for context
edge_x = []
edge_y = []
for u, v in graph.edges():
    if u in coords.index and v in coords.index:
        edge_x.extend([coords.loc[u, 'x'], coords.loc[v, 'x'], None])
        edge_y.extend([coords.loc[u, 'y'], coords.loc[v, 'y'], None])

edge_trace = go.Scatter(
    x=edge_x,
    y=edge_y,
    mode='lines',
    line=dict(color='rgba(0,0,0,0.03)', width=0.4),
    hoverinfo='skip',
    showlegend=False,
)

fig = go.Figure(data=[edge_trace] + list(fig.data), layout=fig.layout)

# Add open-code rings around clusters of entities sharing the same open code
palette = px.colors.qualitative.Pastel
open_groups = coords.reset_index()
open_groups = open_groups[open_groups['open_code'].astype(bool)]
for idx, (oc, grp) in enumerate(open_groups.groupby('open_code')):
    cx, cy = grp[['x', 'y']].mean()
    dists = ((grp[['x', 'y']] - [cx, cy])**2).sum(axis=1).pow(0.5)
    r = max(dists.mean() + dists.std(), 0.15)
    theta = np.linspace(0, 2 * np.pi, 120)
    ring_x = cx + r * np.cos(theta)
    ring_y = cy + r * np.sin(theta)
    fig.add_trace(
        go.Scatter(
            x=ring_x,
            y=ring_y,
            mode='lines',
            line=dict(color=palette[idx % len(palette)], width=2),
            hoverinfo='text',
            hovertext=[oc],
            name=f"Open code: {oc}",
            showlegend=False,
        )
    )

# Save
out_html.parent.mkdir(parents=True, exist_ok=True)
fig.write_html(out_html, include_plotlyjs='cdn')
print(f'Saved interactive HTML → {out_html}')
