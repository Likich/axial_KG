#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install node2vec')


# In[11]:


from node2vec import Node2Vec
import networkx as nx
import pandas as pd

G = nx.read_gexf("qlora_outputs/graph_popular_only_qwen_3000_.gexf")
node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)
model = node2vec.fit(window=10, min_count=1)
emb = pd.DataFrame([model.wv[n] for n in G.nodes()], index=list(G.nodes()))


# In[12]:


from umap import UMAP
um = UMAP(n_neighbors=20, min_dist=0.3, metric='cosine', random_state=42)
coords = pd.DataFrame(um.fit_transform(emb), index=emb.index, columns=['x','y'])


# In[13]:


import numpy as np

pr = nx.pagerank(G)
top_nodes = sorted(pr, key=pr.get, reverse=True)[:10]  # for example
seeds = coords.loc[top_nodes]


# In[14]:


from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt

points = coords[['x','y']].values
vor = Voronoi(points)

fig, ax = plt.subplots(figsize=(8,8))
voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='gray', line_alpha=0.3)
ax.scatter(coords.x, coords.y, s=5, color='black')
ax.scatter(seeds.x, seeds.y, s=80, color='red')
plt.show()


# In[15]:


from scipy.spatial.distance import cdist

seed_coords = seeds[['x','y']].values
dist = cdist(coords[['x','y']], seed_coords)
assign = dist.argmin(axis=1)
coords['seed'] = [seeds.index[i] for i in assign]
coords['dist_to_seed'] = dist.min(axis=1)


# In[16]:


coords['layer'] = pd.qcut(coords['dist_to_seed'], q=3, labels=['core','mid','outer'])


# In[17]:


import seaborn as sns

sns.scatterplot(x='x', y='y', hue='layer', data=coords, s=10, alpha=0.6)
sns.scatterplot(x='x', y='y', data=seeds, color='red', s=100, marker='X')


# In[ ]:


# Define K automatically


# In[21]:


import numpy as np, pandas as pd, networkx as nx
from collections import defaultdict
from scipy.spatial.distance import cdist

GEXF = "fuzzy_louvain_overlapping_cooc_all.gexf"   # your big cooc graph
OUT_PREFIX = "voronoi_layers_cooc"

# -----------------------------
# 1) Load graph + attributes
# -----------------------------
G = nx.read_gexf(GEXF)

# coords: use existing x,y if present, else compute a layout
has_xy = all(("x" in G.nodes[n] and "y" in G.nodes[n]) for n in G.nodes)
if not has_xy:
    pos = nx.spring_layout(G, weight="weight", dim=2, seed=42)
    for n,(x,y) in pos.items():
        G.nodes[n]['x'], G.nodes[n]['y'] = float(x), float(y)

nodes = list(G.nodes())
coords = pd.DataFrame(
    [(n, float(G.nodes[n]['x']), float(G.nodes[n]['y'])) for n in nodes],
    columns=["node","x","y"]
).set_index("node")

# hard labels from fuzzy louvain
hard_labels = {n: int(G.nodes[n].get("hard_comm", -1)) for n in nodes}
if any(v < 0 for v in hard_labels.values()):
    raise ValueError("Some nodes are missing 'hard_comm' in the GEXF.")

# pull soft memberships if you want later: p_comm_0..K-1
soft_cols = [c for c in next(iter(G.nodes(data=True)))[1].keys() if str(c).startswith("p_comm_")]
K = len(soft_cols)

# PageRank for “importance” inside a community
pr = nx.pagerank(G, weight="weight")

# -----------------------------
# 2) Auto-pick seeds (centroids)
# -----------------------------
# policy: at least 1 seed per community; add 1 extra seed for every N nodes
N_PER_SEED = 250   # tune: smaller -> more seeds
by_comm = defaultdict(list)
for n in nodes:
    by_comm[hard_labels[n]].append(n)

seed_nodes = []
for c, members in by_comm.items():
    m = len(members)
    n_seeds = max(1, int(np.ceil(m / N_PER_SEED)))
    # pick the top-n_seeds by PR in this community (no duplicates)
    top = sorted(members, key=lambda n: pr[n], reverse=True)[:n_seeds]
    seed_nodes.extend(top)

seeds_df = coords.loc[seed_nodes].copy()  # centroid list
seeds_df["cluster_id"] = [hard_labels[n] for n in seed_nodes]

# -----------------------------
# 3) Assign each node to nearest seed + make shells
# -----------------------------
X = coords[["x","y"]].values
S = seeds_df[["x","y"]].values

D = cdist(X, S)                 # distances node -> seed
closest = D.argmin(axis=1)      # index of nearest seed
dist_min = D[np.arange(len(X)), closest]

# attach assignment
assign_df = coords.copy()
assign_df["seed_node"] = [seed_nodes[i] for i in closest]
assign_df["seed_cluster"] = [hard_labels[s] for s in assign_df["seed_node"]]
assign_df["dist_to_seed"] = dist_min

# shell labels by distance quantiles (per seed or globally)
# global quantiles are simple and work well visually:
q1, q2 = np.quantile(dist_min, [0.33, 0.66])
def layer_of(d):
    if d <= q1: return "core"
    if d <= q2: return "mid"
    return "outer"
assign_df["layer"] = assign_df["dist_to_seed"].apply(layer_of)

# optional: a "bridge score" from soft memberships (entropy)
def entropy(row):
    vals = np.array([float(G.nodes[row.name].get(f"p_comm_{i}", 0.0)) for i in range(K)], dtype=float)
    vals = vals / (vals.sum() + 1e-12)
    nz = vals[vals > 0]
    return float(-(nz * np.log(nz)).sum())
assign_df["bridge_entropy"] = assign_df.apply(entropy, axis=1)

# -----------------------------
# 4) Exports
# -----------------------------
# CSV for inspection
assign_df.reset_index().rename(columns={"index":"node"}).to_csv(f"{OUT_PREFIX}.csv", index=False)

# add attributes back to graph and export GEXF for Gephi
for n, r in assign_df.iterrows():
    G.nodes[n]["voronoi_seed"]  = str(r["seed_node"])
    G.nodes[n]["cluster_id"]    = int(r["seed_cluster"])
    G.nodes[n]["layer"]         = str(r["layer"])
    G.nodes[n]["dist_to_seed"]  = float(r["dist_to_seed"])
    G.nodes[n]["bridge_entropy"]= float(r["bridge_entropy"])

nx.write_gexf(G, f"{OUT_PREFIX}.gexf")

print("Saved:", f"{OUT_PREFIX}.csv", f"{OUT_PREFIX}.gexf")

# -----------------------------
# 5) Quick summaries you can print
# -----------------------------
# top “core” reps per cluster (closest to seed)
tops = (assign_df
        .sort_values(["seed_cluster","dist_to_seed"])
        .groupby("seed_cluster")
        .head(10)
        .reset_index())
print("Top exemplars per cluster:\n", tops[["seed_cluster"]].assign(entity=tops.index, dist=tops["dist_to_seed"]).head(20))

# most “bridgy” nodes overall
bridges = assign_df.sort_values("bridge_entropy", ascending=False).head(25)
print("\nMost cross-cluster (high-entropy) nodes:\n",
      bridges.reset_index()[["node","bridge_entropy","layer","seed_cluster"]].head(10))


# In[22]:


import matplotlib.pyplot as plt
import seaborn as sns

# merge seeds for highlighting
plot_df = assign_df.reset_index().rename(columns={'index': 'node'})

fig, ax = plt.subplots(figsize=(10, 8))
sns.scatterplot(
    data=plot_df,
    x='x', y='y',
    hue='layer',
    alpha=0.7,
    s=15,
    ax=ax,
    palette={'core':'#e63946','mid':'#457b9d','outer':'#a8dadc'}
)

# overlay seeds (centroids)
sns.scatterplot(
    data=seeds_df,
    x='x', y='y',
    s=180,
    color='gold',
    edgecolor='black',
    marker='*',
    ax=ax,
    label='centroids'
)

# optional: label some centroids
for node, row in seeds_df.iterrows():
    ax.text(row.x, row.y, str(node)[:15], fontsize=8, weight='bold', color='black')

ax.set_title("Voronoi-style Concept Regions with Hierarchical Layers", fontsize=13)
ax.set_xlabel("Embedding X")
ax.set_ylabel("Embedding Y")
ax.legend()
plt.show()


# In[23]:


fig, ax = plt.subplots(figsize=(10,8))
sns.scatterplot(
    data=plot_df,
    x='x', y='y',
    hue='seed_cluster',   # color by community
    style='layer',        # shape by layer
    alpha=0.8,
    s=30,
    ax=ax,
)
sns.scatterplot(
    data=seeds_df,
    x='x', y='y',
    s=200,
    color='gold',
    marker='*',
    edgecolor='black',
    ax=ax,
)
ax.set_title("Community + Hierarchical Voronoi Layers")
plt.show()


# In[26]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import Voronoi

# assume you already have:
# coords      : DataFrame with columns ['x','y'], index=node
# seeds_df    : DataFrame of centroids with ['x','y']
# assign_df   : coords + ['layer', 'seed'] etc.


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite Voronoi regions in a 2D diagram
    to finite regions clipped to a bounding box.

    Returns:
        regions: list of lists of vertex indices for each region
        vertices: ndarray of new vertices
    """
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max() * 2

    # map ridge points to ridges
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # reconstruct each region
    for p, region_idx in enumerate(vor.point_region):
        vertices = vor.regions[region_idx]

        if all(v >= 0 for v in vertices):
            # already finite
            new_regions.append(vertices)
            continue

        # reconstruct infinite region
        ridges = all_ridges[p]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge
                continue

            # compute the missing endpoint
            t = vor.points[p2] - vor.points[p]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_vertices.append(far_point.tolist())
            new_region.append(len(new_vertices) - 1)

        # order region vertices clockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = [v for _, v in sorted(zip(angles, new_region))]
        new_regions.append(new_region)

    return new_regions, np.asarray(new_vertices)


# In[27]:


# build Voronoi from centroid coordinates
seed_points = seeds_df[['x','y']].to_numpy()
vor = Voronoi(seed_points)

regions, vertices = voronoi_finite_polygons_2d(vor)

fig, ax = plt.subplots(figsize=(10, 8))

# 1) filled Voronoi cells (light)
for region in regions:
    polygon = vertices[region]
    ax.fill(
        polygon[:, 0],
        polygon[:, 1],
        alpha=0.06,
        color='gray',
        edgecolor='lightgray',
        linewidth=0.5,
        zorder=0,
    )

# 2) points colored by layer
plot_df = assign_df.reset_index().rename(columns={'index': 'node'})
sns.scatterplot(
    data=plot_df,
    x='x', y='y',
    hue='layer',
    s=15,
    alpha=0.8,
    ax=ax,
    palette={'core': '#e63946', 'mid': '#457b9d', 'outer': '#a8dadc'}
)

# 3) centroids
sns.scatterplot(
    data=seeds_df,
    x='x', y='y',
    s=200,
    color='gold',
    edgecolor='black',
    marker='*',
    ax=ax,
    label='centroid'
)

ax.set_title("Voronoi cells clipped to data region\n+ core/mid/outer layers around auto-centroids")
ax.set_xlabel("Embedding X")
ax.set_ylabel("Embedding Y")
ax.legend()
plt.tight_layout()
plt.show()


# In[ ]:


# I tried so hard and got so far


# In[30]:


import networkx as nx
import pandas as pd

# 1. Read co-occurrence graph
G = nx.read_gexf("qlora_outputs/graph_popular_only_qwen_3000_.gexf")

# 2. Load fuzzy Louvain memberships
memb = pd.read_csv("fuzzy_louvain_overlapping_cooc_soft_memberships_all.csv")
memb = memb.set_index("node")
hard_labels = memb["hard_comm"].astype(int).to_dict()

print("Communities:", sorted(memb["hard_comm"].unique()))


# In[31]:


from node2vec import Node2Vec

node2vec = Node2Vec(
    G, dimensions=64, walk_length=30, num_walks=200, workers=4
)
model = node2vec.fit(window=10, min_count=1)

# Embeddings matrix
emb = pd.DataFrame(
    [model.wv[n] for n in G.nodes()],
    index=list(G.nodes())
)
emb.head()


# In[32]:


from umap import UMAP

um = UMAP(n_neighbors=20, min_dist=0.3, metric='cosine', random_state=42)
coords = pd.DataFrame(
    um.fit_transform(emb),
    index=emb.index,
    columns=['x','y']
)
coords.head()


# In[33]:


coords["hard_comm"] = coords.index.map(hard_labels)
coords = coords.dropna(subset=["hard_comm"])
coords["hard_comm"] = coords["hard_comm"].astype(int)
coords.head()


# In[34]:


import numpy as np

centroids = []

for c in sorted(memb["hard_comm"].unique()):
    nodes_c = coords[coords["hard_comm"] == c]

    if len(nodes_c) == 0:
        continue

    # geometric centroid in embedding space
    mu = nodes_c[["x","y"]].mean()

    # pick actual node closest to geometric center
    idx = ((nodes_c[["x","y"]] - mu)**2).sum(axis=1).idxmin()

    centroids.append({
        "comm": c,
        "node": idx,
        "x": coords.loc[idx, "x"],
        "y": coords.loc[idx, "y"],
    })

centroids_df = pd.DataFrame(centroids).set_index("node")
centroids_df


# In[36]:


from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt
import seaborn as sns

# Build Voronoi on ALL points
points = coords[["x","y"]].values
vor = Voronoi(points)

# Distance from each node to each centroid
seed_coords = centroids_df[["x","y"]].values
D = cdist(coords[["x","y"]], seed_coords)

# closest centroid community per node
nearest_idx = D.argmin(axis=1)
coords["nearest_comm"] = [
    centroids_df["comm"].iloc[i] for i in nearest_idx
]
coords["dist_to_seed"] = D.min(axis=1)

coords["layer"] = "outer"


# In[37]:


for c in sorted(coords["nearest_comm"].unique()):
    grp = coords[coords["nearest_comm"] == c]

    if len(grp) < 3:
        continue

    q1, q2 = grp["dist_to_seed"].quantile([1/3, 2/3])

    coords.loc[grp.index[grp["dist_to_seed"] <= q1], "layer"] = "core"
    coords.loc[grp.index[
        (grp["dist_to_seed"] > q1) &
        (grp["dist_to_seed"] <= q2)
    ], "layer"] = "mid"


# In[38]:


from scipy.spatial import voronoi_plot_2d

fig, ax = plt.subplots(figsize=(10,10))

voronoi_plot_2d(
    vor, ax=ax,
    show_vertices=False,
    line_colors="lightgray",
    line_alpha=0.25
)

palette = {"core":"#e63946", "mid":"#457b9d", "outer":"#a8dadc"}

sns.scatterplot(
    data=coords, x="x", y="y",
    hue="layer",
    palette=palette,
    alpha=0.7,
    s=10,
    ax=ax
)

# centroids as large stars
ax.scatter(
    centroids_df["x"], centroids_df["y"],
    s=250, marker="*", edgecolors="black",
    c=centroids_df["comm"], cmap="tab10"
)

plt.title("Voronoi Tessellation Around Louvain Cluster Centroids\n(core/mid/outer shells)")
plt.show()


# In[41]:


# plt.figure(figsize=(10,10))

# sns.scatterplot(
#     data=coords,
#     x="x", y="y",
#     hue="layer",          # THIS makes colors appear
#     palette={
#         "core": "red",
#         "mid": "orange",
#         "outer": "lightblue"
#     },
#     s=12,
#     alpha=0.7
# )

# # Plot centroids
# plt.scatter(
#     centroids_df["x"],
#     centroids_df["y"],
#     s=280,
#     marker="*",
#     edgecolors="black",
#     linewidths=0.8,
#     color="yellow",
#     label="centroid"
# )

# # Overlay Voronoi lines
# plt.triplot(points[:,0], points[:,1], vor.ridge_vertices, color="gray", linewidth=0.3, alpha=0.3)

# plt.legend()
# plt.title("Voronoi Tessellation Around Louvain Cluster Centroids\n(core/mid/outer shells)")
# plt.show()


# In[42]:


from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import seaborn as sns

# vor already computed above

fig, ax = plt.subplots(figsize=(8,8))

# 1) Voronoi lines only (no points)
voronoi_plot_2d(
    vor,
    ax=ax,
    show_vertices=False,
    show_points=False,
    line_colors="lightgray",
    line_alpha=0.3,
    line_width=0.5,
)

# 2) Your layered points
sns.scatterplot(
    data=coords,
    x="x", y="y",
    hue="layer",
    palette={"core":"red", "mid":"orange", "outer":"lightblue"},
    s=12,
    alpha=0.7,
    ax=ax,
)

# 3) Centroids
ax.scatter(
    centroids_df["x"], centroids_df["y"],
    s=280, marker="*",
    edgecolors="black", linewidths=0.8,
    color="yellow", label="centroid"
)

ax.set_title("Voronoi Tessellation Around Louvain Cluster Centroids\n(core/mid/outer shells)")
ax.legend(title="layer")
plt.show()


# # Now open/axial coding

# In[43]:


# coords: DataFrame indexed by entity string
# columns: ['x','y','nearest_comm','layer', ...]
coords.head()


# In[44]:


import networkx as nx
import pandas as pd

G = nx.read_gexf("qlora_outputs/graph_popular_only_qwen_3000_.gexf")
pr = nx.pagerank(G)

coords['pagerank'] = coords.index.map(pr).fillna(0.0)

def sample_cluster_entities(coords, comm_id,
                            n_core=40, n_mid=25, n_outer=15):
    sub = coords[coords['nearest_comm'] == comm_id].copy()

    def pick(layer, n):
        layer_df = sub[sub['layer'] == layer].sort_values(
            'pagerank', ascending=False
        )
        return layer_df.index.tolist()[:n]

    core = pick('core', n_core)
    mid  = pick('mid',  n_mid)
    outer = pick('outer', n_outer)

    return core, mid, outer
cluster_ids = sorted(coords['nearest_comm'].unique())
cluster_entities = {}

for c in cluster_ids:
    core, mid, outer = sample_cluster_entities(coords, c)
    cluster_entities[c] = {'core': core, 'mid': mid, 'outer': outer}


# # Open coding with Mixtral

# In[45]:


OPEN_CODE_PROMPT = """
You are helping with qualitative open coding of concepts from interview data.

I will give you a list of entities (phrases) that co-occur in the same semantic region of a graph:

- CORE ENTITIES (most central / prototypical for this region):
{core_list}

- MID ENTITIES (strongly related, but slightly further out):
{mid_list}

- OUTER ENTITIES (more peripheral examples or context):
{outer_list}

TASK:
1. Propose ONE short, human-readable CODE LABEL (3–6 words) that best captures the shared idea.
2. List 3–6 SUBCODES (short phrases) that describe distinct aspects inside this region.
3. For each subcode, mention 2–5 example entities that support it.
4. Write a 2–3 sentence explanation of the overall theme.

Output JSON exactly in this format:

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
"""


# In[51]:


import json, re

def parse_json_from_llm(raw, where=""):
    """
    Try very hard to pull a valid JSON object from an LLM reply.
    Returns a Python dict or None.
    """
    if not raw:
        print(f"[WARN] Empty LLM response for {where}")
        return None

    text = raw.strip()

    # 1) If the model used ```json ... ``` fences, extract inside
    m = re.search(r"```json(.*?)```", text, flags=re.S | re.I)
    if m:
        text = m.group(1).strip()
    else:
        # 2) Otherwise grab the first {...} block
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            text = m.group(0).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"[JSON ERROR] {where}: {e}")
        print("----------- RAW LLM OUTPUT (truncated) -----------")
        print(text[:1000])
        print("--------------------------------------------------")
        return None


# In[53]:


from openai import OpenAI
import json
import os
import re

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
                base_url=os.getenv("OPENAI_API_BASE", "http://10.180.132.23:8188/v1"))
MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"


def open_code_cluster(comm_id, ents, k_examples=20):
    # you already have something like this prompt
    prompt = f"""
You are doing open coding on a cluster of related entities.
Return STRICT JSON only, no extra text.

Cluster ID: {comm_id}

Entities (sample):
{json.dumps(ents[:k_examples], ensure_ascii=False, indent=2)}

Respond in this JSON format:
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

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.3,
        max_tokens=400,
        messages=[
            {"role": "system", "content": "You are a careful qualitative coding assistant. Output ONLY valid JSON."},
            {"role": "user", "content": prompt},
        ],
    )

    raw = (resp.choices[0].message.content or "").strip()
    data = parse_json_from_llm(raw, where=f"open_code_cluster {comm_id}")
    return data



# In[54]:


open_codes = {}

for c in cluster_ids:
    info = open_code_cluster(c, cluster_entities[c])
    open_codes[c] = info

