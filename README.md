# AxialKG

Utilities for generating Voronoi-based community layouts, running open coding and axial coding over Louvain clusters, and exporting diagrams/CSVs. Key entry points:

- `coding_workflow.py` – CLI to run node2vec+UMAP, assign shells, sample entities, and perform open/axial coding (supports `--axial-by-layer`).
- `voronoi_workflow.py` – shared helpers for Voronoi plotting, coding prompts, and axial grouping.
- `diagrams/` – rendered figures (ignored by git except when explicitly added).

Data/outputs (CSVs, GEXF, rendered PNGs, qlora outputs) are gitignored to keep the repo lightweight.
