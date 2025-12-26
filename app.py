import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import numpy as np
import os
import random

# ==================================================
# Streamlit Page Configuration
# ==================================================
st.set_page_config(
    page_title="Graph Analytics on Citation Networks",
    layout="wide"
)

st.title("3D Interactive Citation Network")
st.caption("Node2Vec Embeddings + Unsupervised Learning")

# ==================================================
# File Paths (NO outputs/ used anywhere)
# ==================================================
NODES_CSV = "data/nodes.csv"
EDGES_CSV = "data/edges.csv"
CLUSTER_CSV = "node_cluster_assignments.csv"  # optional

# ==================================================
# Load Data
# ==================================================
@st.cache_data
def load_data():
    nodes_df = pd.read_csv(NODES_CSV, dtype=str)
    edges_df = pd.read_csv(EDGES_CSV, dtype=str)

    if os.path.exists(CLUSTER_CSV):
        cluster_df = pd.read_csv(CLUSTER_CSV, dtype=str)
    else:
        cluster_df = None

    return nodes_df, edges_df, cluster_df

nodes_df, edges_df, cluster_df = load_data()

# ==================================================
# Build Citation Graph
# ==================================================
@st.cache_data
def build_graph(nodes_df, edges_df):
    G = nx.DiGraph()

    for _, row in nodes_df.iterrows():
        G.add_node(
            row["nodeId"],
            subject=row.get("subject", "Unknown"),
            label=row.get("labels", "Unknown")
        )

    for _, row in edges_df.iterrows():
        G.add_edge(
            row["sourceNodeId"],
            row["targetNodeId"]
        )

    return G

G = build_graph(nodes_df, edges_df)
node_list = list(G.nodes())

# ==================================================
# Graph Summary
# ==================================================
st.markdown(
    f"""
**Graph Summary**
- Nodes: `{G.number_of_nodes()}`
- Edges: `{G.number_of_edges()}`
- Type: Directed citation network
"""
)

# ==================================================
# Cluster Handling (Optional)
# ==================================================
if cluster_df is not None:
    cluster_map = {
        str(row["nodeId"]): int(row["kmeans_cluster"])
        for _, row in cluster_df.iterrows()
    }
    labels = np.array([cluster_map.get(n, -1) for n in node_list])
    show_clusters = True
else:
    labels = np.zeros(len(node_list), dtype=int)
    show_clusters = False
    st.warning("Cluster file not found. Showing graph without clustering.")

# ==================================================
# 3D Graph Layout
# ==================================================
@st.cache_data
def compute_3d_layout(G):
    return nx.spring_layout(
        G.to_undirected(),
        dim=3,
        seed=42,
        k=0.15,
        iterations=80
    )

pos = compute_3d_layout(G)

# ==================================================
# Build Edge Traces (Sampled)
# ==================================================
edge_x, edge_y, edge_z = [], [], []

MAX_EDGES = 3000
edges = list(G.edges())
edges = edges if len(edges) <= MAX_EDGES else random.sample(edges, MAX_EDGES)

for u, v in edges:
    if u in pos and v in pos:
        x0, y0, z0 = pos[u]
        x1, y1, z1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_z += [z0, z1, None]

edge_trace = go.Scatter3d(
    x=edge_x,
    y=edge_y,
    z=edge_z,
    mode="lines",
    line=dict(color="rgba(120,120,120,0.5)", width=1),
    hoverinfo="none"
)

# ==================================================
# Build Node Traces
# ==================================================
node_x, node_y, node_z, hover_texts, sizes = [], [], [], [], []
degrees = dict(G.degree())

for n in node_list:
    if n not in pos:
        continue

    x, y, z = pos[n]
    node_x.append(x)
    node_y.append(y)
    node_z.append(z)

    nd = G.nodes[n]
    hover_texts.append(
        f"Node ID: {n}"
        f"<br>Subject: {nd.get('subject', 'NA')}"
        f"<br>Degree: {degrees.get(n, 0)}"
        f"<br>Cluster: {cluster_map.get(n, 'NA') if show_clusters else 'NA'}"
    )

    sizes.append(6 + min(degrees.get(n, 0), 20))

node_trace = go.Scatter3d(
    x=node_x,
    y=node_y,
    z=node_z,
    mode="markers",
    text=hover_texts,
    hoverinfo="text",
    marker=dict(
        size=sizes,
        color=labels,
        colorscale="Viridis",
        showscale=show_clusters,
        colorbar=dict(title="Cluster ID") if show_clusters else None,
        line=dict(width=0.3, color="black")
    )
)

# ==================================================
# Plotly Figure
# ==================================================
fig = go.Figure(
    data=[edge_trace, node_trace],
    layout=go.Layout(
        height=780,
        margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        ),
        title="3D Citation Network (Node2Vec + Unsupervised Learning)"
    )
)

# ==================================================
# Render in Streamlit
# ==================================================
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
### 🧭 Interaction Guide
- **Rotate:** Left-click  
- **Zoom:** Scroll  
- **Pan:** Right-click  
- **Hover:** View node metadata  
""")
