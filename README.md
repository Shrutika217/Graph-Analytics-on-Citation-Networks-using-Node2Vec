# Graph Analytics on Citation Networks using Node2Vec

This repository implements an **unsupervised graph analytics pipeline for citation networks** using **Node2Vec embeddings**, clustering, and **interactive 3D visualization** deployed via **Streamlit**.

The project explores latent structural and semantic relationships between scientific papers without using labels during training.

---

## 🔗 Live Demo & Report

- 🌐 **Interactive 3D Graph (Streamlit App):**  
  👉 https://graph-analytics-on-citation-networks-using-node2vec.streamlit.app/

- 📄 **Project Report (PDF):**  
  👉 https://github.com/Shrutika217/Graph-Analytics-on-Citation-Networks-using-Node2Vec/blob/main/Project_Report.pdf  

---

## 🧠 Method Overview

1. **Citation Graph Construction**
   - Nodes: research papers  
   - Edges: citation relationships (directed)

2. **Node2Vec Embeddings**
   - Random walks + Word2Vec to learn node representations

3. **Dimensionality Reduction**
   - UMAP for compact embedding space

4. **Unsupervised Clustering**
   - KMeans with silhouette-based model selection

5. **Interactive Visualization**
   - 3D force-directed layout (NetworkX)
   - Plotly-based interaction
   - Deployed using Streamlit

---

## 🖥️ Streamlit Application

The Streamlit app allows users to:

- Explore the citation network in **3D**
- Hover over nodes to view metadata (paper ID, degree, subject, cluster)
- Interactively rotate, zoom, and pan the graph
- Visualize communities learned via Node2Vec

---

## 📂 Repository Structure

```text
├── app.py
├── requirements.txt
├── README.md
├── node_cluster_assignments.csv   # optional
├── data/
│   ├── nodes.csv
│   └── edges.csv
```

---

## 📊 Evaluation Metrics (Offline)

- **Silhouette Score**
- **Calinski–Harabasz Index**
- **Davies–Bouldin Index**
- **NMI / ARI** (when ground-truth labels are available)

---

## 🎯 Key Highlights

- Fully unsupervised learning approach  
- Node2Vec-based graph representation learning  
- Interactive 3D citation network visualization  
- Streamlit Cloud deployable  
- Academic and research-oriented implementation  

---

## 👩‍💻 Author

**Shrutika Gupta**  
Graph Analytics & Machine Learning Project  

GitHub: https://github.com/Shrutika217

