# Improving What's Cooking: Using GNNs to Redefine Culinary Boundaries

An advanced implementation of Graph Neural Networks (GNNs) for culinary ingredient compatibility prediction using the FlavorGraph dataset. This project explores multiple GNN architectures to understand and predict ingredient pairings, offering insights into culinary science through deep learning on graphs.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Implemented Models](#implemented-models)
- [Key Features](#key-features)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [License](#license)

## Overview

This project leverages Graph Neural Networks to analyze and predict ingredient compatibility using the FlavorGraph knowledge graph. By treating ingredients as nodes and their relationships as edges, we apply state-of-the-art GNN architectures to learn meaningful embeddings that capture culinary patterns and ingredient synergies.

The implementation compares four different GNN architectures (GCN, GraphSAGE, GAT, and GIN) to understand how different message-passing mechanisms affect the quality of ingredient relationship predictions.

## Project Structure

```
.
├── add_GAT_and_GIN.ipynb                    # Complete implementation with GAT and GIN models
├── Initial_implementation_analysis.ipynb     # Base implementation with GCN and GraphSAGE
├── Link_Prediction_on_Heterogenous_Graphs_with_PyG.ipynb  # Heterogeneous graph exploration
├── docs/
│   └── FlavorGraph.pdf                       # Dataset documentation
├── LICENSE
└── README.md
```

## Dataset

**FlavorGraph** is a culinary knowledge graph containing:

- **Nodes**: Ingredients, recipes, and chemical compounds
- **Edges**: Compatibility relationships with weighted scores
- **Focus**: Ingredient-to-ingredient relationships for link prediction

The dataset captures the complex interactions between ingredients based on their chemical composition and culinary traditions.

## Implemented Models

### 1. Graph Convolutional Network (GCN)

The baseline model using spectral graph convolutions to aggregate neighborhood information.

- **Architecture**: 2-layer GCN (input → 128 → 64 dimensions)
- **Mechanism**: Message passing through normalized adjacency matrix
- **Strengths**: Simple, efficient, and interpretable

### 2. GraphSAGE (Sample and Aggregate)

An inductive learning framework that samples and aggregates features from local neighborhoods.

- **Architecture**: 2-layer SAGE with mean aggregation
- **Mechanism**: Neighborhood sampling with learnable aggregation
- **Strengths**: Scalable to large graphs, generalizes to unseen nodes

### 3. Graph Attention Network (GAT)

Incorporates attention mechanisms to weight the importance of neighboring nodes dynamically.

- **Architecture**: 2-layer GAT with 4 attention heads
- **Mechanism**: Multi-head attention for adaptive neighbor weighting
- **Features**: Batch normalization, dropout (0.3), learning rate 0.001
- **Strengths**: Learns which neighbors are most relevant for each node

### 4. Graph Isomorphism Network (GIN)

Based on the Weisfeiler-Lehman graph isomorphism test, maximizing expressive power.

- **Architecture**: 2-layer GIN with MLPs
- **Mechanism**: Sum aggregation with learnable neural networks
- **Features**: Xavier initialization, batch normalization, dropout (0.3), learning rate 0.0005
- **Strengths**: Theoretically most expressive among message-passing GNNs

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Maintainer**: Erwann Lesech  
**Institution**: EPITA  
**Year**: 2025
