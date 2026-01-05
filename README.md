# Improving What's Cooking: Using GNNs to Redefine Culinary Boundaries

Advanced Graph Neural Network implementation for culinary ingredient compatibility prediction using the FlavorGraph dataset. This project explores multiple GNN architectures with enriched features and comprehensive evaluation metrics to understand and predict ingredient pairings.

## Overview

This project applies state-of-the-art Graph Neural Networks to analyze ingredient compatibility using FlavorGraph, a culinary knowledge graph. By treating ingredients as nodes and their relationships as edges, we learn embeddings that capture culinary patterns enhanced by chemical compound information and categorical features.

**Key Innovations:**
- Enriched node features (chemical compounds, structural properties, ingredient categories)
- Multiple GNN architectures (GCN, GraphSAGE, GAT, GIN, DeeperGCN)
- Advanced evaluation metrics (Precision@K, Recall@K, MRR, Diversity)
- Recipe generation with diversity analysis

## Project Structure

```
.
├── add_Features_and_metrics.ipynb           # Main notebook with all improvements
├── add_Gat_GIN_Commented.ipynb             # Additional GAT/GIN exploration
├── Link_Prediction_on_Heterogenous_Graphs_with_PyG.ipynb  # Original baseline
├── docs/
│   └── FlavorGraph.pdf                      # Dataset documentation
├── LICENSE
└── README.md
```

## Dataset

**FlavorGraph** - A comprehensive culinary knowledge graph:
- **Nodes**: 1,561 ingredients + chemical compounds + recipes
- **Edges**: Ingredient-ingredient compatibility relationships
- **Features**: Chemical compound associations, ingredient categories, structural properties

## Implemented Models

### Baseline Models

**GCN (Graph Convolutional Network)**: Spectral convolutions with normalized adjacency matrix  
**GraphSAGE**: Inductive learning with neighborhood sampling and mean aggregation

### Advanced Models

**GAT (Graph Attention Network)**: Multi-head attention (4 heads) with batch normalization and dropout  
**GIN (Graph Isomorphism Network)**: Theoretically maximal expressive power with sum aggregation  
**DeeperGCN**: 4-layer architecture with residual connections for capturing complex patterns  
**Enriched Variants**: GCN and GAT with chemical compound and categorical features

## Feature Enrichment

**Structural**: Node degrees, graph topology  
**Chemical**: Compound associations for molecular similarity  
**Categorical**: 7 ingredient categories (meats, seafood, vegetables, fruits, spices, dairy, other)  
**Statistical**: Average edge scores, normalized features

## Evaluation Metrics

**Standard**: ROC-AUC for link prediction quality  
**Ranking**: Precision@K, Recall@K, Hit Rate@K (K=5,10,20)  
**Quality**: MRR (Mean Reciprocal Rank), Diversity@K for category balance

## Results

Enriched models demonstrate significant improvements across all metrics. GAT with enriched features achieves best Precision@10 and diversity scores. DeeperGCN captures indirect ingredient relationships more effectively.

## License

MIT License - see [LICENSE](LICENSE) file.

---

**Author**: Erwann Lesech | **Institution**: EPITA | **Year**: 2025
