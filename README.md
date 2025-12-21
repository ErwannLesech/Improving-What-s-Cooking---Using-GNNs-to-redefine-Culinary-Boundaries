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

## Key Features

### Data Processing
- Ingredient-only graph extraction from FlavorGraph
- Feature normalization (z-score standardization)
- Train/validation/test split (65/17.5/17.5)
- Negative sampling for link prediction

### Training Pipeline
- Binary Cross-Entropy loss with logits
- Adam optimizer with gradient clipping (max_norm=1.0)
- 150 training epochs per model
- ROC-AUC metric for evaluation

### Advanced Techniques
- Batch normalization for training stability
- Dropout regularization to prevent overfitting
- Edge weight incorporation for relationship strength
- Negative sampling for unobserved link generation

## Getting Started

### Prerequisites

```bash
Python >= 3.8
PyTorch >= 1.12.0
PyTorch Geometric
pandas
numpy
scikit-learn
matplotlib
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/ErwannLesech/Improving-What-s-Cooking---Using-GNNs-to-redefine-Culinary-Boundaries.git
cd Improving-What-s-Cooking---Using-GNNs-to-redefine-Culinary-Boundaries
```

2. Install PyTorch Geometric dependencies:
```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}.html
pip install pyg-lib -f https://data.pyg.org/whl/nightly/torch-${TORCH}.html
pip install git+https://github.com/pyg-team/pytorch_geometric.git
```

3. Install additional requirements:
```bash
pip install pandas numpy scikit-learn matplotlib
```

## Usage

### Running the Complete Pipeline

Open and execute the Jupyter notebooks in the following order:

1. **Initial_implementation_analysis.ipynb**: Baseline GCN and GraphSAGE implementation
2. **add_GAT_and_GIN.ipynb**: Extended implementation with GAT and GIN models
3. **Link_Prediction_on_Heterogenous_Graphs_with_PyG.ipynb**: Heterogeneous graph experiments

Each notebook contains:
- Environment setup and library imports
- Data loading and preprocessing
- Model definition and training
- Evaluation and visualization
- Comparative analysis

### Training a Model

All models follow a similar training loop:

```python
for epoch in range(1, epochs + 1):
    loss = train()
    val_auc = test(val_data)
    test_auc = test(test_data)
    
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        final_test_auc = test_auc
```

Models are evaluated using ROC-AUC score, which measures the ability to distinguish compatible ingredient pairs from incompatible ones.

## Results

The models demonstrate varying levels of performance in predicting ingredient compatibility:

- **GCN**: Strong baseline performance with spectral convolutions
- **GraphSAGE**: Improved generalization through sampling
- **GAT**: Enhanced performance via attention mechanisms
- **GIN**: Maximum expressive power for complex patterns

Key observations:
- All models benefit significantly from feature normalization
- Gradient clipping prevents training instability
- Attention mechanisms (GAT) provide interpretable relationship weights
- GIN's higher expressiveness captures subtle ingredient interactions

## Future Improvements

### Model Architecture Enhancements

1. **Hierarchical Graph Neural Networks**
   - Multi-scale graph representations
   - Ingredient category hierarchies
   - Recipe-level aggregation

2. **Heterogeneous Graph Learning**
   - Incorporate ingredient types (vegetables, proteins, spices)
   - Add recipe nodes and compound nodes
   - Use relation-specific message passing (R-GCN, HGT)

3. **Temporal Dynamics**
   - Seasonal ingredient availability
   - Cooking process sequences
   - Temporal attention mechanisms

### Feature Engineering

4. **Rich Node Features**
   - Chemical compound composition
   - Nutritional information (calories, vitamins, minerals)
   - Flavor profiles (sweet, salty, umami, bitter, sour)
   - Texture and cooking properties
   - Geographic origin and cultural context

5. **Edge Features Integration**
   - Use compatibility scores as edge weights in convolutions
   - Edge attribute neural networks
   - Multi-dimensional relationship types

6. **External Knowledge Integration**
   - Recipe databases (cooking methods, cuisines)
   - Food pairing theory principles
   - Molecular gastronomy insights

### Training and Optimization

7. **Advanced Training Strategies**
   - Curriculum learning (easy to hard pairings)
   - Self-supervised pre-training on graph structure
   - Contrastive learning for embedding quality
   - Meta-learning for few-shot ingredient discovery

8. **Hyperparameter Optimization**
   - Automated architecture search (NAS)
   - Learning rate scheduling
   - Adaptive dropout rates
   - Layer-wise learning rates

9. **Ensemble Methods**
   - Model averaging across architectures
   - Stacking different GNN types
   - Uncertainty quantification

### Evaluation and Interpretability

10. **Comprehensive Evaluation**
    - Cross-cuisine validation
    - Out-of-distribution ingredient testing
    - Human expert evaluation studies
    - A/B testing in recipe recommendation systems

11. **Explainability Tools**
    - GNNExplainer for important subgraphs
    - Attention weight visualization
    - Counterfactual explanations
    - Feature importance analysis

### Applications and Deployment

12. **Interactive Recommendation System**
    - Real-time ingredient suggestions
    - Constraint-based search (dietary restrictions, allergies)
    - Recipe generation from available ingredients
    - Novelty-aware recommendations

13. **Multi-task Learning**
    - Joint prediction of compatibility and taste profiles
    - Recipe difficulty estimation
    - Nutritional balance optimization
    - Cost-aware meal planning

14. **Domain Adaptation**
    - Transfer learning across cuisines
    - Adaptation to personal taste preferences
    - Regional ingredient substitution
    - Fusion cuisine exploration

### Scalability and Efficiency

15. **Production Optimization**
    - Model quantization and pruning
    - Graph sampling strategies for large-scale deployment
    - Distributed training on multiple GPUs
    - Efficient inference with approximate methods

16. **Data Augmentation**
    - Graph structure augmentation
    - Synthetic edge generation
    - Node feature perturbation
    - Cross-lingual recipe data integration

### Theoretical Advances

17. **Advanced GNN Architectures**
    - Graph Transformers
    - Equivariant Graph Neural Networks
    - Higher-order message passing
    - Continuous graph neural networks

18. **Causal Reasoning**
    - Causal inference for ingredient effects
    - Counterfactual ingredient substitution
    - Treatment effect estimation in recipes

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Maintainer**: Erwann Lesech  
**Institution**: EPITA  
**Year**: 2025
