This directory contains the knowledge graph data files required to train and evaluate ProtHGT, for both the GO term prediction and EC number prediction tasks.

## Downloading the Data

The data files are hosted on Hugging Face: [HUBioDataLab/ProtHGT](https://huggingface.co/datasets/HUBioDataLab/ProtHGT), under `knowledge_graphs/`. Download the files for your task and place them in the matching local directory so that `train.py` / `predict.py` can find them:

- GO term prediction → `knowledge_graphs/go/` ➜ place in `data/go/`
- EC number prediction → `knowledge_graphs/ec/` ➜ place in `data/ec/`

The large `.pt` files are not stored in git (see `.gitignore`); download them here before running.

## GO term prediction — `data/go/`
```
├── prothgt-kg.pt                      # Full knowledge graph (default TAPE embeddings); covers all train+val+test proteins.
├── prothgt-train-graph.pt             # Training set (80% of the full KG).
├── prothgt-val-graph.pt               # Validation set (10%).
├── prothgt-test-graph.pt              # Test set (10%).
└── alternative_protein_embeddings/    # Alternative protein representations (ESM-2, ProtT5).
    ├── esm2/
    │   └── prothgt-esm2-{kg,train-graph,val-graph,test-graph}.pt
    └── prott5/
        └── prothgt-prott5-{kg,train-graph,val-graph,test-graph}.pt
```

## EC number prediction — `data/ec/`
```
├── prothgt-ec-kg.pt                   # Full EC knowledge graph (default TAPE embeddings); covers all train+val+test proteins.
├── prothgt-ec-train-graph.pt          # Training set.
├── prothgt-ec-val-graph.pt            # Validation set.
├── prothgt-ec-test-graph.pt           # Test set.
└── alternative_protein_embeddings/    # Alternative protein representations (ESM-2, ProtT5).
    ├── esm2/
    │   └── prothgt-ec-esm2-{kg,train-graph,val-graph,test-graph}.pt
    └── prott5/
        └── prothgt-ec-prott5-{kg,train-graph,val-graph,test-graph}.pt
```

Once downloaded, place the files inside the matching `data/go/` or `data/ec/` directory so that `train.py` and `predict.py` can access them.
