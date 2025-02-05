This directory contains the knowledge graph data files required to train and evaluate the ProtHGT model.

## Downloading the Data

You can download the ProtHGT knowledge graph data files from this [Google Drive link](https://drive.google.com/drive/u/0/folders/1VcMcayVnBD82F7xcUzLFNzlEixRSoFSu).

## Available Files
```
├── prothgt-kg.pt                      # The default full knowledge graph containing TAPE embeddings as the initial protein representations.
├── prothgt-train-graph.pt             # Training set (80% of the default full KG).
├── prothgt-val-graph.pt               # Validation set (10% of the default full KG).
├── prothgt-test-graph.pt              # Test set (10% of the default full KG).
└── alternative_protein_embeddings/    # Contains alternative KGs with different protein representations (e.g., APAAC, ESM-2 and ProtT5).
    ├──apaac/
        ├── prothgt-apaac-train-graph.pt
        ├── prothgt-apaac-val-graph.pt
        └── prothgt-apaac-test-graph.pt
    ├──esm2/
    │   ├── prothgt-esm2-train-graph.pt
    │   ├── prothgt-esm2-val-graph.pt
    │   └── prothgt-esm2-test-graph.pt
    └──prot_t5/
        ├── prothgt-prot_t5-train-graph.pt
        ├── prothgt-prot_t5-val-graph.pt
        └── prothgt-prot_t5-test-graph.pt
```

Once downloaded, place the files inside this `data/` directory so that `train.py` and `predict.py` can access them.