This directory contains the knowledge graph data files required to train and evaluate the ProtHGT model.

## Downloading the Data

You can download the ProtHGT knowledge graph data files from this [Hugging Face link](https://huggingface.co/datasets/HUBioDataLab/ProtHGT).

## Available Files
```
├── prothgt-kg.pt                      # The default full knowledge graph containing TAPE embeddings as the initial protein representations.
├── prothgt-train-graph.pt             # Training set (80% of the default full KG).
├── prothgt-val-graph.pt               # Validation set (10% of the default full KG).
├── prothgt-test-graph.pt              # Test set (10% of the default full KG).
└── alternative_protein_embeddings/    # Contains alternative KGs with different protein representations (e.g., APAAC, ESM-2 and ProtT5).
    ├──apaac/
        ├── prothgt-apaac-kg.pt
        ├── prothgt-apaac-train-graph.pt
        ├── prothgt-apaac-val-graph.pt
        └── prothgt-apaac-test-graph.pt
    ├──esm2/
        ├── prothgt-esm2-kg.pt
        ├── prothgt-esm2-train-graph.pt
        ├── prothgt-esm2-val-graph.pt
        └── prothgt-esm2-test-graph.pt
    └──prott5/
        ├── prothgt-prott5-kg.pt
        ├── prothgt-prott5-train-graph.pt
        ├── prothgt-prott5-val-graph.pt
        └── prothgt-prott5-test-graph.pt
```

Once downloaded, place the files inside this `data/` directory so that `train.py` and `predict.py` can access them.