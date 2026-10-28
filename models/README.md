This directory holds the trained ProtHGT model weights, for both the GO term prediction and EC number prediction tasks.

## Downloading the Models

The weights are hosted on Hugging Face: [HUBioDataLab/ProtHGT](https://huggingface.co/datasets/HUBioDataLab/ProtHGT), under `models/`. Download the files for your task and place them here:

- GO models → `models/go/` (one model per GO category × embedding)
- EC models → `models/ec/` (one joint hierarchical model per embedding)

Each model is a plain `state_dict` and must be paired with the matching config in `configs/` (GO) or `configs/ec/` (EC) and the matching knowledge graph in `data/`. Be sure to pick the model whose protein embedding type (TAPE / ESM-2 / ProtT5) matches the KG you use.

## GO term prediction — `models/go/`
```
├── prothgt-model-molecular-function.pt      # default TAPE embeddings
├── prothgt-model-biological-process.pt
├── prothgt-model-cellular-component.pt
└── alternative_protein_embeddings/
    ├── esm2/   prothgt-esm2-model-{molecular-function,biological-process,cellular-component}.pt
    └── prott5/ prothgt-prott5-model-{molecular-function,biological-process,cellular-component}.pt
```

## EC number prediction — `models/ec/`
One joint hierarchical model per embedding predicts all four EC levels (L1–L4).
```
├── prothgt-ec-model.pt                       # default TAPE embeddings
└── alternative_protein_embeddings/
    ├── esm2/   prothgt-ec-esm2-model.pt
    └── prott5/ prothgt-ec-prott5-model.pt
```
