This directory contains the knowledge graph data files required to train and evaluate the ProtHGT model.

## Downloading the Data

You can download the ProtHGT knowledge graph data files from this [Google Drive link](https://drive.google.com/drive/u/0/folders/1VcMcayVnBD82F7xcUzLFNzlEixRSoFSu).

## Available Files

| File                          | Description                                  |
|-------------------------------|----------------------------------------------|
| `prothgt-kg.pt`               | The full knowledge graph.                    |
| `prothgt-train-graph.pt`      | Training set (80% of the full KG).          |
| `prothgt-val-graph.pt`        | Validation set (10% of the full KG).        |
| `prothgt-test-graph.pt`       | Test set (10% of the full KG).              |

## Usage

Once downloaded, place the files inside this `data/` directory so that `train.py` and `predict.py` can access them.