import os
import torch
import argparse
import pandas as pd
import yaml
from datetime import datetime
from tqdm.auto import tqdm

from model import ProtHGTHier, LEVELS, LEVEL_TARGET_TYPE
from data_loader import load_hierarchy_masks

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))


def _root_path(*parts: str) -> str:
    return os.path.join(PROJECT_ROOT, *parts)

MODEL_CONFIG_PATHS = {
    'tape':   _root_path('configs', 'ec', 'prothgt-ec-config.yaml'),
    'prott5': _root_path('configs', 'ec', 'alternative_protein_embeddings', 'prott5', 'prothgt-ec-prott5-config.yaml'),
    'esm2':   _root_path('configs', 'ec', 'alternative_protein_embeddings', 'esm2', 'prothgt-ec-esm2-config.yaml'),
}

MODEL_PATHS = {
    'tape':   _root_path('models', 'ec', 'prothgt-ec-model.pt'),
    'prott5': _root_path('models', 'ec', 'alternative_protein_embeddings', 'prott5', 'prothgt-ec-prott5-model.pt'),
    'esm2':   _root_path('models', 'ec', 'alternative_protein_embeddings', 'esm2', 'prothgt-ec-esm2-model.pt'),
}

HETERODATA_PATHS = {
    'tape':   _root_path('data', 'ec', 'prothgt-ec-kg.pt'),
    'prott5': _root_path('data', 'ec', 'alternative_protein_embeddings', 'prott5', 'prothgt-ec-prott5-kg.pt'),
    'esm2':   _root_path('data', 'ec', 'alternative_protein_embeddings', 'esm2', 'prothgt-ec-esm2-kg.pt'),
}

# CLI level name -> internal level key; 'all' predicts every level.
EC_LEVELS = {'all': None, '1': 'l1', '2': 'l2', '3': 'l3', '4': 'l4'}
LEVEL_LABEL = {'l1': 'L1', 'l2': 'L2', 'l3': 'L3', 'l4': 'L4'}
# Which hierarchy-children map feeds each level's candidate set in the cascade.
CHILDREN_KEY = {'l2': 'l1_children', 'l3': 'l2_children', 'l4': 'l3_children'}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protein_ids', type=str, required=True, help='Path to the file containing the protein IDs')
    parser.add_argument('--protein_embedding', type=str, default='tape', choices=['tape', 'prott5', 'esm2'], help='Protein embedding to use')
    parser.add_argument('--ec_level', type=str, default='all', choices=['all', '1', '2', '3', '4'], help='Deepest EC hierarchy level to predict')
    parser.add_argument('--output_dir', type=str, default='../../predictions', help='Path to the output directory')
    parser.add_argument('--threshold', type=float, default=0.5, help='Probability threshold for keeping a prediction at each level')
    parser.add_argument('--top_k', type=int, default=0, help='Keep only top-k candidates per protein per level (0 = keep all above threshold).')
    return parser.parse_args()


def load_model_weights(model, model_path: str, device):
    """Load either a plain state_dict or a checkpoint dict containing 'model_state_dict'."""
    obj = torch.load(model_path, map_location=device)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        state = obj["model_state_dict"]
        print(f"Loaded checkpoint from {model_path} (epoch={obj.get('epoch')}); using model_state_dict")
    else:
        state = obj
        print(f"Loaded state_dict from {model_path}")
    model.load_state_dict(state, strict=True)
    return model


@torch.no_grad()
def cascade_predict_protein(model, x_dict, masks, protein_idx, inv_maps, max_level,
                            threshold, top_k, device):
    """Top-down cascade for one protein: L1 is scored over all classes, and each
    deeper level is scored only over the children of the kept nodes at the previous
    level, so the predictions form a consistent hierarchy. `x_dict` are the backbone
    embeddings (computed once for the whole graph). Returns rows up to `max_level`:
    (level_key, ec_string, probability)."""
    rows = []
    kept_prev = []
    for level in LEVELS:
        target_type = LEVEL_TARGET_TYPE[level]
        if level == 'l1':
            candidates = list(range(x_dict[target_type].size(0)))
        else:
            ckey = CHILDREN_KEY[level]
            candidates = sorted({c for p in kept_prev for c in masks[ckey].get(p, [])})

        if not candidates:
            kept_prev = []
            if level == max_level:
                break
            continue

        cand = torch.tensor(candidates, dtype=torch.long, device=device)
        p_emb = x_dict['Protein'][protein_idx].unsqueeze(0).expand(len(candidates), -1)
        z = torch.cat([p_emb, x_dict[target_type][cand]], dim=-1)
        probs = torch.sigmoid(model.decoders[level](z).view(-1))

        keep = (probs > threshold).nonzero(as_tuple=False).view(-1)
        if top_k and int(top_k) > 0 and keep.numel() > int(top_k):
            top = torch.topk(probs[keep], k=int(top_k)).indices
            keep = keep[top]

        kept_prev = [candidates[i] for i in keep.tolist()]
        for i in keep.tolist():
            rows.append((level, inv_maps[target_type][candidates[i]], float(probs[i])))

        if level == max_level:
            break
    return rows


def generate_and_save_predictions(protein_ids, heterodata_path, model_path, model_config_path,
                                   max_level, requested_levels, output_dir, prediction_file_path,
                                   threshold: float = 0.5, top_k: int = 0):
    os.makedirs(output_dir, exist_ok=True)

    heterodata = torch.load(heterodata_path)

    # read protein id list from file or as string seperated by commas
    if isinstance(protein_ids, str) and protein_ids.endswith('.txt'):
        with open(protein_ids, 'r') as file:
            protein_ids = [line.strip() for line in file if line.strip()]
    elif isinstance(protein_ids, str):
        protein_ids = [pid.strip() for pid in protein_ids.split(',') if pid.strip()]
    elif isinstance(protein_ids, list):
        protein_ids = [str(pid).strip() for pid in protein_ids if str(pid).strip()]
    else:
        raise ValueError("protein_ids must be a string (file path or comma-separated) or a list.")
    protein_ids = list(dict.fromkeys(protein_ids))  # Remove duplicates, preserve order

    # check if all protein ids are in the heterodata
    protein_ids_not_found = [pid for pid in protein_ids if pid not in heterodata['Protein']['id_mapping']]
    if protein_ids_not_found:
        print('The following proteins were not found in our input knowledge graph and have been discarded:')
        for pid in protein_ids_not_found:
            print(f'{pid}')
        not_found = set(protein_ids_not_found)
        protein_ids = [pid for pid in protein_ids if pid not in not_found]

    print(f'Processing {len(protein_ids)}/{len(protein_ids) + len(protein_ids_not_found)} proteins')
    print('--------------------------------')

    # Load the joint model config and weights once (shared across all levels).
    with open(model_config_path, 'r') as file:
        model_config = yaml.safe_load(file)
    print(f'Loaded model config from {model_config_path}')

    mlp_hidden_layers = [4 * model_config['hidden_channels'], model_config['hidden_channels'], 1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ProtHGTHier(
        heterodata,
        hidden_channels=model_config['hidden_channels'],
        num_heads=model_config['num_heads'],
        num_layers=model_config['num_layers'],
        mlp_hidden_layers=mlp_hidden_layers,
        mlp_dropout=model_config['mlp_dropout'],
    )
    load_model_weights(model, model_path, device)
    model.eval()

    # Backbone embeddings are computed once for the whole graph; the cascade then
    # only applies the per-level decoders to the relevant (protein, node) pairs.
    # The full-graph forward over the entire KG is memory-heavy for the larger
    # embeddings (e.g. TAPE), so fall back to CPU if the GPU runs out of memory.
    def _embed(dev):
        model.to(dev)
        hd = heterodata.to(dev)
        with torch.no_grad():
            return hd, model.generate_embeddings(hd.x_dict, hd.edge_index_dict)

    try:
        heterodata, x_dict = _embed(device)
    except RuntimeError as e:
        if device.type == 'cuda' and 'out of memory' in str(e).lower():
            print('GPU out of memory during the full-graph forward; falling back to CPU...')
            torch.cuda.empty_cache()
            device = torch.device('cpu')
            heterodata, x_dict = _embed(device)
        else:
            raise
    masks = load_hierarchy_masks(heterodata)

    inv_maps = {
        LEVEL_TARGET_TYPE[level]: {idx: tok for tok, idx in heterodata[LEVEL_TARGET_TYPE[level]]['id_mapping'].items()}
        for level in LEVELS
    }
    id_mapping = heterodata['Protein']['id_mapping']

    all_proteins, all_terms, all_levels, all_probabilities = [], [], [], []
    for protein_id in tqdm(protein_ids, desc='Proteins'):
        rows = cascade_predict_protein(
            model, x_dict, masks, id_mapping[protein_id], inv_maps, max_level,
            threshold=threshold, top_k=top_k, device=device)
        for level, token, prob in rows:
            if level not in requested_levels:
                continue
            all_proteins.append(protein_id)
            all_terms.append(token)
            all_levels.append(LEVEL_LABEL[level])
            all_probabilities.append(prob)

    final_df = pd.DataFrame({
        'Protein': all_proteins,
        'EC': all_terms,
        'EC_level': all_levels,
        'Probability': all_probabilities,
    })
    final_df.to_csv(prediction_file_path, index=False, float_format="%.3f")
    print(f'Predictions saved to {prediction_file_path}')


def main():
    args = parse_args()

    prefix = os.path.splitext(os.path.basename(args.protein_ids))[0] if args.protein_ids.endswith('.txt') else 'predictions'
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    prediction_file_name = f'{prefix}_{args.protein_embedding}_ec{args.ec_level}_{now_str}.csv'
    prediction_file_path = os.path.join(args.output_dir, prediction_file_name)

    embedding = args.protein_embedding
    print(f'Using protein embedding: {embedding}')
    heterodata_path = HETERODATA_PATHS[embedding]
    print(f'Using heterodata path: {heterodata_path}')

    if args.ec_level == 'all':
        print('Predicting for all EC levels')
        max_level = 'l4'
        requested_levels = set(LEVELS)
    else:
        max_level = EC_LEVELS[args.ec_level]
        print(f'Predicting for EC level: {args.ec_level}')
        requested_levels = {max_level}

    print('Generating and saving predictions...')
    print('--------------------------------')
    generate_and_save_predictions(
        args.protein_ids,
        heterodata_path,
        MODEL_PATHS[embedding],
        MODEL_CONFIG_PATHS[embedding],
        max_level,
        requested_levels,
        args.output_dir,
        prediction_file_path,
        threshold=args.threshold,
        top_k=args.top_k,
    )


if __name__ == '__main__':
    main()
