import os
import torch
import argparse
import pandas as pd
import yaml
import copy
from model import ProtHGT

MODEL_CONFIG_PATHS = {
    'GO_term_F': os.path.join('../configs', "prothgt-config-molecular-function.yaml"),
    'GO_term_P': os.path.join('../configs', "prothgt-config-biological-process.yaml"),
    'GO_term_C': os.path.join('../configs', "prothgt-config-cellular-component.yaml")
}

MODEL_PATHS = {
        'GO_term_F': os.path.join('../models', "prothgt-model-molecular-function.pt"),
        'GO_term_P': os.path.join('../models', "prothgt-model-biological-process.pt"),
        'GO_term_C': os.path.join('../models', "prothgt-model-cellular-component.pt")
    }

GO_CATEGORIES = {
    'all': None,
    'molecular_function': 'GO_term_F',
    'biological_process': 'GO_term_P',
    'cellular_component': 'GO_term_C'
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protein_ids', type=str, required=True)
    parser.add_argument('--heterodata_path', type=str, default='../data/prothgt-kg.pt',)
    parser.add_argument('--go_category', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='../predictions')
    parser.add_argument('--batch_size', type=int, default=100)
    return parser.parse_args()

def load_data(heterodata, protein_ids, go_category):

    """Process the loaded heterodata for specific proteins and GO categories."""
    # Get protein indices for all input proteins
    protein_indices = [heterodata['Protein']['id_mapping'][pid] for pid in protein_ids]
    n_terms = len(heterodata[go_category]['id_mapping'])

    all_edges = []
    for protein_idx in protein_indices:
        for term_idx in range(n_terms):
            all_edges.append([protein_idx, term_idx])

    edge_index = torch.tensor(all_edges).t()

    heterodata[('Protein', 'protein_function', go_category)].edge_index = edge_index
    heterodata[(go_category, 'rev_protein_function', 'Protein')].edge_index = torch.stack([edge_index[1], edge_index[0]])
    
    return heterodata

def generate_predictions(heterodata, model, target_type):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()
    heterodata = heterodata.to(device)

    with torch.no_grad():
        edge_label_index = heterodata.edge_index_dict[('Protein', 'protein_function', target_type)]
        predictions, _ = model(heterodata.x_dict, heterodata.edge_index_dict, edge_label_index, target_type)
        predictions = torch.sigmoid(predictions)
    
    return predictions.cpu()

def create_prediction_df(predictions, heterodata, protein_ids, go_category):
        

    go_category_dict = {
        'GO_term_F': 'Molecular Function',
        'GO_term_P': 'Biological Process',
        'GO_term_C': 'Cellular Component'
    }

    # Get number of GO terms for this category
    n_go_terms = len(heterodata[go_category]['id_mapping'])
    
    # Create lists to store the data
    all_proteins = []
    all_go_terms = []
    all_categories = []
    all_probabilities = []
    
    # Get list of GO terms once
    go_terms = list(heterodata[go_category]['id_mapping'].keys())
    
    # Process predictions for each protein
    for i, protein_id in enumerate(protein_ids):
        # Get predictions for this protein
        start_idx = i * n_go_terms
        end_idx = (i + 1) * n_go_terms
        protein_predictions = predictions[start_idx:end_idx]
        
        # Extend the lists
        all_proteins.extend([protein_id] * n_go_terms)
        all_go_terms.extend(go_terms)
        all_categories.extend([go_category_dict[go_category]] * n_go_terms)
        all_probabilities.extend(protein_predictions.tolist())
    
    # Create DataFrame
    prediction_df = pd.DataFrame({
        'Protein': all_proteins,
        'GO_term': all_go_terms,
        'GO_category': all_categories,
        'Probability': all_probabilities
    })
    
    return prediction_df

def generate_and_save_predictions(protein_ids, heterodata_path, model_paths, model_config_paths, go_category, output_dir, batch_size=100):
    
    os.makedirs(output_dir, exist_ok=True)

    heterodata = torch.load(heterodata_path)

    # read protein id list from file or as string seperated by commas
    if protein_ids.endswith('.txt'):
        with open(protein_ids, 'r') as file:
            protein_ids = file.read().splitlines()
            protein_ids = [pid.strip() for pid in protein_ids]
            protein_ids = list(dict.fromkeys(protein_ids))
    else:
        protein_ids = protein_ids.split(',')

    # check if all protein ids are in the heterodata
    protein_ids_not_found = []
    for pid in protein_ids:
        if pid not in heterodata['Protein']['id_mapping']:
            protein_ids_not_found.append(pid)

    if protein_ids_not_found:
        print(f'The following proteins were not found in our input knowledge graph and have been discarded:')
        for pid in protein_ids_not_found:
            print(pid)
    
    print(f'Processing {len(protein_ids)-len(protein_ids_not_found)}/{len(protein_ids)} proteins')

    # Convert single protein ID to list if necessary
    if isinstance(protein_ids, str):
        protein_ids = [protein_ids]
    
    def batch_generator(items, batch_size):
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_predictions_dfs = []

    for go_cat, model_config_path, model_path in zip(go_category, model_config_paths, model_paths):
        print(f'Generating predictions for {go_cat}...')
        
        # Load model config and initialize model (moved outside batch loop)
        with open(model_config_path, 'r') as file:
            model_config = yaml.safe_load(file)
        
        model = ProtHGT(
            heterodata,
            hidden_channels=model_config['hidden_channels'][0],
            num_heads=model_config['num_heads'],
            num_layers=model_config['num_layers'],
            mlp_hidden_layers=model_config['hidden_channels'][1],
            mlp_dropout=model_config['mlp_dropout']
        )
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f'Loaded model weights from {model_path}')
        
        category_predictions = []
        
        # Process each batch
        for batch_idx, batch_proteins in enumerate(batch_generator(protein_ids, batch_size)):
            print(f'Processing batch {batch_idx + 1} for {go_cat}...')
            
            # Process data for current batch
            processed_data = load_data(copy.deepcopy(heterodata), batch_proteins, go_cat)
            
            # Generate predictions for batch
            predictions = generate_predictions(processed_data, model, go_cat)
            prediction_df = create_prediction_df(predictions, processed_data, batch_proteins, go_cat)
            category_predictions.append(prediction_df)
            
            # Clean up batch memory
            del processed_data
            del predictions
            torch.cuda.empty_cache()
        
        # Combine all batch predictions for this category
        category_df = pd.concat(category_predictions, ignore_index=True)
        all_predictions_dfs.append(category_df)
        
        # Clean up category memory
        del model
        del category_predictions
        torch.cuda.empty_cache()

    del heterodata

    # Combine all predictions across categories
    final_df = pd.concat(all_predictions_dfs, ignore_index=True)
    
    # Clean up
    del all_predictions_dfs
    torch.cuda.empty_cache()

    final_df.to_csv(os.path.join(output_dir, f'predictions.csv'), index=False)
    print(f'Predictions saved to {os.path.join(output_dir, "predictions.csv")}')

def main():
    args = parse_args()

    if args.go_category == 'all':
        model_config_paths = [MODEL_CONFIG_PATHS[cat] for cat in ['GO_term_F', 'GO_term_P', 'GO_term_C']]
        model_paths = [MODEL_PATHS[cat] for cat in ['GO_term_F', 'GO_term_P', 'GO_term_C']]
        go_categories = ['GO_term_F', 'GO_term_P', 'GO_term_C']
    else:
        model_config_paths = [MODEL_CONFIG_PATHS[args.go_category]]
        model_paths = [MODEL_PATHS[args.go_category]]
        go_categories = GO_CATEGORIES[args.go_category]

    generate_and_save_predictions(
        args.protein_ids, 
        args.heterodata_path, 
        model_paths, 
        model_config_paths, 
        go_categories, 
        args.output_dir,
        args.batch_size
    )

if __name__ == '__main__':
    main()