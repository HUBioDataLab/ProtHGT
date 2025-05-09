import os
import json
import torch
import numpy as np
import pandas as pd
import warnings
import argparse
from datetime import datetime
from tqdm import tqdm
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric import seed_everything

from data_loader import load_and_prepare_data, create_data_loaders
from model import ProtHGT
from utils import metrics

seed_everything(42)

def parse_args():
    parser = argparse.ArgumentParser(description='Train ProtHGT model for protein function prediction')
    parser.add_argument('--train-data', type=str, 
                       default='../data/prothgt-train-graph.pt',
                       help='Path to training data')
    parser.add_argument('--val-data', type=str, 
                       default='../data/prothgt-val-graph.pt',
                       help='Path to validation data')
    parser.add_argument('--test-data', type=str, 
                       default='../data/prothgt-test-graph.pt',
                       help='Path to test data')
    parser.add_argument('--target-type', type=str, required=True, help='Target prediction type')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--output-dir', type=str, default='../outputs', help='Output directory')
    parser.add_argument('--checkpoint-dir', type=str, default=None, help='Checkpoint directory')
    parser.add_argument('--num-workers', type=int, default=2, help='Number of data loading workers')
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from file or use defaults."""
    with open(config_path, 'r') as f:
        return json.load(f)

# initialize lazy parameters
@torch.no_grad()
def init_params(model,loader, device, target_type):
    batch = next(iter(loader))
    batch = batch.to(device)
    model(batch.x_dict, batch.edge_index_dict, batch[("Protein", "protein_function", target_type)].edge_label_index, target_type)

def train(model, optim, loader, epoch, device, target_type, config):
    model.train()
    
    total_examples = total_loss = 0
    num_batches = len(loader)
    
    pbar = tqdm(enumerate(loader), total=num_batches, desc=f'Epoch {epoch}')
    
    try:
        for batch_idx, batch in pbar:
            optim.zero_grad()
            batch = batch.to(device)
            
            try:
                out, _ = model(
                    batch.x_dict, 
                    batch.edge_index_dict,
                    batch[("Protein", "protein_function", target_type)].edge_label_index,
                    target_type
                )
            except RuntimeError as e:
                print(f"Error in model forward pass: {str(e)}")
                continue

            true_label = batch[("Protein", "protein_function", target_type)].edge_label
            true_label_size = len(true_label)

            if batch_idx == 0 and epoch == 1:
                print(f'\nFirst batch statistics:')
                print(f'Output shape: {out.shape}, range: [{out.min():.3f}, {out.max():.3f}]')
                print(f'Label shape: {true_label.shape}, unique values: {torch.unique(true_label).tolist()}\n')

            pos_weight = torch.tensor([config["pos_weight"]]).to(device)
            loss = F.binary_cross_entropy_with_logits(out, true_label, pos_weight=pos_weight)
            loss.backward()
                        
            optim.step()

            total_examples += true_label_size
            total_loss += float(loss) * true_label_size
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    except Exception as e:
        print(f"Training error occurred: {str(e)}")
        raise e
    
    avg_loss = total_loss / total_examples
    return round(avg_loss, 4)

@torch.no_grad()
def test(model, test_loader, epoch, device, target_type, config):
    if target_type == 'GO_term_P':
        print('Evaluating on CPU')
        model.to('cpu')

    model.eval()

    total_examples = total_loss = 0
    total_precision = total_recall = total_f1 = total_pr_auc = total_acc = total_mcc = 0

    num_batches = len(test_loader)
    pbar = tqdm(enumerate(test_loader), total=num_batches, desc=f'Evaluation Epoch {epoch}')

    try:
        for batch_idx, batch in pbar:
            if target_type == 'GO_term_P':
                batch = batch.to('cpu')
            else:
                batch = batch.to(device)

            try:
                pred, _ = model(
                    batch.x_dict, 
                    batch.edge_index_dict,
                    batch[("Protein", "protein_function", target_type)].edge_label_index,
                    target_type
                )
            except RuntimeError as e:
                print(f"Error in model forward pass: {str(e)}")
                continue

            pred = pred.cpu()
            true_label = batch[("Protein", "protein_function", target_type)].edge_label.cpu()
            true_label_size = len(true_label)

            # Calculate loss and predictions
            pos_weight = torch.tensor([config["pos_weight"]]).to(pred.device)
            ts_loss = F.binary_cross_entropy_with_logits(pred, true_label, pos_weight=pos_weight)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                binary_pred = np.where(pred.numpy() > 0.0, 1, 0)
                prob_pred = torch.sigmoid(pred).numpy()

            total_examples += true_label_size
            total_loss += float(ts_loss) * true_label_size
            
            # Calculate metrics for current batch
            score_list = metrics(prob_pred, binary_pred, true_label)
            
            # Update running totals
            total_precision += float(score_list[0]) * true_label_size
            total_recall += float(score_list[1]) * true_label_size
            total_f1 += float(score_list[2]) * true_label_size
            total_pr_auc += float(score_list[3]) * true_label_size
            total_acc += float(score_list[4]) * true_label_size
            total_mcc += float(score_list[5]) * true_label_size
            
            pbar.set_postfix({
                'loss': f'{ts_loss:.4f}',
                'f1': f'{score_list[2]:.4f}'
            })

    except Exception as e:
        print(f"Evaluation error occurred: {str(e)}")
        raise e

    # Calculate final averaged metrics
    score_dict = {
        'loss': round(total_loss/total_examples, 4),
        'precision': round(total_precision/total_examples, 4),
        'recall': round(total_recall/total_examples, 4),
        'f1': round(total_f1/total_examples, 4),
        'pr_auc': round(total_pr_auc/total_examples, 4),
        'acc': round(total_acc/total_examples, 4),
        'mcc': round(total_mcc/total_examples, 4)
    }

    if target_type == 'GO_term_P':
        model.to(device)

    return score_dict

def train_validation(
    train_data, 
    val_data, 
    test_data, 
    config, 
    run_name,
    target_type,
    output_dir="./outputs",
    checkpoint_dir=None,
    num_workers=2
):
    """Training and validation loop for the HGT model."""
    # Create output directories
    model_dir = os.path.join(output_dir, "models", run_name)
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    start = datetime.now()
    print(f"Starting training run: {run_name}")

    # Initialize data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_data, val_data, test_data, config, target_type, num_workers
    )

    # Setup device and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model = ProtHGT(
        train_data,
        hidden_channels=config['hidden_channels'][0],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        mlp_hidden_layers=config['hidden_channels'][1],
        mlp_dropout=config['mlp_dropout']
    )
    model.to(device)

    # Initialize parameters and optimizer
    init_params(model, train_loader, device, target_type)
    optimizer = torch.optim.Adam([
        dict(params=model.convs.parameters(), weight_decay=config['wd'], lr=config['lr']),
        dict(params=model.mlp.parameters(), weight_decay=config['mlp_wd'], lr=config['mlp_lr']),
    ])

    # Training loop
    results = []
    
    for epoch in range(1, config['epochs'] + 1):
        print(f'\nEpoch {epoch}/{config["epochs"]}')
        epoch_start = datetime.now()

        # Training phase
        train_loss = train(model, optimizer, train_loader, epoch, device, target_type, config)
        
        # Validation phase
        val_metrics = test(model, val_loader, epoch, device, target_type, config)
        
        # Test phase (only at final epoch)
        if epoch == config['epochs']:
            test_metrics = test(model, test_loader, epoch, device, target_type, config)
            print(f'Final test metrics: {test_metrics}')
            
            # Save test results
            with open(os.path.join(results_dir, f'{run_name}_test_results.json'), 'w') as f:
                json.dump(test_metrics, f, indent=4)

        # Save checkpoint at halfway point
        if epoch == config['epochs']//2 and checkpoint_dir:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }
            torch.save(checkpoint, os.path.join(checkpoint_dir, f'{run_name}_checkpoint.pt'))

        # Log metrics
        epoch_time = datetime.now() - epoch_start
        print(f'Epoch completed in {epoch_time}')
        print(f'Train loss: {train_loss:.4f}, Val loss: {val_metrics["loss"]:.4f}')
        print(f'Validation metrics: {val_metrics}')

        # Store results
        results.append([
            epoch, train_loss, val_metrics['loss'],
            val_metrics['precision'], val_metrics['recall'],
            val_metrics['f1'], val_metrics['pr_auc'],
            val_metrics['acc'], val_metrics['mcc']
        ])

    # Save final model and results
    torch.save(model.state_dict(), os.path.join(model_dir, f'{run_name}_final.pt'))
    
    results_df = pd.DataFrame(
        results,
        columns=['epoch', 'train_loss', 'val_loss', 'val_precision',
                'val_recall', 'val_f1', 'val_pr_auc', 'val_acc', 'val_mcc']
    )
    results_df.to_csv(os.path.join(results_dir, f'{run_name}_training_history.csv'), index=False)

    total_time = datetime.now() - start
    print(f'Training completed for {run_name} in {total_time}')
    
def main():
    args = parse_args()
    target_type_dict = {
        'GO_term_F': 'Molecular Function',
        'GO_term_P': 'Biological Process',
        'GO_term_C': 'Cellular Component'
    }

    print(f'Starting training model for {target_type_dict[args.target_type]} target type')
    # System information
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Load and prepare data
    print("Loading and preparing datasets:")
    print(f'Train data: {args.train_data}')
    print(f'Val data: {args.val_data}')
    print(f'Test data: {args.test_data}')

    train_data, val_data, test_data = load_and_prepare_data(
        args.train_data, 
        args.val_data, 
        args.test_data,
        args.target_type,
    )

    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    print(f"Configuration:\n{json.dumps(config, indent=2)}\n")

    # Start training
    print('----------------------Starting training----------------------\n')
    run_name = f'{args.target_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    
    try:
        train_validation(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            config=config,
            run_name=run_name,
            target_type=args.target_type,
            output_dir=args.output_dir,
            checkpoint_dir=args.checkpoint_dir,
            num_workers=args.num_workers
        )

    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise

if __name__ == '__main__':
    main()