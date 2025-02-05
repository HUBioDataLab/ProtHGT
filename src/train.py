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

# Assuming these are from local modules
from model import ProtHGT  # Your model implementation
from utils import metrics  # Your metrics implementation


def parse_args():
    parser = argparse.ArgumentParser(description='Train ProtHGT model for protein function prediction')
    parser.add_argument('--train-data', type=str, required=True, help='Path to training data')
    parser.add_argument('--val-data', type=str, required=True, help='Path to validation data')
    parser.add_argument('--test-data', type=str, required=True, help='Path to test data')
    parser.add_argument('--target-type', type=str, required=True, help='Target prediction type')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--output-dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--checkpoint-dir', type=str, default=None, help='Checkpoint directory')
    parser.add_argument('--num-workers', type=int, default=2, help='Number of data loading workers')
    parser.add_argument('--disjoint-ratio', type=float, default=0.3, 
                       help='Disjoint train ratio for link split')
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from file or use defaults."""
    with open(config_path, 'r') as f:
        return json.load(f)

def prepare_data(data, is_train=False, disjoint_ratio=0.3, target_type=None):
    """Prepare dataset for training/validation/testing."""
    if is_train:
        transform = T.RandomLinkSplit(
            num_val=0.0,
            num_test=0.0,
            neg_sampling_ratio=0.0,
            disjoint_train_ratio=disjoint_ratio,
            add_negative_train_samples=False,
            edge_types=[('Protein', 'protein_function', target_type)],
            rev_edge_types=[(target_type, 'rev_protein_function', 'Protein')],
        )
        train_data, _, _ = transform(data)
        return train_data
    else:
        # For validation and test sets
        data[("Protein", "protein_function", target_type)].edge_label_index = \
            data[("Protein", "protein_function", target_type)].edge_index
        data[("Protein", "protein_function", target_type)].edge_label = \
            torch.ones(data[("Protein", "protein_function", target_type)].edge_index.shape[1], 
                      dtype=torch.float)
        return data

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
    
    # Use tqdm for progress tracking
    pbar = tqdm(enumerate(loader), total=num_batches, desc=f'Epoch {epoch}')
    
    try:
        for batch_idx, batch in pbar:
            optim.zero_grad()
            batch = batch.to(device)
            
            # Wrap model forward pass in try-except
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

            # Debug information for first batch of first epoch
            if batch_idx == 0 and epoch == 1:
                print(f'\nFirst batch statistics:')
                print(f'Output shape: {out.shape}, range: [{out.min():.3f}, {out.max():.3f}]')
                print(f'Label shape: {true_label.shape}, unique values: {torch.unique(true_label).tolist()}\n')

            # Calculate loss with gradient clipping
            pos_weight = torch.tensor([config["pos_weight"]]).to(device)
            loss = F.binary_cross_entropy_with_logits(out, true_label, pos_weight=pos_weight)
            loss.backward()
            
            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optim.step()

            total_examples += true_label_size
            total_loss += float(loss) * true_label_size
            
            # Update progress bar with current loss
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
                warnings.simplefilter("ignore")  # Suppress numpy warnings
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
            
            # Update progress bar with current metrics
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
    """
    Training and validation loop for the HGT model.
    
    Args:
        train_data: Training dataset
        val_data: Validation dataset
        test_data: Test dataset
        config: Configuration dictionary containing model and training parameters
        run_name: Name of the current run
        target_type: Type of target prediction
        output_dir: Directory to save outputs (default: "./outputs")
        checkpoint_dir: Directory to save checkpoints (default: None)
        num_workers: Number of workers for data loading (default: 2)
    """
    # Create output directories
    model_dir = os.path.join(output_dir, "models", run_name)
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    start = datetime.now()
    print(f"Starting training run: {run_name}")

    # Initialize data loaders
    train_loader = LinkNeighborLoader(
        train_data,
        num_neighbors=config['num_neighbors'],
        neg_sampling_ratio=config['neg_sample_ratio'],
        shuffle=True,
        edge_label=train_data["Protein", "protein_function", target_type].edge_label,
        edge_label_index=(("Protein", "protein_function", target_type), 
                         train_data["Protein", "protein_function", target_type].edge_label_index),
        batch_size=config['tr_batch_size'],
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = LinkNeighborLoader(
        val_data,
        num_neighbors=[-1],
        neg_sampling_ratio=config['neg_sample_ratio'],
        shuffle=False,
        edge_label=val_data["Protein", "protein_function", target_type].edge_label,
        edge_label_index=(("Protein", "protein_function", target_type),
                         val_data["Protein", "protein_function", target_type].edge_label_index),
        batch_size=int(len(val_data["Protein", "protein_function", target_type].edge_label)),
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = LinkNeighborLoader(
        test_data,
        num_neighbors=[-1],
        neg_sampling_ratio=config['neg_sample_ratio'],
        shuffle=False,
        edge_label=test_data["Protein", "protein_function", target_type].edge_label,
        edge_label_index=(("Protein", "protein_function", target_type),
                         test_data["Protein", "protein_function", target_type].edge_label_index),
        batch_size=int(len(test_data["Protein", "protein_function", target_type].edge_label)),
        num_workers=num_workers,
        pin_memory=True
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
    
    # System information
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Load data
    print("Loading datasets...")
    train_data_split = torch.load(args.train_data)
    val_data = torch.load(args.val_data)
    test_data = torch.load(args.test_data)

    # Print initial statistics
    print("Initial data statistics:")
    print(f"Train edges: {len(train_data_split['Protein', 'protein_function', args.target_type].edge_index[0])}")

    # Prepare datasets
    print("\nPreparing datasets...")
    train_data = prepare_data(train_data_split, is_train=True, 
                            disjoint_ratio=args.disjoint_ratio, 
                            target_type=args.target_type)
    val_data = prepare_data(val_data, target_type=args.target_type)
    test_data = prepare_data(test_data, target_type=args.target_type)

    # Print dataset statistics
    print("\nDataset statistics:")
    print(f"Train edges: {train_data['Protein', 'protein_function', args.target_type].edge_label_index.shape[1]}")
    print(f"Validation edges: {val_data['Protein', 'protein_function', args.target_type].edge_label_index.shape[1]}")
    print(f"Test edges: {test_data['Protein', 'protein_function', args.target_type].edge_label_index.shape[1]}\n")

    # Load configuration
    config = load_config(args.config)
    print(f"Configuration:\n{json.dumps(config, indent=2)}\n")

    # Print model information
    print("Dataset information:")
    print(f"Train Data: {train_data}")
    print(f"Validation Data: {val_data}")
    print(f"Test Data: {test_data}\n")

    # Start training
    print('----------------------Starting training----------------------\n')
    run_name = f'mlp_{args.target_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    
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