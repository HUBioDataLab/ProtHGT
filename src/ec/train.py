import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import json
import argparse
from datetime import datetime

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric import seed_everything

from data_loader import (
    LEVEL_EDGES,
    load_and_prepare_data, create_level_loaders, build_train_loader_for_level,
    compute_hard_negative_pool, build_epoch_negatives, add_hard_negatives,
    sample_protein_subset,
)
from model import ProtHGTHier, LEVELS
from eval import score_all_levels, metrics_from_scores

seed_everything(42)


def parse_args():
    parser = argparse.ArgumentParser(description='Train ProtHGT model for hierarchical EC number prediction')
    parser.add_argument('--train-data', type=str,
                        default='../../data/ec/prothgt-ec-train-graph.pt',
                        help='Path to training data')
    parser.add_argument('--val-data', type=str,
                        default='../../data/ec/prothgt-ec-val-graph.pt',
                        help='Path to validation data')
    parser.add_argument('--test-data', type=str,
                        default='../../data/ec/prothgt-ec-test-graph.pt',
                        help='Path to test data')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--output-dir', type=str, default='../../outputs', help='Output directory')
    parser.add_argument('--checkpoint-dir', type=str, default=None, help='Checkpoint directory')
    parser.add_argument('--num-workers', type=int, default=2, help='Number of data loading workers')
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from file or use defaults."""
    with open(config_path, 'r') as f:
        return json.load(f)


# initialize lazy parameters (one forward pass per level)
@torch.no_grad()
def init_params(model, train_loaders, device):
    for level in LEVELS:
        edge_type, _ = LEVEL_EDGES[level]
        batch = next(iter(train_loaders[level])).to(device)
        model(batch.x_dict, batch.edge_index_dict, batch[edge_type].edge_label_index, level)


def train(model, optim, train_loaders, epoch, device, config):
    """Joint training step: one forward/backward per level per batch, one optim
    step. Returns the per-level mean training loss."""
    model.train()
    total_examples = {level: 0 for level in LEVELS}
    total_loss_sum = {level: 0.0 for level in LEVELS}

    n_batches = min(len(train_loaders[level]) for level in LEVELS)
    iterators = {level: iter(train_loaders[level]) for level in LEVELS}
    pbar = tqdm(range(n_batches), desc=f'Epoch {epoch}')

    for _ in pbar:
        optim.zero_grad()
        loss_report = {}

        for level in LEVELS:
            edge_type, _ = LEVEL_EDGES[level]
            batch = next(iterators[level]).to(device)

            try:
                out, _ = model(batch.x_dict, batch.edge_index_dict,
                               batch[edge_type].edge_label_index, level)
            except RuntimeError as e:
                print(f"Error in model forward pass ({level}): {str(e)}")
                continue

            true_label = batch[edge_type].edge_label
            pos_weight = torch.tensor([config[f'pos_weight_{level}']], device=device)
            level_loss = F.binary_cross_entropy_with_logits(out, true_label, pos_weight=pos_weight)
            level_loss.backward()
            loss_report[level] = float(level_loss)

            total_examples[level] += len(true_label)
            total_loss_sum[level] += float(level_loss) * len(true_label)

        # Gradient clipping (if configured) on the combined gradient, before the
        # step, to guard against a rare divergent step collapsing the model.
        if config.get('grad_clip_norm'):
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip_norm'])

        optim.step()
        pbar.set_postfix({lvl: f'{loss_report.get(lvl, 0.0):.3f}' for lvl in LEVELS})

    return {level: round(total_loss_sum[level] / total_examples[level], 4) for level in LEVELS}


@torch.no_grad()
def val_loss(model, loader, level, epoch, device, config):
    """Per-level validation loss over the block-negative val loader (training signal)."""
    edge_type, _ = LEVEL_EDGES[level]
    model.eval()

    total_examples = total_loss = 0
    pos_weight = torch.tensor([config[f'pos_weight_{level}']])
    for batch in tqdm(loader, total=len(loader), desc=f'Val {level} epoch {epoch}'):
        batch = batch.to(device)
        try:
            pred, _ = model(batch.x_dict, batch.edge_index_dict,
                            batch[edge_type].edge_label_index, level)
        except RuntimeError as e:
            print(f"Error in model forward pass ({level}): {str(e)}")
            continue
        pred = pred.cpu()
        true_label = batch[edge_type].edge_label.cpu()
        loss = F.binary_cross_entropy_with_logits(pred, true_label, pos_weight=pos_weight)
        total_examples += len(true_label)
        total_loss += float(loss) * len(true_label)

    return round(total_loss / total_examples, 4)


def evaluate_checkpoint(model, val_data, val_subset, test_data, test_subset,
                        config, device, num_workers, tag):
    """Per-level metrics (val and test) at the configured probability threshold"""
    threshold = config.get('eval_threshold', 0.5)
    val_scored = score_all_levels(model, val_data, config, device, val_subset,
                                  num_workers=num_workers, desc_prefix=f'{tag} Val')
    val_results = metrics_from_scores(val_scored, threshold)

    test_scored = score_all_levels(model, test_data, config, device, test_subset,
                                   num_workers=num_workers, desc_prefix=f'{tag} Test')
    test_results = metrics_from_scores(test_scored, threshold)

    for level in LEVELS:
        print(f'[{level}] Val:  {val_results[level]}')
        print(f'[{level}] Test: {test_results[level]}')

    return {'val': val_results, 'test': test_results}


def train_validation(train_data, val_data, test_data, config, run_name,
                     output_dir="./outputs", checkpoint_dir=None, num_workers=2):
    """Training and validation loop"""
    model_dir = os.path.join(output_dir, "models", run_name)
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    start = datetime.now()
    print(f"Starting training run: {run_name}")

    # Block negatives for all four levels (train's random part is refreshed per epoch; val is static)
    train_hard_neg_pools = {}
    for level in LEVELS:
        pos_edge_index, hard_negs = compute_hard_negative_pool(train_data, level)
        train_hard_neg_pools[level] = (pos_edge_index, hard_negs)
        build_epoch_negatives(train_data, level, pos_edge_index, hard_negs,
                              config[f'neg_sampling_ratio_{level}'], seed=0)
        add_hard_negatives(val_data, level, num_neg_per_positive=config[f'neg_sampling_ratio_{level}'])

    train_loaders, val_loaders = {}, {}
    for level in LEVELS:
        tr, va, _ = create_level_loaders(train_data, val_data, test_data, level, config, num_workers=num_workers)
        train_loaders[level], val_loaders[level] = tr, va

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Decoder input is 4*hidden_channels (concat of normalized pre/post-GNN Protein+target embeddings)
    mlp_hidden_layers = [4 * config['hidden_channels'], config['hidden_channels'], 1]
    model = ProtHGTHier(
        train_data,
        hidden_channels=config['hidden_channels'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        mlp_hidden_layers=mlp_hidden_layers,
        mlp_dropout=config['mlp_dropout'],
    )
    model.to(device)

    init_params(model, train_loaders, device)
    optimizer = torch.optim.Adam([
        dict(params=model.backbone_parameters(), weight_decay=config['wd'], lr=config['lr']),
        dict(params=model.decoders.parameters(), weight_decay=config['mlp_wd'], lr=config['mlp_lr']),
    ])

    checkpoint_interval = config.get('checkpoint_interval', 25)
    history = []

    # Protein subsets for the checkpoint evaluation (fixed across epochs)
    val_subset = sample_protein_subset(val_data, 'l4', config.get('val_eval_num_proteins', 500))
    test_subset = sample_protein_subset(test_data, 'l4', None)

    for epoch in range(1, config['epochs'] + 1):
        epoch_start = datetime.now()

        # Refresh each level's random negatives and rebuild its train loader
        for level in LEVELS:
            pos_edge_index, hard_negs = train_hard_neg_pools[level]
            build_epoch_negatives(train_data, level, pos_edge_index, hard_negs,
                                  config[f'neg_sampling_ratio_{level}'], seed=epoch)
            train_loaders[level] = build_train_loader_for_level(train_data, level, config, num_workers=num_workers)

        # Training phase
        train_losses = train(model, optimizer, train_loaders, epoch, device, config)

        # Validation phase
        val_losses = {level: val_loss(model, val_loaders[level], level, epoch, device, config)
                      for level in LEVELS}

        print(f'\nEpoch {epoch}/{config["epochs"]} completed in {datetime.now() - epoch_start}')
        for level in LEVELS:
            print(f'[{level}] Train loss: {train_losses[level]:.4f}, Val loss: {val_losses[level]:.4f}')

        history.append({'epoch': epoch,
                        **{f'train_{lvl}_loss': train_losses[lvl] for lvl in LEVELS},
                        **{f'val_{lvl}_loss': val_losses[lvl] for lvl in LEVELS}})

        # Checkpoint evaluation: at every checkpoint interval (and the final epoch)
        # compute the per-level metrics and save that epoch's model + metrics
        if epoch % checkpoint_interval == 0 or epoch == config['epochs']:
            print(f'\n----- Checkpoint evaluation (epoch {epoch}) -----')
            ckpt_results = evaluate_checkpoint(
                model, val_data, val_subset, test_data, test_subset,
                config, device, num_workers, tag=f'Epoch {epoch}')

            torch.save(model.state_dict(), os.path.join(model_dir, f'{run_name}_epoch{epoch}.pt'))
            with open(os.path.join(results_dir, f'{run_name}_epoch{epoch}_results.json'), 'w') as f:
                json.dump(ckpt_results, f, indent=4)
            if checkpoint_dir:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                }, os.path.join(checkpoint_dir, f'{run_name}_epoch{epoch}_checkpoint.pt'))

    # Save final model and training history
    torch.save(model.state_dict(), os.path.join(model_dir, f'{run_name}_final.pt'))
    pd.DataFrame(history).to_csv(
        os.path.join(results_dir, f'{run_name}_training_history.csv'), index=False)

    print(f'\nTraining completed for {run_name} in {datetime.now() - start}')


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    print(f"Configuration:\n{json.dumps(config, indent=2)}\n")

    # Load and prepare data
    print("Loading and preparing datasets:")
    print(f'Train data: {args.train_data}')
    print(f'Val data: {args.val_data}')
    print(f'Test data: {args.test_data}')
    train_data, val_data, test_data = load_and_prepare_data(
        args.train_data, args.val_data, args.test_data,
        disjoint_ratio=config.get('disjoint_train_ratio', 0.3),
    )

    # Start training
    print('----------------------Starting training----------------------\n')
    run_name = f'ec_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    train_validation(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        config=config,
        run_name=run_name,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        num_workers=args.num_workers,
    )


if __name__ == '__main__':
    main()
