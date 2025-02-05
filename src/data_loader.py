import torch
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader

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

def load_and_prepare_data(train_path, val_path, test_path, target_type, disjoint_ratio=0.3):
    """Load and prepare all datasets."""
    # Load data
    train_data_split = torch.load(train_path)
    val_data = torch.load(val_path)
    test_data = torch.load(test_path)

    # Prepare datasets
    train_data = prepare_data(train_data_split, is_train=True, 
                            disjoint_ratio=disjoint_ratio, 
                            target_type=target_type)
    val_data = prepare_data(val_data, target_type=target_type)
    test_data = prepare_data(test_data, target_type=target_type)

    return train_data, val_data, test_data

def create_data_loaders(train_data, val_data, test_data, config, target_type, num_workers=2):
    """Create data loaders for training, validation and testing."""
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

    return train_loader, val_loader, test_loader