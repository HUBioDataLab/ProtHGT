import torch
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader


def load_and_prepare_data(train_path, val_path, test_path, target_type, disjoint_ratio=0.3):
    """Load and prepare all datasets."""
    # Load data
    train_data_split = torch.load(train_path)
    val_data_split = torch.load(val_path)
    test_data_split = torch.load(test_path)

    # split datasets into message passing and prediction edges. 70% used for message passing.
    transform = T.RandomLinkSplit(
        num_val=0.0,
        num_test=0.0,
        neg_sampling_ratio=0.0,
        disjoint_train_ratio=disjoint_ratio,
        add_negative_train_samples=False,
        edge_types=[('Protein', 'protein_function', target_type)],
        rev_edge_types=[(target_type, 'rev_protein_function', 'Protein')],
    )

    # Prepare datasets
    train_data, _, _ = transform(train_data_split)
    val_data, _, _ = transform(val_data_split)
    test_data, _, _ = transform(test_data_split)

    print("\nDataset statistics:\n")
    for data, name in zip([train_data, val_data, test_data], ['Train', 'Validation', 'Test']):
        print(f'----------------------------------{name.upper()}----------------------------------')
        print(f'Number of message passing edges in {name} data: {data["Protein", "protein_function", target_type].edge_index.shape[1]}')
        print(f'Number of prediction edges in {name} data: {len(data["Protein", "protein_function", target_type].edge_label_index[0])}\n')

        print(f"Detailed Node and Edge Statistics:\n")
        for node_type in data.node_types:
            print(f"{node_type} features shape: {data[node_type].x.shape}")
        for edge_type in data.edge_types:
            print(f"{edge_type} edge index: {data[edge_type].edge_index.shape[1]}")
            if hasattr(data[edge_type], 'edge_label_index'):
                print(f"{edge_type} edge label index: {data[edge_type].edge_label_index.shape[1]}")
                print(f"{edge_type} edge label: {data[edge_type].edge_label.shape[0]}")
        print()

    assert train_data["Protein", "protein_function", target_type].edge_index.shape[1] == train_data[target_type, "rev_protein_function", "Protein"].edge_index.shape[1]
    assert val_data["Protein", "protein_function", target_type].edge_index.shape[1] == val_data[target_type, "rev_protein_function", "Protein"].edge_index.shape[1]
    assert test_data["Protein", "protein_function", target_type].edge_index.shape[1] == test_data[target_type, "rev_protein_function", "Protein"].edge_index.shape[1]

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