import json
from collections import defaultdict

import torch
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader

from model import LEVELS

# Supervised edge (and its reverse) for each EC level
LEVEL_EDGES = {
    'l1': (('Protein', 'protein_ec1', 'EC_class_1'), ('EC_class_1', 'rev_protein_ec1', 'Protein')),
    'l2': (('Protein', 'protein_ec2', 'EC_class_2'), ('EC_class_2', 'rev_protein_ec2', 'Protein')),
    'l3': (('Protein', 'protein_ec3', 'EC_class_3'), ('EC_class_3', 'rev_protein_ec3', 'Protein')),
    'l4': (('Protein', 'protein_ec', 'EC_number'), ('EC_number', 'rev_protein_ec', 'Protein')),
}
LEVEL_NODE_TYPE = {level: edges[0][2] for level, edges in LEVEL_EDGES.items()}

# EC taxonomy edges (child -> parent), used to find hierarchy siblings for hard negatives
PARENT_EDGE_TYPES = {
    'l4_parent': ('EC_number', 'ec_parent_3', 'EC_class_3'),
    'l3_parent': ('EC_class_3', 'ec_parent_2', 'EC_class_2'),
    'l2_parent': ('EC_class_2', 'ec_parent_1', 'EC_class_1'),
}
CHILD_EDGE_TYPES = {
    'l1_children': ('EC_class_1', 'rev_ec_parent_1', 'EC_class_2'),
    'l2_children': ('EC_class_2', 'rev_ec_parent_2', 'EC_class_3'),
    'l3_children': ('EC_class_3', 'rev_ec_parent_3', 'EC_number'),
}

# (parent_edge, children_edge) per level for sibling hard negatives (L3/L4 only)
HARD_NEG_KEYS = {
    'l3': ('l3_parent', 'l2_children'),
    'l4': ('l4_parent', 'l3_children'),
}


def _expand_num_neighbors(config):
    """Repeat the configured neighbor count once per GNN layer so sampling depth matches model depth."""
    base = config['num_neighbors'][0]
    return [base] * config['num_layers']


def make_inductive(data, edge_type, rev_edge_type):
    """Make a split fully inductive: move all positives to edge_label_index and empty edge_index."""
    mp_edge_index = data[edge_type].edge_index
    sup_edge_index = data[edge_type].edge_label_index
    sup_edge_label = data[edge_type].edge_label

    data[edge_type].edge_label_index = torch.cat([mp_edge_index, sup_edge_index], dim=1)
    data[edge_type].edge_label = torch.cat([
        torch.ones(mp_edge_index.shape[1]), sup_edge_label,
    ])
    data[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long)
    data[rev_edge_type].edge_index = torch.empty((2, 0), dtype=torch.long)


def load_id_mappings(mapping_path):
    """Load a {node_type: {key: index}} mapping.json inverted as {node_type: {index: key}}."""
    with open(mapping_path) as f:
        raw = json.load(f)
    return {node_type: {v: k for k, v in mapping.items()} for node_type, mapping in raw.items()}


def load_and_prepare_data(train_path, val_path, test_path, disjoint_ratio=0.3):
    """Load and prepare all datasets, splitting all four EC levels together.

    Train keeps the disjoint message-passing/supervision split; val is made fully
    inductive. The test graph is transductive: it already carries training
    proteins' protein_ec* edges as message passing (the test proteins' own EC
    edges are excluded), so the EC nodes form class centroids.
    """
    train_data_raw = torch.load(train_path)
    val_data_raw = torch.load(val_path)
    test_data = torch.load(test_path)

    edge_types = [edges[0] for edges in LEVEL_EDGES.values()]
    rev_edge_types = [edges[1] for edges in LEVEL_EDGES.values()]

    # The test graph must already carry the injected (training-protein) EC edges.
    for et in edge_types:
        assert et in test_data.edge_types and test_data[et].get('edge_index') is not None \
            and test_data[et].edge_index.shape[1] > 0, \
            f'{et} has no message-passing edges in the test graph (expected a transductive test graph)'

    # split datasets into message passing and prediction edges. 70% used for message passing.
    transform = T.RandomLinkSplit(
        num_val=0.0,
        num_test=0.0,
        neg_sampling_ratio=0.0,
        disjoint_train_ratio=disjoint_ratio,
        add_negative_train_samples=False,
        edge_types=edge_types,
        rev_edge_types=rev_edge_types,
    )

    train_data, _, _ = transform(train_data_raw)
    val_data, _, _ = transform(val_data_raw)
    for et, rev_et in LEVEL_EDGES.values():
        make_inductive(val_data, et, rev_et)

    print('\nHierarchical dataset statistics:\n')
    print('----------------------------------TRAIN----------------------------------')
    for level, (et, _) in LEVEL_EDGES.items():
        print(f'  {level} ({et}): message-passing={train_data[et].edge_index.shape[1]}, '
              f'supervision={train_data[et].edge_label_index.shape[1]}')
    print('\n----------------------------------VALIDATION----------------------------------')
    for level, (et, _) in LEVEL_EDGES.items():
        print(f'  {level} ({et}): supervision={val_data[et].edge_label_index.shape[1]}')
    print('\n----------------------------------TEST----------------------------------')
    for level, (et, _) in LEVEL_EDGES.items():
        mp = test_data[et].edge_index.shape[1] if test_data[et].get('edge_index') is not None else 0
        print(f'  {level} ({et}): message-passing={mp}, supervision={test_data[et].edge_label_index.shape[1]}')
    print()

    for et, rev_et in LEVEL_EDGES.values():
        assert train_data[et].edge_index.shape[1] == train_data[rev_et].edge_index.shape[1]

    return train_data, val_data, test_data


def load_hierarchy_masks(data):
    """Static EC taxonomy relations for one split (parent/children maps per level)."""
    masks = {}
    for name, et in PARENT_EDGE_TYPES.items():
        row, col = data[et].edge_index
        masks[name] = {r.item(): c.item() for r, c in zip(row, col)}
    for name, et in CHILD_EDGE_TYPES.items():
        row, col = data[et].edge_index
        children = defaultdict(list)
        for r, c in zip(row.tolist(), col.tolist()):
            children[r].append(c)
        masks[name] = dict(children)
    return masks


def compute_hard_negative_pool(data, level, max_hard_per_positive=50, seed=42):
    """Precompute each positive's hierarchy-sibling hard negatives (L3/L4 only; empty for L1/L2)."""
    edge_type, _ = LEVEL_EDGES[level]
    pos_edge_index = data[edge_type].edge_label_index

    if level not in HARD_NEG_KEYS:
        return pos_edge_index, [[] for _ in range(pos_edge_index.shape[1])]

    parent_key, children_key = HARD_NEG_KEYS[level]
    masks = load_hierarchy_masks(data)

    full_pos_edge_index = torch.cat([data[edge_type].edge_index, pos_edge_index], dim=1)
    pos_by_protein = defaultdict(set)
    for p, n in zip(full_pos_edge_index[0].tolist(), full_pos_edge_index[1].tolist()):
        pos_by_protein[p].add(n)

    gen = torch.Generator().manual_seed(seed)
    hard_negs = []
    for p, n in zip(pos_edge_index[0].tolist(), pos_edge_index[1].tolist()):
        exclude = pos_by_protein[p]
        parent = masks[parent_key].get(n)
        siblings = [s for s in masks[children_key].get(parent, []) if s not in exclude] if parent is not None else []
        if len(siblings) > max_hard_per_positive:
            perm = torch.randperm(len(siblings), generator=gen)[:max_hard_per_positive]
            siblings = [siblings[i] for i in perm.tolist()]
        hard_negs.append(siblings)

    return pos_edge_index, hard_negs


def build_epoch_negatives(data, level, pos_edge_index, hard_negs, num_neg_per_positive, seed):
    """Write edge_label_index/edge_label (in place) as [1 pos + K neg] blocks: hard negatives + fresh random negatives."""
    edge_type, node_type = LEVEL_EDGES[level][0], LEVEL_NODE_TYPE[level]
    num_nodes = data[node_type].x.shape[0]
    n_pos = pos_edge_index.shape[1]
    K = num_neg_per_positive
    gen = torch.Generator().manual_seed(seed)

    # Exclusion set: never draw a protein's own true positive as a negative
    full_pos = torch.cat([data[edge_type].edge_index, pos_edge_index], dim=1)
    pos_by_protein = defaultdict(set)
    for p, n in zip(full_pos[0].tolist(), full_pos[1].tolist()):
        pos_by_protein[p].add(n)

    # Small vocabularies enumerate the eligible pool; the large L4 vocab uses rejection sampling
    SMALL_VOCAB = 1000
    all_nodes = list(range(num_nodes))

    block_protein, block_node, block_label = [], [], []
    n_hard_total = 0
    for (p, n), siblings in zip(zip(pos_edge_index[0].tolist(), pos_edge_index[1].tolist()), hard_negs):
        n_hard_total += len(siblings)
        combined = list(siblings)
        seen = pos_by_protein[p] | set(combined)
        n_easy = K - len(combined)

        if n_easy > 0:
            if num_nodes <= SMALL_VOCAB:
                eligible = [i for i in all_nodes if i not in seen]
                if len(eligible) >= n_easy:
                    idx = torch.randperm(len(eligible), generator=gen)[:n_easy].tolist()
                    combined.extend(eligible[i] for i in idx)
                else:
                    combined.extend(eligible)
                    while len(combined) < K:
                        combined.append(eligible[torch.randint(0, len(eligible), (1,), generator=gen).item()])
            else:
                while len(combined) < K:
                    c = torch.randint(0, num_nodes, (1,), generator=gen).item()
                    if c not in seen:
                        combined.append(c)
                        seen.add(c)

        block_protein.append(p)
        block_node.append(n)
        block_label.append(1.0)
        for c in combined:
            block_protein.append(p)
            block_node.append(c)
            block_label.append(0.0)

    block_size = 1 + K
    edge_index = torch.tensor([block_protein, block_node], dtype=torch.long).view(2, n_pos, block_size)
    label = torch.tensor(block_label).view(n_pos, block_size)

    block_perm = torch.randperm(n_pos, generator=gen)
    edge_index = edge_index[:, block_perm, :].reshape(2, n_pos * block_size)
    label = label[block_perm, :].reshape(n_pos * block_size)

    data[edge_type].edge_label_index = edge_index
    data[edge_type].edge_label = label

    return n_pos * K, n_hard_total


def add_hard_negatives(data, level, num_neg_per_positive, max_hard_per_positive=50, seed=42):
    """One-shot block-negative build for splits that do not need per-epoch refreshing (val)."""
    pos_edge_index, hard_negs = compute_hard_negative_pool(
        data, level, max_hard_per_positive=max_hard_per_positive, seed=seed)
    n_neg, n_hard_total = build_epoch_negatives(
        data, level, pos_edge_index, hard_negs, num_neg_per_positive, seed=seed)

    n_pos = pos_edge_index.shape[1]
    print(f'  {level} negatives: +{n_neg} pairs from {n_pos} positives '
          f'({n_hard_total} hard / {n_neg - n_hard_total} easy, block size {1 + num_neg_per_positive})')


def build_train_loader_for_level(train_data, level, config, num_workers=2):
    """Train loader over the pre-baked block negatives (rebuilt each epoch)."""
    edge_type, _ = LEVEL_EDGES[level]
    K = config[f'neg_sampling_ratio_{level}']
    # neg_sampling_ratio=0, shuffle=False, batch_size scaled by (1+K) keeps each batch a whole set of blocks
    return LinkNeighborLoader(
        train_data,
        num_neighbors=_expand_num_neighbors(config),
        neg_sampling_ratio=0.0,
        shuffle=False,
        edge_label=train_data[edge_type].edge_label,
        edge_label_index=(edge_type, train_data[edge_type].edge_label_index),
        batch_size=config['tr_batch_size'] * (1 + K),
        num_workers=num_workers,
        pin_memory=True,
    )


def create_level_loaders(train_data, val_data, test_data, level, config, num_workers=2):
    """Create train/val/test loaders for a single EC level."""
    edge_type, _ = LEVEL_EDGES[level]
    num_neighbors = _expand_num_neighbors(config)

    train_loader = build_train_loader_for_level(train_data, level, config, num_workers=num_workers)

    val_loader = LinkNeighborLoader(
        val_data,
        num_neighbors=num_neighbors,
        neg_sampling_ratio=0.0,
        shuffle=False,
        edge_label=val_data[edge_type].edge_label,
        edge_label_index=(edge_type, val_data[edge_type].edge_label_index),
        batch_size=int(len(val_data[edge_type].edge_label)),
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = LinkNeighborLoader(
        test_data,
        num_neighbors=num_neighbors,
        neg_sampling_ratio=0.0,
        shuffle=False,
        edge_label=test_data[edge_type].edge_label,
        edge_label_index=(edge_type, test_data[edge_type].edge_label_index),
        batch_size=int(len(test_data[edge_type].edge_label)),
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def sample_protein_subset(data, level, num_proteins, seed=42):
    """Fixed random subset of the proteins scored at `level` (None = all of them)."""
    edge_type, _ = LEVEL_EDGES[level]
    unique_proteins = data[edge_type].edge_label_index[0].unique()
    if num_proteins is None or num_proteins >= unique_proteins.shape[0]:
        return unique_proteins.tolist()
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(unique_proteins.shape[0], generator=gen)
    return unique_proteins[perm[:num_proteins]].tolist()


def _positive_pairs(data, level, protein_subset=None):
    """{protein: set(node)} true positives at `level`, from edge_index + label-1 edge_label_index."""
    edge_type, _ = LEVEL_EDGES[level]
    sup_edge_index = data[edge_type].edge_label_index
    sup_edge_label = data[edge_type].get('edge_label')
    if sup_edge_label is not None:
        sup_edge_index = sup_edge_index[:, sup_edge_label == 1]
    full_pos_edge_index = torch.cat([data[edge_type].edge_index, sup_edge_index], dim=1)

    protein_subset = set(protein_subset) if protein_subset is not None else None
    pos_by_protein = defaultdict(set)
    for p, e in zip(full_pos_edge_index[0].tolist(), full_pos_edge_index[1].tolist()):
        if protein_subset is None or p in protein_subset:
            pos_by_protein[p].add(e)
    return pos_by_protein


def build_full_matrix_loader(data, level, config, protein_subset, num_workers=2):
    """Loader over (protein_subset x all nodes at this level) with real 0/1 labels, for evaluation."""
    edge_type, _ = LEVEL_EDGES[level]
    node_type = LEVEL_NODE_TYPE[level]

    pos_by_protein = _positive_pairs(data, level, protein_subset)
    num_nodes = data[node_type].x.shape[0]

    protein_cols, node_cols, label_cols = [], [], []
    for p in protein_subset:
        protein_cols.append(torch.full((num_nodes,), p, dtype=torch.long))
        node_cols.append(torch.arange(num_nodes, dtype=torch.long))
        labels = torch.zeros(num_nodes)
        pos_nodes = pos_by_protein.get(p)
        if pos_nodes:
            labels[list(pos_nodes)] = 1.0
        label_cols.append(labels)

    edge_label_index = torch.stack([torch.cat(protein_cols), torch.cat(node_cols)], dim=0)
    edge_label = torch.cat(label_cols)

    matrix_data = data.clone()
    matrix_data[edge_type].edge_label_index = edge_label_index
    matrix_data[edge_type].edge_label = edge_label

    print(f'  {level} full matrix: {len(protein_subset)} proteins x {num_nodes} {node_type} '
          f'= {edge_label.shape[0]} pairs, {int(edge_label.sum().item())} positive')

    loader = LinkNeighborLoader(
        matrix_data,
        num_neighbors=_expand_num_neighbors(config),
        neg_sampling_ratio=0.0,
        shuffle=False,
        edge_label=matrix_data[edge_type].edge_label,
        edge_label_index=(edge_type, matrix_data[edge_type].edge_label_index),
        batch_size=int(len(matrix_data[edge_type].edge_label)),
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader, edge_label_index[0].numpy(), edge_label_index[1].numpy()
