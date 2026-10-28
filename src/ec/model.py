import torch
from torch_geometric.nn import HGTConv, Linear
from torch_geometric.nn import MLP

# EC hierarchy levels and the graph node type each one is predicted against
LEVELS = ['l1', 'l2', 'l3', 'l4']
LEVEL_TARGET_TYPE = {
    'l1': 'EC_class_1',
    'l2': 'EC_class_2',
    'l3': 'EC_class_3',
    'l4': 'EC_number',
}


# Define the model backbone (shared across all EC levels)
class ProtHGT(torch.nn.Module):
    """
    Args:
        data: Graph data object containing node features and edge indices
        hidden_channels (int): Number of hidden channels in the model
        num_heads (int): Number of attention heads
        num_layers (int): Number of HGT layers
        mlp_hidden_layers (list): List of hidden layer dimensions for the MLP
        mlp_dropout (float): Dropout rate for the MLP
    """
    def __init__(self, data, hidden_channels, num_heads, num_layers, mlp_hidden_layers, mlp_dropout):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict({
            node_type: Linear(data.x_dict[node_type].size(-1), hidden_channels)
            for node_type in data.node_types
        })

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(), num_heads, group='sum')
            self.convs.append(conv)

        # Normalize each skip-connection branch before concatenation
        self.initial_norm = torch.nn.LayerNorm(hidden_channels)
        self.gnn_norm = torch.nn.LayerNorm(hidden_channels)

        self.mlp = MLP(mlp_hidden_layers, dropout=mlp_dropout, norm=None)

    def generate_embeddings(self, x_dict, edge_index_dict):
        # Pre-GNN projection, kept as a skip source
        initial_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }

        gnn_dict = initial_dict
        for conv in self.convs:
            out = conv(gnn_dict, edge_index_dict)
            # Carry a node type forward if it received no messages in this batch
            gnn_dict = {
                node_type: (out[node_type] if out.get(node_type) is not None else prev)
                for node_type, prev in gnn_dict.items()
            }

        # Concat skip connection: normalized pre-GNN and post-GNN embeddings
        return {
            node_type: torch.cat([
                self.initial_norm(initial_dict[node_type]),
                self.gnn_norm(gnn_dict[node_type]),
            ], dim=-1)
            for node_type in initial_dict
        }

    def forward(self, x_dict, edge_index_dict, edge_label_index, target_type):
        # Get updated embeddings
        x_dict = self.generate_embeddings(x_dict, edge_index_dict)

        # Make predictions
        row, col = edge_label_index
        z = torch.cat([x_dict["Protein"][row], x_dict[target_type][col]], dim=-1)

        return self.mlp(z).view(-1), x_dict


# Hierarchical model: shared ProtHGT backbone + one MLP decoder per EC level
class ProtHGTHier(torch.nn.Module):
    """Hierarchical model: shared ProtHGT backbone + one MLP decoder per EC level"""
    def __init__(self, data, hidden_channels, num_heads, num_layers, mlp_hidden_layers, mlp_dropout):
        super().__init__()
        self.backbone = ProtHGT(
            data, hidden_channels, num_heads, num_layers, mlp_hidden_layers, mlp_dropout,
        )
        # backbone.mlp is unused here (one decoder per level instead)
        self.backbone.mlp = torch.nn.Identity()

        self.decoders = torch.nn.ModuleDict({
            level: MLP(mlp_hidden_layers, dropout=mlp_dropout, norm=None)
            for level in LEVELS
        })

    def generate_embeddings(self, x_dict, edge_index_dict):
        return self.backbone.generate_embeddings(x_dict, edge_index_dict)

    def forward(self, x_dict, edge_index_dict, edge_label_index, level):
        x_dict = self.generate_embeddings(x_dict, edge_index_dict)
        target_type = LEVEL_TARGET_TYPE[level]
        row, col = edge_label_index
        z = torch.cat([x_dict['Protein'][row], x_dict[target_type][col]], dim=-1)
        return self.decoders[level](z).view(-1), x_dict

    def backbone_parameters(self):
        """All trainable backbone parameters (everything except the decoders)."""
        return list(self.backbone.lin_dict.parameters()) \
            + list(self.backbone.convs.parameters()) \
            + list(self.backbone.initial_norm.parameters()) \
            + list(self.backbone.gnn_norm.parameters())
