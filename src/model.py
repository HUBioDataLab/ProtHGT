import torch
from torch_geometric.nn import HGTConv, Linear
from torch_geometric.nn import MLP

# Define the model
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
    def __init__(self, data,hidden_channels, num_heads, num_layers, mlp_hidden_layers, mlp_dropout):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict({
            node_type: Linear(data.x_dict[node_type].size(-1), hidden_channels)
            for node_type in data.node_types
        })

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(), num_heads, group='sum')
            self.convs.append(conv)
        
        self.mlp = MLP(mlp_hidden_layers , dropout=mlp_dropout, norm=None)

    def generate_embeddings(self, x_dict, edge_index_dict):
        # Generate updated embeddings through the GNN layers
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            
        return x_dict

    def forward(self, x_dict, edge_index_dict, tr_edge_label_index, target_type):
        # Get updated embeddings
        x_dict = self.generate_embeddings(x_dict, edge_index_dict)

        # Make predictions
        row, col = tr_edge_label_index
        z = torch.cat([x_dict["Protein"][row], x_dict[target_type][col]], dim=-1)

        return self.mlp(z).view(-1), x_dict
