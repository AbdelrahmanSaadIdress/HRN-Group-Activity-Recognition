import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class RelationLayer(MessagePassing):
    def __init__(self,
                in_channels=2048,
                hidden_channels=512,
                output_channels=128):

        super().__init__(aggr='sum')

        self.message_mlp = nn.Sequential(
            nn.Linear(in_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )

        self.update_mlp = nn.Sequential(
            nn.Linear(in_channels + hidden_channels,
                    output_channels),
            nn.ReLU()
        )

    def forward(self, input_, edge_index):
        """
        x: (num_nodes, 2048)
        edge_index: (2, num_edges)
        """

        return self.propagate(
            edge_index=edge_index,
            x=input_
        )

    def message(self, x_i, x_j):
        """
        x_i: target node feature
        x_j: source node feature

        shape:
            (num_edges, 2048)
        """

        pair = torch.cat([x_i, x_j], dim=-1)

        return self.message_mlp(pair)

    def update(self, aggr_out, x):
        """
        aggr_out:
            aggregated messages
            (num_nodes, hidden_channels)

        x:
            original node feature
            (num_nodes, 2048)
        """

        out = torch.cat([x, aggr_out], dim=-1)

        return self.update_mlp(out)