import torch
import torch_geometric

import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential as Seq, Linear, ReLU, Sigmoid




class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, L=3, bias=True):
        super().__init__()

        layers = []
        layers.append(nn.Linear(input_size, hidden_dim, bias=bias))
        for _l in range(1, L - 1):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_size, bias=bias))
        self.layers = nn.ModuleList(layers)

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x





class IN(MessagePassing):
    def __init__(
        self,
        node_indim: int,
        edge_indim: int,
        node_outdim=3,
        edge_outdim=4,
        node_hidden_dim=40,
        edge_hidden_dim=40,
        aggr="add",
        L=3,
    ):
        super().__init__(aggr=aggr, flow="source_to_target")
        self.relational_model = MLP(
            2 * node_indim + edge_indim,
            edge_outdim,
            edge_hidden_dim,
            L=L,
        )
        self.object_model = MLP(
            node_indim + edge_outdim,
            node_outdim,
            node_hidden_dim,
            L=L
        )


    def forward(self, x, edge_index, edge_attr):
        x_tilde = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)
        return x_tilde, self.E_tilde

    def message(self, x_i, x_j, edge_attr):
        # x_i --> incoming, x_j --> outgoing
        m = torch.cat([x_i, x_j, edge_attr], dim=1)
        self.E_tilde = self.relational_model(m)
        return self.E_tilde

    def update(self, aggr_out, x):
        c = torch.cat([x, aggr_out], dim=1)
        return self.object_model(c)





class InteractionNetwork(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=8, edge_dim=4, output_dim=4, n_iters=1, aggr='add',
#                 norm=torch.tensor([1./500., 1./500., 1./54., 1/25., 1./1000.])):
                 # norm=torch.tensor([1./500., 1./500., 1./54.])):
                 norm=torch.tensor([1., 1., 1.])):

        node_indim = input_dim
        edge_indim = edge_dim
        L = 4                                 #Number of IN blocks
        node_latentdim = 4 * input_dim
        edge_latentdim = 3 * edge_dim
        r_hidden_size = hidden_dim
        o_hidden_size = hidden_dim
        hidden_layers = 2



        super().__init__()
        self.node_encoder = MLP(node_indim, node_latentdim, r_hidden_size, L=1)
        self.edge_encoder = MLP(edge_indim, edge_latentdim, o_hidden_size, L=1)
        gnn_layers = []
        for _l in range(L):
            gnn_layers.append(
                IN(
                    node_latentdim,
                    edge_latentdim,
                    node_outdim=node_latentdim,
                    edge_outdim=edge_latentdim,
                    edge_hidden_dim=r_hidden_size,
                    node_hidden_dim=o_hidden_size,
                    L=hidden_layers
                )
            )
        self.gnn_layers = nn.ModuleList(gnn_layers)
        self.W = MLP(2*node_latentdim + edge_latentdim, 1, r_hidden_size, L=hidden_layers)


    def forward(self, data):
        x = data.x
        edge_index, edge_attr = data.edge_index, data.edge_attr

        node_latent = self.node_encoder(x)
        edge_latent = self.edge_encoder(edge_attr)
        # node_latent = x
        # edge_latent = edge_attr

        for i in range(2):    #number of times IN blocks are reused
            for layer in self.gnn_layers:
                node_latent, edge_latent = layer(node_latent, edge_index, edge_latent)

        m2 = torch.cat([node_latent[edge_index[1]],
                        node_latent[edge_index[0]],
                        edge_latent], dim=1)
        edge_weights = torch.sigmoid(self.W(m2))

        return edge_weights
