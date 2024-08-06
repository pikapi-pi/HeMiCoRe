from torch import nn
import torch

class CL(nn.Module):
    def __init__(self, encoder, projection_dim, n_features):
        super(CL, self).__init__()
        self.encoder = encoder

        self.projection_dim = projection_dim
        self.n_features = n_features

    def forward(self, graph_views):
        assert isinstance(graph_views, list) and len(graph_views[0]) == 2, f""
        h_views = [self.encoder(x[0], x[1])[1] for x in graph_views]
        return h_views, [h.detach() for h in h_views]