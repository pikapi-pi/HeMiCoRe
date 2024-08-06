import math

import dhg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .hypergraph_utils import topk, get_batch_id

class HGNN_conv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, use_bn: bool = False, drop_rate: float = 0.5,
        is_last: bool = False,):
        super(HGNN_conv, self).__init__()

        # self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        # if bias:
        #     self.bias = Parameter(torch.Tensor(out_ft))
        # else:
        #     self.register_parameter('bias', None)
        # self.reset_parameters()
        self.is_last = is_last
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.act = nn.LeakyReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)

    # def reset_parameters(self):
    #     stdv = 1. / math.sqrt(self.weight.size(1))
    #     self.weight.data.uniform_(-stdv, stdv)
    #     if self.bias is not None:
    #         self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, Hg):
        # if len(x.shape) < 3:
        #     x = x.unsqueeze(0)
        # b, n, f = x.shape
        # # print(b,n,f)
        # x = torch.mm(x.reshape(-1, f), self.weight)
        # if self.bias is not None:
        #     x = x + self.bias
        # # print(x.dtype, G.dtype)
        # f = x.shape[-1]
        # x = torch.bmm(G, x.reshape(-1, n, f))
        # return x
        if isinstance(Hg, list):
            if len(x.shape) < 3:
                x = x.unsqueeze(0)
            b, n, f = x.shape
            x = self.theta(x.reshape(-1, f))
            if self.bn is not None:
                x = self.bn(x)
            # print(Hg[0].L_HGNN.shape)
            G = [hg.L_HGNN.unsqueeze(0) for hg in Hg]
            G = torch.cat(G, dim=0).to(x.device)
            f = x.shape[-1]
            x = torch.bmm(G, x.reshape(-1, n, f))
            if not self.is_last:
                x = self.drop(self.act(x))
        else:
            x = self.theta(x)
            if self.bn is not None:
                x = self.bn(x)
            x = Hg.smoothing_with_HGNN(x)
            if not self.is_last:
                x = self.drop(self.act(x))
        return x





class Attention_Pooling(nn.Module):
    def __init__(self, in_ch, out_ch: int = 1):
        super(Attention_Pooling, self).__init__()
        self.AP = nn.Linear(in_ch, out_ch, bias=True, batch_first=True)

    def forward(self, x):
        return self.AP(x)



class HGNN_classifier(nn.Module):
    def __init__(self, n_hid, n_class):
        super(HGNN_classifier, self).__init__()
        self.fc1 = nn.Linear(n_hid, n_class)

    def forward(self, x):
        x = self.fc1(x)
        return x

class aggregation(torch.nn.Module):
    def __init__(self, n_edge: int, bias=False):
        super(aggregation, self).__init__()
        self.n_edge = n_edge
        self.edge_weights = Parameter(torch.Tensor(n_edge, 1))
        # self.node_weigts = Parameter(torch.Tensor(n_node, 1))
        if bias:
            self.bias = Parameter(torch.Tensor(1))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.edge_weights.size(1))
        self.edge_weights.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x_node, x_edge):
        x_node_readout = torch.mean(x_node, dim=-2).unsqueeze(0)
        x_edge_readout = torch.mm(x_edge.T, self.edge_weights).T
        if self.bias is not None:
            x_edge_readout = x_edge_readout + self.bias
        x = torch.mean(torch.cat((x_node_readout, x_edge_readout), dim=0), dim=0)
        return x

class SAGPool(torch.nn.Module):
    def __init__(
            self,
            in_dim: int,
            ratio=0.2,
            conv_op=HGNN_conv,
            non_linearity=torch.tanh,
    ):
        super(SAGPool, self).__init__()
        self.in_dim = in_dim
        self.ratio = ratio
        self.score_layer = conv_op(in_dim, 1)
        self.non_linearity = non_linearity

    def forward(self, x, G):
        score = self.score_layer(x, G)

        attention_indices, num_nodes = topk(
            score.view(1, -1),
            self.ratio,
            get_batch_id(torch.tensor(x.shape[-2]).unsqueeze(0)).cuda(),
            torch.tensor(x.shape[-2]).unsqueeze(0).cuda()
        )
        # print(x[attention_indices].shape)
        # print(score[attention_indices].shape)
        feature = x[attention_indices] * self.non_linearity(score[attention_indices]).view(-1, 1)
        mean_readout = torch.mean(feature, dim=-2)
        return mean_readout

# class Attention_Pooling(nn.Module):
#     def __init__(self):
#         super(Attention_Pooling, self).__init__()
#
#     def forward(self, feat, indices):
#         selected_feat = torch.gather(feat, 0, indices.unsqueeze(1).expand(-1, feat.shape[-1]))
#         mean_out = torch.mean(selected_feat, dim=-2)
#         return mean_out

class Local_Pooling(nn.Module):
    """
    Simple average pooing layer for node feature
    """

    def __init__(self):
        super(Local_Pooling, self).__init__()

    def forward(self, feat, indices):
        selected_feat = torch.gather(feat, 0, indices.unsqueeze(1).expand(-1, feat.shape[-1]))
        mean_out = torch.mean(selected_feat, dim=-2)
        return mean_out

class Global_Pooling(nn.Module):
    """
    Simple average pooing layer for node feature
    """

    def __init__(self):
        super(Global_Pooling, self).__init__()

    def forward(self, feat):
        # print(f"In Global_Pooling, feat'shape is {feat.shape}")
        # selected_feat = torch.gather(feat, 0, indices.unsqueeze(1).expand(-1, feat.shape[-1]))
        mean_out = torch.mean(feat, dim=-2)
        max_out, _ = torch.max(feat, dim=1)
        # return mean_out + max_out
        return mean_out