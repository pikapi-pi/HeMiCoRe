import torch
from torch import nn
from .layers import HGNN_conv, HGNN_classifier, Local_Pooling, SAGPool, Global_Pooling, aggregation
import torch.nn.functional as F
from torch.autograd import Variable
from itertools import chain
import numpy as np
from collections import defaultdict


class HGNN(nn.Module):
    def __init__(self, in_dim: int, n_hid: int, out_dim: int, use_bn: bool = True, drop_rate: float = 0.5):
        super(HGNN, self).__init__()
        self.dropout = drop_rate
        self.hgc1 = HGNN_conv(in_dim, n_hid, drop_rate=drop_rate, use_bn=use_bn)
        self.hgc2 = HGNN_conv(n_hid, out_dim, drop_rate=drop_rate, use_bn=use_bn, is_last=True)
        self.readout = Global_Pooling()
        self.in_features = n_hid




    def forward(self, x: torch.Tensor, hg: "dhg.Hypergraph"):
        x = self.hgc1(x, hg)
        x = self.hgc2(x, hg)
        x_3 = self.readout(x)

        x = x_3

        return x.detach(), x

def generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    if type(H) != list:
        return _generate_G_from_H(H, variable_weight)
    else:
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H, variable_weight))
        return G


def _generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    torch.cuda.empty_cache()
    H = H.cuda().to(torch.float32)

    n_edge = H.shape[1]
    # the weight of the hyperedge
    W = torch.ones(n_edge).cuda()
    # the degree of the node
    DV = torch.sparse.sum(H * W, dim=1).cuda()
    # the degree of the hyperedge
    DE = torch.sparse.sum(H, dim=0).cuda()

    invDE = torch.diag(torch.pow(DE.to_dense(), -1))
    DV2 = torch.diag(torch.pow(DV.to_dense(), -0.5))
    W = torch.diag(W)
    HT = H.T

    # memory as sparse matrix in order to save the GPU memory
    invDE = torch.sparse_coo_tensor(indices=invDE.nonzero().t(),
                            values=invDE[invDE.nonzero().t()[0], invDE.nonzero().t()[1]],
                            size=invDE.size(),)
    DV2 = torch.sparse_coo_tensor(indices=DV2.nonzero().t(),
                            values=DV2[DV2.nonzero().t()[0], DV2.nonzero().t()[1]],
                            size=DV2.size(),)
    W = torch.sparse_coo_tensor(indices=W.nonzero().t(),
                            values=W[W.nonzero().t()[0], W.nonzero().t()[1]],
                            size=W.size(),)

    del DV, DE
    torch.cuda.empty_cache()

    if variable_weight:
        DV2_H = DV2 @ H
        invDE_HT_DV2 = invDE @ HT @ DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        # print(DV2.dtype, H.dtype, W.dtype, invDE.dtype)
        G = DV2 @ H @ W @ invDE @ HT @ DV2
        del H, HT, W, invDE, DV2
        torch.cuda.empty_cache()
        return G


# def data_initialise(HGNN_data, args):
#     X = HGNN_data['features']
#     Y = HGNN_data['labels']
#
#     # node features in sparse representation
#
#     # cuda
#     # args.Cuda = args.cuda and torch.cuda.is_available()
#     # if args.Cuda:
#     X, Y = X.cuda(), Y.cuda()
#
#     # update HGNN_data with torch autograd variable
#     HGNN_data['features'] = Variable(X)
#     HGNN_data['labels'] = Variable(Y)
#     del X, Y
#     torch.cuda.empty_cache()