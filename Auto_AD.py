import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import math
import dgl
import sympy
import scipy
import numpy as np
from torch import nn
from torch.nn import init
from dgl.nn.pytorch import GraphConv, EdgeWeightNorm, ChebConv, GATConv, HeteroGraphConv




class Edge_mask_learner(nn.Module):
    def __init__(self, embedding_dim, src, dst, device):
        super(Edge_mask_learner, self).__init__()
        # self.g = g
        self.device = device
        self.embedding_dim = embedding_dim
        self.linear1 = nn.Linear(embedding_dim*3, embedding_dim).to(self.device)
        self.linear2 = nn.Linear(embedding_dim, 1).to(self.device)
        self.act1 = nn.LeakyReLU()
        self.act2 = nn.Sigmoid()
        self.src = src
        self.dst = dst
        self.bias = 0.0+0.0001
        self.tmp = 0.1
    def forward(self, feature):
        edge_feature = torch.cat([feature[self.src]-feature[self.dst], feature[self.src]-torch.mean(feature, dim = 0), feature[self.dst]-torch.mean(feature, dim = 0)],-1)
        edge_feature = self.linear1(edge_feature)
        
        edge_feature = self.act1(edge_feature)
        edge_feature = self.linear2(edge_feature)
        edge_feature = self.act2(edge_feature)
        edge_feature = edge_feature.squeeze(-1)

        eps = ((self.bias - (1 - self.bias)) * torch.rand(edge_feature.size()) + (1 - self.bias)).to(self.device)
        edge_gate_inputs = torch.log(eps) - torch.log(1 - eps)
        edge_gate_inputs = edge_gate_inputs.to(self.device)
        edge_gate_inputs = (edge_gate_inputs + edge_feature) / self.tmp
        edge_mask = torch.sigmoid(edge_gate_inputs)
        return edge_mask

class PolyConv_new(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 theta,
                 activation=F.leaky_relu,
                 lin=False,
                 bias=False):
        super(PolyConv_new, self).__init__()
        self._theta = theta
        self._k = len(self._theta)
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.activation = activation
        self.linear = nn.Linear(in_feats, out_feats, bias)
        self.lin = lin

    def reset_parameters(self):
        if self.linear.weight is not None:
            init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            init.zeros_(self.linear.bias)

    def forward(self, graph, feat):

        def getDegree(graph):
            graph.update_all(fn.copy_e('mask', 'm'), fn.sum('m', 'degree'))
            return graph.ndata.pop('degree')

        def unnLaplacian(feat, D_invsqrt, graph):
            """ Operation Feat * D^-1/2 A D^-1/2 """
            graph.ndata['h'] = feat * D_invsqrt
            graph.update_all(fn.u_mul_e('h', 'mask', 'm'), fn.sum('m', 'h'))
            return feat - graph.ndata.pop('h') * D_invsqrt

        with graph.local_scope():
            degrees = getDegree(graph)
            D_invsqrt = torch.pow(degrees.float().clamp(
                min=1), -0.5).unsqueeze(-1).to(feat.device)
            h = self._theta[0]*feat
            for k in range(1, self._k):
                feat = unnLaplacian(feat, D_invsqrt, graph)
                h += self._theta[k]*feat
        if self.lin:
            h = self.linear(h)
            h = self.activation(h)
        return h

class Auto_AD(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, graph, d=2, batch=False):
        super(Auto_AD, self).__init__()
        self.g = graph
        self.src = self.g.edges()[0]
        self.dst = self.g.edges()[1]
        self.thetas = calculate_theta2(d=d)
        self.conv = []
        # self.learner = []
        self.learner = Edge_mask_learner(h_feats, self.src, self.dst, 'cuda:0')
        for i in range(len(self.thetas)):
            if not batch:
                self.conv.append(PolyConv_new(h_feats, h_feats, self.thetas[i], lin=False))
                # self.learner.append(Edge_mask_learner(h_feats, self.src, self.dst, 'cuda:0'))
            # else:
            #     self.conv.append(PolyConvBatch(h_feats, h_feats, self.thetas[i], lin=False))
        self.linear = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        
        self.linear3 = nn.Linear(h_feats*len(self.conv), h_feats)
        self.linear3_1 = nn.Linear(h_feats*len(self.conv)*2, h_feats)
        self.linear4 = nn.Linear(h_feats, num_classes)
        self.act = nn.ReLU()
        self.d = d

    def forward(self, in_feat):
        h = self.linear(in_feat)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)
        h_final1 = torch.zeros([len(in_feat), 0]).to('cuda:0')
        for conv in self.conv:
            self.g.edata['mask']= torch.ones(self.g.num_edges()).to('cuda:0')
            h0 = conv(self.g, h)
            # self.g.edata['mask'] = learner(h0)
            h_final1 = torch.cat([h_final1, h0], -1)
        
        h1 = self.linear3(h_final1)
        mask = self.learner(h1)
        self.g.edata['mask']=mask
        for conv in self.conv:
            h0 = conv(self.g, h)
            # h_final1 = torch.cat([h_final1, h0], -1)
            h_final1 = torch.cat([h_final1, h0], -1)
        h = self.linear3_1(h_final1)
        h = self.act(h)
        h = self.linear4(h)
        return h

def calculate_theta2(d):
    thetas = []
    x = sympy.symbols('x')
    for i in range(d+1):
        f = sympy.poly((x/2) ** i * (1 - x/2) ** (d-i) / (scipy.special.beta(i+1, d+1-i)))
        coeff = f.all_coeffs()
        inv_coeff = []
        for i in range(d+1):
            inv_coeff.append(float(coeff[d-i]))
        thetas.append(inv_coeff)
    return thetas
