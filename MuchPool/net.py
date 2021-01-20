import torch
import torch.nn as nn
from modules import GCNBlock
from classifier import MLPClassifier
from muchPool import MuchPool
from gat import GAT
from graphSage import SageGCN

class Net(nn.Module):
    def __init__(self, args, convolution_method):
        super(Net, self).__init__()
        self.hierarchical_num = args.hierarchical_num
        embed_method = convolution_method
        self.embeds = nn.ModuleList()
        if embed_method == 'GCN':
            self.embed = GCNBlock(args.input_dim, args.hidden_dim, args.bn, args.gcn_res, args.gcn_norm, args.dropout, args.relu)
            for i in range(self.hierarchical_num):
                self.embeds.append(GCNBlock(args.hidden_dim, args.hidden_dim, args.bn, args.gcn_res, args.gcn_norm, args.dropout, args.relu))
        elif embed_method == 'GAT':
            self.embed = GAT(args.input_dim, args.hidden_dim, args.dropout, 0.2, 2)
            for i in range(self.hierarchical_num):
                self.embeds.append(GAT(args.hidden_dim, args.hidden_dim, args.dropout, 0.2, 2))
        elif embed_method == 'GraphSage':
            self.embed = SageGCN(args.input_dim, args.hidden_dim)
            for i in range(self.hierarchical_num):
                self.embeds.append(SageGCN(args.hidden_dim, args.hidden_dim))
        self.muchPools = nn.ModuleList()
        for i in range(self.hierarchical_num):
            self.muchPools.append(MuchPool(args))
        if args.readout == 'mean':
            self.readout = self.mean_readout
        elif args.readout == 'sum':
            self.readout = self.sum_readout
        self.mlpc = MLPClassifier(input_size=args.hidden_dim, hidden_size=args.hidden_dim, num_class=args.num_class)

    def forward(self, xs, adjs, masks):
        H = self.embed(xs, adjs, masks)
        for i in range(self.hierarchical_num):
            H = self.embeds[i](H, adjs, masks)
            H, adjs, masks = self.muchPools[i](H, adjs, masks)
        Z = self.readout(H)
        logits = self.mlpc(Z)
        return logits

    def mean_readout(self, H):
        return torch.mean(H, dim=1)

    def sum_readout(self, H):
        return torch.sum(H, dim=1)