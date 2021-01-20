import torch
import torch.nn as nn
from modules import GCNBlock
from modules import AttPoolBlock
from modules import DegreePickBlock
from modules import SoftPoolingGcnEncoder

class MuchPool(nn.Module):
    def __init__(self, args):
        super(MuchPool, self).__init__()
        self.args = args
        self.degreePick = DegreePickBlock(args)
        self.AttPool = AttPoolBlock(args)
        self.DiffPool = SoftPoolingGcnEncoder(args.diffPool_max_num_nodes, args.hidden_dim, args.hidden_dim, args.hidden_dim,
            args.diffPool_num_classes, args.diffPool_num_gcn_layer, args.hidden_dim, assign_ratio=args.diffPool_assign_ratio,
            num_pooling=args.diffPool_num_pool, bn=args.diffPool_bn, dropout=args.diffPool_dropout)
        if args.readout == 'mean':
            self.readout = self.mean_readout
        elif args.readout == 'sum':
            self.readout = self.sum_readout

    def forward(self, X, adj, mask):
        '''
        input:
            X:  node input features , [batch,node_num,input_dim],dtype=float
            adj: adj matrix, [batch,node_num,node_num], dtype=float
            mask: mask for nodes, [batch,node_num]
        outputs:
            out:unormalized classification prob, [batch,hidden_dim]
            H: batch of node hidden features, [batch,node_num,pass_dim]
            new_adj: pooled new adj matrix, [batch, k_max, k_max]
            new_mask: [batch, k_max]
        '''
        # result of DiffPool model
        assign_matrix, H_coarse = self.DiffPool(X, adj, mask)

        if self.args.local_topology and self.args.with_feature and self.args.global_topology:
            degree_based_index, H1, k_list1 = self.degreePick.forward(X, adj, mask, assign_matrix, H_coarse)
            feature_based_index, H2, k_list2 = self.AttPool(X, adj, mask, assign_matrix, H_coarse)

            index1 = [x[:k] for x, k in zip(degree_based_index.tolist(), k_list1)]
            index2 = [y[:k] for y, k in zip(feature_based_index.tolist(), k_list2)]
            intersection_index = [list(set(x) & set(y)) for x, y in zip(index1, index2)]
            union_index = [list(set(x) | set(y)) for x, y in zip(index1, index2)]
        elif not self.args.local_topology and self.args.with_feature and self.args.global_topology:
            H1 = None
            feature_based_index, H2, k_list2 = self.AttPool(X, adj, mask, assign_matrix, H_coarse)

            index1 = []
            index2 = [y[:k] for y, k in zip(feature_based_index.tolist(), k_list2)]
            intersection_index = index1
            union_index = index2
        elif self.args.local_topology and not self.args.with_feature and self.args.global_topology:
            degree_based_index, H1, k_list1 = self.degreePick.forward(X, adj, mask, assign_matrix, H_coarse)
            H2 = None

            index1 = [x[:k] for x, k in zip(degree_based_index.tolist(), k_list1)]
            index2 = []
            intersection_index = index2
            union_index = index1
        elif self.args.local_topology and self.args.with_feature and not self.args.global_topology:
            H_coarse = H_coarse / H_coarse
            degree_based_index, H1, k_list1 = self.degreePick.forward(X, adj, mask, assign_matrix, H_coarse)
            feature_based_index, H2, k_list2 = self.AttPool(X, adj, mask, assign_matrix, H_coarse)

            index1 = [x[:k] for x, k in zip(degree_based_index.tolist(), k_list1)]
            index2 = [y[:k] for y, k in zip(feature_based_index.tolist(), k_list2)]
            intersection_index = [list(set(x) & set(y)) for x, y in zip(index1, index2)]
            union_index = [list(set(x) | set(y)) for x, y in zip(index1, index2)]

        k_list = [len(x) for x in union_index]
        k_max = max(k_list)

        # for update embedding and adjacency matrix
        new_mask = X.new_zeros(X.shape[0], k_max)
        S_reserve = X.new_zeros(X.shape[0], k_max, adj.shape[-1])

        for i, k in enumerate(k_list):
            new_mask[i][0:k] = 1
            S_reserve[i, 0:k] = adj[i, union_index[i]]
        
        # update feature matrix and adjacency matrix
        new_adj = torch.matmul(torch.matmul(S_reserve, adj), torch.transpose(S_reserve, 1, 2))
        new_H = self.reconstruct_feature_matrix(H1, H2, index1, index2, union_index, intersection_index, k_max, k_list)
        return new_H, new_adj, new_mask

    def reconstruct_feature_matrix(self, H1, H2, index1, index2, union_index, intersection_index, k_max, k_list):
        difference_set1 = [list(set(x) - set(y)) for x, y in zip(index1, intersection_index)]
        difference_set2 = [list(set(x) - set(y)) for x, y in zip(index2, intersection_index)]
        if H1 is not None and H2 is not None:
            new_H = H1.new_zeros(H1.shape[0], k_max, H1.shape[-1])
            for i, k in enumerate(k_list):
                idx1 = [index1[i].index(x) for x in difference_set1[i]]
                idx_common = [union_index[i].index(x) for x in intersection_index[i]]
                idx2 = [index2[i].index(x) for x in difference_set2[i]]

                idx_common_new = [union_index[i].index(x) for x in intersection_index[i]]
                idx_common_origin1 = [index1[i].index(x) for x in intersection_index[i]]
                idx_common_origin2 = [index2[i].index(x) for x in intersection_index[i]]

                idx_new_1 = [union_index[i].index(x) for x in difference_set1[i]]
                idx1 = [index1[i].index(x) for x in difference_set1[i]]

                idx_new_2 = [union_index[i].index(x) for x in difference_set2[i]]
                idx2 = [index2[i].index(x) for x in difference_set2[i]]

                new_H[i, idx_common_new] = (H1[i, idx_common_origin1] + H2[i, idx_common_origin2]) / 2
                new_H[i, idx_new_1] = H1[i, idx1]
                new_H[i, idx_new_2] = H2[i, idx2]
        elif H1 is not None and H2 is None:
            new_H = H1
        elif H1 is not None and H2 is not None:
            new_H = H2
        
        return new_H


    def mean_readout(self, H):
        return torch.mean(H, dim=1)

    def sum_readout(self, H):
        return torch.sum(H, dim=1)