import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import math
import numpy as np

# GCN basic operation
class GCNBlock(nn.Module):
    def __init__(self, input_dim, output_dim, bn=0, add_self=0, normalize_embedding=0,
                 dropout=0.0, relu=0, bias=True):
        super(GCNBlock, self).__init__()
        self.add_self = add_self
        self.dropout = dropout
        self.relu = relu
        self.bn = bn
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        if self.bn:
            self.bn_layer = torch.nn.BatchNorm1d(output_dim)

        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        nn.init.xavier_normal_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))
        else:
            self.bias = None

    def forward(self, x, adj, mask):
        y = torch.matmul(adj, x)
        if self.add_self:
            y += x
        y = torch.matmul(y, self.weight)
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)
        if self.bn:
            index = mask.sum(dim=1).long().tolist()
            bn_tensor_bf = mask.new_zeros((sum(index), y.shape[2]))
            bn_tensor_af = mask.new_zeros(*y.shape)
            start_index = []
            ssum = 0
            for i in range(x.shape[0]):
                start_index.append(ssum)
                ssum += index[i]
            start_index.append(ssum)
            for i in range(x.shape[0]):
                bn_tensor_bf[start_index[i]
                    :start_index[i+1]] = y[i, 0:index[i]]
            bn_tensor_bf = self.bn_layer(bn_tensor_bf)
            for i in range(x.shape[0]):
                bn_tensor_af[i, 0:index[i]
                             ] = bn_tensor_bf[start_index[i]:start_index[i+1]]
            y = bn_tensor_af
        if self.dropout > 0.001:
            y = self.dropout_layer(y)
        if self.relu == 'relu':
            y = torch.nn.functional.relu(y)
        elif self.relu == 'lrelu':
            y = torch.nn.functional.leaky_relu(y, 0.1)
        return y


# GCN basic operation
class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, add_self=False, normalize_embedding=False,
            dropout=0.0, bias=True):
        super(GraphConv, self).__init__()
        self.add_self = add_self
        self.dropout = dropout
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).cuda())
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim).cuda())
        else:
            self.bias = None

    def forward(self, x, adj):
        if self.dropout > 0.001:
            x = self.dropout_layer(x)
        y = torch.matmul(adj, x)
        if self.add_self:
            y += x
        y = torch.matmul(y,self.weight)
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)
        return y

class DegreePickBlock(object):
    def __init__(self, config):
        self.filt_percent = config.percent
        self.assign_ratio = config.diffPool_assign_ratio    # diffpool module assign ratio
        self.max_node_num = config.diffPool_max_num_nodes   # maximum node number in dataset
        self.inter_channel_gcn = InterChannelGCN(config.hidden_dim, config.hidden_dim)

    def forward(self, X, adj, mask, assign_matrix, H_coarse):
        '''
        input:
            X:  node input features , [batch,node_num,input_dim],dtype=float
            adj: adj matrix, [batch,node_num,node_num], dtype=float
            mask: mask for nodes, [batch,node_num]
            assign_matrix: assign matrix in diffpool module, [batch, node_num, cluster_num]
            H_coarse: embedding matrix of coarse graph, [batch, cluster_num, hidden_dim], dtype=float
        outputs:
            out: unormalized classification prob, [batch,hidden_dim]
            H: batch of node hidden features, [batch,node_num,pass_dim]
            new_adj: pooled new adj matrix, [batch, k_max, k_max]
            new_mask: [batch, k_max]
        '''
        k_max = int(math.ceil(self.filt_percent * adj.shape[-1]))
        k_list = [int(math.ceil(self.filt_percent * x)) for x in mask.sum(dim=1).tolist()]

        # for inter_channel convolution
        cluster_num = int(self.max_node_num * self.assign_ratio)

        degrees = adj.sum(dim=-1)

        _, top_index = torch.topk(degrees, k_max, dim=1)

        # for update embedding and adjacency matrix
        new_mask = X.new_zeros(X.shape[0], k_max)
        S_reserve = X.new_zeros(X.shape[0], k_max, adj.shape[-1])
        inter_channel_adj = X.new_zeros(X.shape[0], k_max, cluster_num)

        # new_mask[:, 0:k] = 1
        for i, k in enumerate(k_list):
            new_mask[i][0:k] = 1
            S_reserve[i, 0:k] = adj[i, top_index[i, :k]]
            inter_channel_adj[i, 0:k] = assign_matrix[i, top_index[i, :k]]
            # for j in range(k):
            #     # new_mask[i][j] = 1
            #     S_reserve[i][j] = adj[i][top_index[i][j]]
            #     inter_channel_adj[i][j] = assign_matrix[i][top_index[i][j]]

        H = torch.matmul(S_reserve, X)
        H = self.inter_channel_gcn(H, H_coarse, inter_channel_adj)
        new_adj = torch.matmul(torch.matmul(S_reserve, adj), torch.transpose(S_reserve, 1, 2))

        return top_index, H, k_list

class AttPoolBlock(nn.Module):
    def __init__(self, config):
        super(AttPoolBlock, self).__init__()
        # self.gcns = nn.ModuleList()
        # self.gcns.append(GCNBlock(config.input_dim, config.hidden_dim, config.bn, config.gcn_res, config.gcn_norm, config.dropout, config.relu))
        self.gcn = GCNBlock(config.hidden_dim, config.hidden_dim, config.bn, config.gcn_res, config.gcn_norm, config.dropout, config.relu)
        self.inter_channel_gcn = InterChannelGCN(config.hidden_dim, config.hidden_dim)
        self.filt_percent = config.percent
        self.assign_ratio = config.diffPool_assign_ratio # diffpool module assign ratio
        self.max_node_num = config.diffPool_max_num_nodes   # maximum node number in dataset
        self.w = nn.Parameter(torch.zeros(config.hidden_dim, config.hidden_dim))
        torch.nn.init.normal_(self.w)

    def forward(self, X, adj, mask, assign_matrix, H_coarse):
        '''
        input:
            X:  node input features , [batch,node_num,input_dim],dtype=float
            adj: adj matrix, [batch,node_num,node_num], dtype=float
            mask: mask for nodes, [batch,node_num]
            assign_matrix: assign matrix in diffpool module, [batch, node_num, next_layer_node_num]
            H_coarse: embedding matrix of coarse graph, [batch, cluster_num, hidden_dim], dtype=float
        outputs:
            out: unormalized classification prob, [batch,hidden_dim]
            H: batch of node hidden features, [batch,node_num,pass_dim]
            new_adj: pooled new adj matrix, [batch, k_max, k_max]
            new_mask: [batch, k_max]
        '''
        # hidden = self.gcn(X, adj, mask)
        # hidden = mask.unsqueeze(2)*hidden
        
        # x = hidden
        hidden = self.readout(X)
        reference_hidden = F.relu(torch.matmul(hidden, self.w))
        k_max = int(math.ceil(self.filt_percent * adj.shape[-1]))
        k_list = [int(math.ceil(self.filt_percent * x)) for x in mask.sum(dim=1).tolist()]

        reference_hidden = reference_hidden.unsqueeze(1)
        inner_prod = torch.mul(X, reference_hidden).sum(dim=-1)
        scores = F.softmax(inner_prod, dim=1)

        _, top_index = torch.topk(scores, k_max, dim=1)

        # for update embedding and adjacency matrix
        new_mask = X.new_zeros(X.shape[0], k_max)
        S_reserve = X.new_zeros(X.shape[0], k_max, adj.shape[-1])

        # for inter-channel convolution
        cluster_num = int(self.max_node_num * self.assign_ratio)
        inter_channel_adj = X.new_zeros(X.shape[0], k_max, cluster_num)
        
        for i, k in enumerate(k_list):
            new_mask[i][0:k] = 1
            S_reserve[i, 0:k] = adj[i, top_index[i, :k]]
            inter_channel_adj[i, 0:k] = assign_matrix[i, top_index[i, :k]]

        H = torch.matmul(S_reserve, X)
        H = self.inter_channel_gcn(H, H_coarse, inter_channel_adj)
        new_adj = torch.matmul(torch.matmul(S_reserve, adj), torch.transpose(S_reserve, 1, 2))
        return top_index, H, k_list

    def readout(self, x):
        return x.sum(dim=1)

#Inter-channel GCN Block
class InterChannelGCN(nn.Module):
    def __init__(self, input_dim, output_dim, add_self=True, normalize=False):
        super(InterChannelGCN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.add_self = add_self
        self.normalize = normalize
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim)).cuda()
        nn.init.xavier_normal_(self.weight)

    def forward(self, H_fine, H_coarse, inter_channel_adj):
        out = torch.matmul(inter_channel_adj, H_coarse)
        if self.add_self:
            out += H_fine
        out = torch.matmul(out, self.weight)
        out = F.relu(out)
        if self.normalize:
            out = F.normalize(out)
        return out

class GcnEncoderGraph(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
            pred_hidden_dims=[], concat=True, bn=True, dropout=0.0, args=None):
        super(GcnEncoderGraph, self).__init__()
        self.concat = concat
        add_self = not concat
        self.bn = bn
        self.num_layers = num_layers
        self.num_aggs=1

        self.bias = True
        if args is not None:
            self.bias = args.diffPool_bias

        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layers(
                input_dim, hidden_dim, embedding_dim, num_layers, 
                add_self, normalize=True, dropout=dropout)
        pred_input_dim = hidden_dim * (2 + len(self.conv_block))
        self.transform = self.build_pred_layers(pred_input_dim, [], hidden_dim)
        self.act = nn.ReLU()
        self.label_dim = label_dim

        if concat:
            self.pred_input_dim = hidden_dim * (num_layers - 1) + embedding_dim
        else:
            self.pred_input_dim = embedding_dim
        self.pred_model = self.build_pred_layers(self.pred_input_dim, pred_hidden_dims, 
                label_dim, num_aggs=self.num_aggs)

        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def build_conv_layers(self, input_dim, hidden_dim, embedding_dim, num_layers, add_self,
            normalize=False, dropout=0.0):
        conv_first = GraphConv(input_dim=input_dim, output_dim=hidden_dim, add_self=add_self,
                normalize_embedding=normalize, bias=self.bias)
        conv_block = nn.ModuleList(
                [GraphConv(input_dim=hidden_dim, output_dim=hidden_dim, add_self=add_self,
                        normalize_embedding=normalize, dropout=dropout, bias=self.bias) 
                 for i in range(num_layers-2)])
        conv_last = GraphConv(input_dim=hidden_dim, output_dim=embedding_dim, add_self=add_self,
                normalize_embedding=normalize, bias=self.bias)
        return conv_first, conv_block, conv_last

    def build_pred_layers(self, pred_input_dim, pred_hidden_dims, label_dim, num_aggs=1):
        pred_input_dim = pred_input_dim * num_aggs
        if len(pred_hidden_dims) == 0:
            pred_model = nn.Linear(pred_input_dim, label_dim)
        else:
            pred_layers = []
            for pred_dim in pred_hidden_dims:
                pred_layers.append(nn.Linear(pred_input_dim, pred_dim))
                pred_layers.append(self.act)
                pred_input_dim = pred_dim
            pred_layers.append(nn.Linear(pred_dim, label_dim))
            pred_model = nn.Sequential(*pred_layers)
        return pred_model

    def construct_mask(self, max_nodes, batch_num_nodes): 
        ''' For each num_nodes in batch_num_nodes, the first num_nodes entries of the 
        corresponding column are 1's, and the rest are 0's (to be masked out).
        Dimension of mask: [batch_size x max_nodes x 1]
        '''
        # masks
        packed_masks = [torch.ones(int(num)) for num in batch_num_nodes]
        batch_size = len(batch_num_nodes)
        out_tensor = torch.zeros(batch_size, max_nodes)
        for i, mask in enumerate(packed_masks):
            out_tensor[i, :batch_num_nodes[i]] = mask
        return out_tensor.unsqueeze(2).cuda()

    def apply_bn(self, x):
        ''' Batch normalization of 3D tensor x
        '''
        bn_module = nn.BatchNorm1d(x.size()[1]).cuda()
        return bn_module(x)

    def gcn_forward(self, x, adj, conv_first, conv_block, conv_last, embedding_mask=None):

        ''' Perform forward prop with graph convolution.
        Returns:
            Embedding matrix with dimension [batch_size x num_nodes x embedding]
        '''
        x = conv_first(x, adj)
        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)
        x_all = [x]
        #out_all = []
        #out, _ = torch.max(x, dim=1)
        #out_all.append(out)
        for i in range(len(conv_block)):
            x = conv_block[i](x,adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            x_all.append(x)
        x = conv_last(x,adj)
        x_all.append(x)
        # x_tensor: [batch_size x num_nodes x embedding]
        x_tensor = torch.cat(x_all, dim=2)
        embedding_mask = embedding_mask.unsqueeze(-1)
        if embedding_mask is not None:
            x_tensor = x_tensor * embedding_mask
        return x_tensor

    def forward(self, x, adj, batch_num_nodes=None, **kwargs):
        # mask
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            self.embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            self.embedding_mask = None

        # conv
        x = self.conv_first(x, adj)
        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)
        out_all = []
        out, _ = torch.max(x, dim=1)
        out_all.append(out)
        for i in range(self.num_layers-2):
            x = self.conv_block[i](x,adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            out,_ = torch.max(x, dim=1)
            out_all.append(out)
            if self.num_aggs == 2:
                out = torch.sum(x, dim=1)
                out_all.append(out)
        x = self.conv_last(x,adj)
        #x = self.act(x)
        out, _ = torch.max(x, dim=1)
        out_all.append(out)
        if self.num_aggs == 2:
            out = torch.sum(x, dim=1)
            out_all.append(out)
        if self.concat:
            output = torch.cat(out_all, dim=1)
        else:
            output = out
        ypred = self.pred_model(output)
        #print(output.size())
        return ypred

    def loss(self, pred, label, type='softmax'):
        # softmax + CE
        if type == 'softmax':
            return F.cross_entropy(pred, label, reduction='mean')
        elif type == 'margin':
            batch_size = pred.size()[0]
            label_onehot = torch.zeros(batch_size, self.label_dim).long().cuda()
            label_onehot.scatter_(1, label.view(-1,1), 1)
            return torch.nn.MultiLabelMarginLoss()(pred, label_onehot)


class SoftPoolingGcnEncoder(GcnEncoderGraph):
    def __init__(self, max_num_nodes, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
            assign_hidden_dim, assign_ratio=0.25, assign_num_layers=-1, num_pooling=1,
            pred_hidden_dims=[50], concat=True, bn=True, dropout=0.0, assign_input_dim=-1, args=None):
        '''
        Args:
            num_layers: number of gc layers before each pooling
            num_nodes: number of nodes for each graph in batch
        '''

        super(SoftPoolingGcnEncoder, self).__init__(input_dim, hidden_dim, embedding_dim, label_dim,
                num_layers, pred_hidden_dims=pred_hidden_dims, concat=concat, args=args)
        add_self = not concat
        self.num_pooling = num_pooling

        if assign_num_layers == -1:
            assign_num_layers = num_layers
        if assign_input_dim == -1:
            assign_input_dim = input_dim

        assign_dim = int(max_num_nodes * assign_ratio)
        self.assign_conv_first, self.assign_conv_block, self.assign_conv_last = self.build_conv_layers(
                assign_input_dim, assign_hidden_dim, assign_dim, assign_num_layers, add_self,
                normalize=True)
        assign_pred_input_dim = assign_hidden_dim * (num_layers - 1) + assign_dim if concat else assign_dim
        self.assign_pred = self.build_pred_layers(assign_pred_input_dim, [], assign_dim, num_aggs=1)
        self.gcn_after_pooling = GraphConv(input_dim, hidden_dim, add_self=add_self, normalize_embedding=True)

        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def forward(self, x, adj, mask):
        embedding_tensor = self.gcn_forward(x, adj, self.conv_first, self.conv_block, self.conv_last, mask)
        assign_matrix = self.gcn_forward(x, adj, 
                self.assign_conv_first, self.assign_conv_block, self.assign_conv_last, mask)

        # [batch_size x num_nodes x next_lvl_num_nodes]
        embedding_tensor = self.transform(embedding_tensor)
        assign_matrix = nn.Softmax(dim=-1)(self.assign_pred(assign_matrix))
        mask = mask.unsqueeze(-1)
        if mask is not None:
            assign_matrix = assign_matrix * mask

        # update pooled features and adj matrix
        x_new = torch.matmul(torch.transpose(assign_matrix, 1, 2), embedding_tensor)
        adj_new = torch.transpose(assign_matrix, 1, 2) @ adj @ assign_matrix

        return assign_matrix, x_new