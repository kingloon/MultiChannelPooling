import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp
import matplotlib.pyplot as plt
import argparse
import json
import os.path as osp
import torch
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold


class Hyper_Graph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graphs
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a numpy array of continuous node features
        '''
        super().__init__()
        self.num_nodes = len(node_tags)
        self.node_tags = self.__rerange_tags(node_tags, list(g.nodes())) # rerange nodes index
        self.label = label
        self.g = g
        self.node_features = self.__rerange_fea(node_features, list(g.nodes())) # numpy array (node_num * feature_dim)
        self.degs = list(dict(g.degree()).values()) # type(g.degree()) is dict
        self.adj = self.__preprocess_adj(nx.adjacency_matrix(g)) # torch.FloatTensor

    def __rerange_fea(self, node_features, node_list):
        if node_features == None or node_features == []:
            return node_features
        else:
            new_node_features = []
            for i in range(node_features.shape[0]):
                new_node_features.append(node_features[node_list[i]])

            new_node_features = np.vstack(new_node_features)
            return new_node_features

    def __rerange_tags(self, node_tags, node_list):
        new_node_tags = []
        if node_tags != []:
            for i in range(len(node_tags)):
                new_node_tags.append(node_tags[node_list[i]])

        return new_node_tags

    def __sparse_to_tensor(self, adj):
        '''
            adj: sparse matrix in Coordinate format
        '''
        assert sp.isspmatrix_coo(adj), 'not coo format sparse matrix'

        values = adj.data
        indices = np.vstack((adj.row, adj.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = adj.shape
        
        return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()

    def __normalize_adj(self, sp_adj):
        adj = sp.coo_matrix(sp_adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    def __preprocess_adj(self, sp_adj):
        '''
            sp_adj: sparse matrix in compressed Sparse Row format
        '''
        adj_normalized = self.__normalize_adj(sp_adj + sp.eye(sp_adj.shape[0]))

        return self.__sparse_to_tensor(adj_normalized)

class Graph(object):
    def __init__(self, feature, adj, mask, label):
        self.feature = feature
        self.adj = adj
        self.mask = mask
        self.label = label

def get_graph_list(dataset, cmd_args):
    g_list = []
    label_dict = {}
    feat_dict = {}

    with open('./data/%s/%s.txt' % (dataset, dataset), 'r') as f:
        n_g = int(f.readline().strip()) # number of graph
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row] #node number & graph label
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if tmp > len(row):
                    node_features.append(attr)
                
                n_edges += row[1]
                for k in range(2, tmp):         # for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                node_feature = np.stack(node_features)
                node_feature_flag = True
            else:
                node_feature = None
                node_feature_flag = False
                
            #assert len(g.edges()) * 2 == n_edges (some graphs in COLLAB have self-loops, ignored here)
            assert len(g) == n
            g_list.append(Hyper_Graph(g, l, node_tags, node_features))
    
    for g in g_list:
        g.label = label_dict[g.label]
    cmd_args.num_class = len(label_dict)
    cmd_args.feat_dim = len(feat_dict)  # maximum node label (tag)
    if node_feature_flag == True:
        cmd_args.attr_dim = node_features.shape[1]  # dimension of node features (attributes)
    else:
        cmd_args.attr_dim = 0
    cmd_args.input_dim = cmd_args.feat_dim + cmd_args.attr_dim

    return g_list

def generate_final_result(result_file):
    fb = open(result_file, 'r')
    line = fb.readline()
    folds = []
    epochs = []
    train_accs = []
    train_losses = []
    test_accs = []
    test_losses = []
    best_test_accs = []
    best_epochs = []
    while line:
        data_dict = json.loads(line)
        folds.append(int(data_dict['fold']))
        epochs.append(int(data_dict['epoch']))
        train_accs.append(float(data_dict['train acc']))
        train_losses.append(float(data_dict['train loss']))
        test_accs.append(float(data_dict['test acc']))
        test_losses.append(float(data_dict['test loss']))
        best_test_accs.append(float(data_dict['best test acc']))
        best_epochs.append(int(data_dict['best epoch']))
        line = fb.readline()

    avg_train_accs = np.array(train_accs).reshape(10, -1)
    avg_train_losses = np.array(train_losses).reshape(10, -1)
    avg_test_accs = np.array(test_accs).reshape(10, -1)
    avg_test_losses = np.array(test_losses).reshape(10, -1)

    x = range(avg_train_accs.shape[-1])
    for i in range(10):
        plt.figure(i+1)
        plt.style.use('ggplot')
        plt.subplot(2, 2, 1)
        plt.ylabel('acc')
        plt.plot(x, avg_train_accs[i])
        plt.subplot(2, 2, 2)
        plt.plot(x, avg_train_losses[i])
        plt.ylabel('loss')
        plt.subplot(2, 2, 3)
        plt.ylabel('test_acc')
        plt.plot(x, avg_test_accs[i])
        plt.subplot(2, 2, 4)
        plt.plot(x, avg_test_losses[i])
        plt.ylabel('test_loss')
        plt.show()
    

def PrepareFeatureLabel(batch_graph, feat_dim, attr_dim):
    batch_size = len(batch_graph)
    labels = torch.LongTensor(batch_size)
    max_node_num = 0

    for i in range(batch_size):
        labels[i] = batch_graph[i].label
        max_node_num = max(max_node_num, batch_graph[i].num_nodes)

    masks = torch.zeros(batch_size, max_node_num)
    adjs = torch.zeros(batch_size, max_node_num, max_node_num)

    if batch_graph[0].node_tags is not None:
        node_tag_flag = True
        batch_node_tag = torch.zeros(batch_size, max_node_num, feat_dim)
    else:
        node_tag_flag = False

    if batch_graph[0].node_features is not None and batch_graph[0].node_features != []:
        node_feat_flag = True
        batch_node_feat = torch.zeros(batch_size, max_node_num, attr_dim)
    else:
        node_feat_flag = False

    for i in range(batch_size):
        cur_node_num = batch_graph[i].num_nodes

        #   node tag
        if node_tag_flag == True:
            tmp_tag_idx = torch.LongTensor(batch_graph[i].node_tags).view(-1, 1)
            tmp_node_tag = torch.zeros(cur_node_num, feat_dim)
            tmp_node_tag.scatter_(1, tmp_tag_idx, 1)
            batch_node_tag[i, 0:cur_node_num] = tmp_node_tag

        #   node attribute feature
        if node_feat_flag == True:
            tmp_node_feat = torch.from_numpy(batch_graph[i].node_features).type('torch.FloatTensor')
            batch_node_feat[i, 0:cur_node_num] = tmp_node_fea

        # adjs
        adjs[i, 0:cur_node_num, 0:cur_node_num] = batch_graph[i].adj

        # masks
        masks[i, 0:cur_node_num] = 1

    # combine the two kinds of node feature
    if node_feat_flag == True:
        node_feat = batch_node_feat.clone()

    if node_feat_flag and node_tag_flag:
        # concatenate one-hot embedding of node tags (node labels) with continuous node features
        node_feat = torch.cat([batch_node_tag.type_as(node_feat), node_feat], 1)
    elif node_feat_flag == False and node_tag_flag == True:
        node_feat = batch_node_tag
    elif node_feat_flag == True and node_tag_flag == False:
        pass
    else:
        node_feat = torch.ones(batch_size, max_node_num, 1)     # use all-one vector as node features

    # copy data to gpu
    node_feat = node_feat.cuda()
    adjs = adjs.cuda()
    masks = masks.cuda()
    labels = labels.cuda()
    
    return node_feat, adjs, masks, labels

class Result_generator(object):
    def __init__(self, log_file, dataset, layer, local_topology=True, with_feature=True, global_topology=True):
        self.log_file = log_file
        self.read_log_data()
        self.result_file_path = osp.join(osp.dirname(osp.abspath(__file__)), 'result', dataset)
        if not local_topology:
            self.acc_result_file_name = 'Final_result_on_%s_with_%d_layer_without_local_topology.txt' % (dataset, layer)
            self.loss_curve_file_name = 'Loss_curve_on_%s_with_%d_layer_without_local_topology.pdf' % (dataset, layer)
            self.train_loss_curve_file_name = 'Train_loss_curve_on_%s_with_%d_layer_without_local_topology.pdf' % (dataset, layer)
            self.acc_curve_file_name = 'Acc_curve_on_%s_with_%d_layer_without_local_topology.pdf' % (dataset, layer)
        elif not with_feature:
            self.acc_result_file_name = 'Final_result_on_%s_with_%d_layer_without_feature.txt' % (dataset, layer)
            self.loss_curve_file_name = 'Loss_curve_on_%s_with_%d_layer_without_feature.pdf' % (dataset, layer)
            self.train_loss_curve_file_name = 'Train_loss_curve_on_%s_with_%d_layer_without_feature.pdf' % (dataset, layer)
            self.acc_curve_file_name = 'Acc_curve_on_%s_with_%d_layer_without_feature.pdf' % (dataset, layer)
        elif not global_topology:
            self.acc_result_file_name = 'Final_result_on_%s_with_%d_layer_without_global_topology.txt' % (dataset, layer)
            self.loss_curve_file_name = 'Loss_curve_on_%s_with_%d_layer_without_global_topology.pdf' % (dataset, layer)
            self.train_loss_curve_file_name = 'Train_loss_curve_on_%s_with_%d_layer_without_global_topology.pdf' % (dataset, layer)
            self.acc_curve_file_name = 'Acc_curve_on_%s_with_%d_layer_without_global_topology.pdf' % (dataset, layer)
        else:
            self.acc_result_file_name = 'Final_result_on_%s_with_%d_layer.txt' % (dataset, layer)
            self.loss_curve_file_name = 'Loss_curve_on_%s_with_%d_layer.pdf' % (dataset, layer)
            self.train_loss_curve_file_name = 'Train_loss_curve_on_%s_with_%d_layer.pdf' % (dataset, layer)
            self.acc_curve_file_name = 'Acc_curve_on_%s_with_%d_layer.pdf' % (dataset, layer)


    def read_log_data(self):
        fb = open(self.log_file, 'r')
        line = fb.readline()
        folds = []
        epochs = []
        train_accs = []
        train_losses = []
        test_accs = []
        test_losses = []
        best_test_accs = []
        best_epochs = []
        while line:
            data_dict = json.loads(line)
            folds.append(int(data_dict['fold']))
            epochs.append(int(data_dict['epoch']))
            train_accs.append(float(data_dict['train acc']))
            train_losses.append(float(data_dict['train loss']))
            test_accs.append(float(data_dict['test acc']))
            test_losses.append(float(data_dict['test loss']))
            best_test_accs.append(float(data_dict['best test acc']))
            best_epochs.append(int(data_dict['best epoch']))
            line = fb.readline()
        fb.close()

        self.train_accs = np.array(train_accs).reshape(10, -1) * 100
        self.train_losses = np.array(train_losses).reshape(10, -1)
        self.test_accs = np.array(test_accs).reshape(10, -1) * 100
        self.test_losses = np.array(test_losses).reshape(10, -1)
        self.epoch_num = self.train_accs.shape[-1]

        best_10_results = []
        for i in range(10):
            best_10_results.append(best_test_accs[self.epoch_num * i + (self.epoch_num - 1)])
        self.best_10_results = np.array(best_10_results) * 100

    def generate_acc_std(self):
        # avg_test_accs = sum(self.test_accs/10, 1)
        # max_avg_test_acc = max(avg_test_accs)
        mean_best_result = np.mean(self.best_10_results)
        std_best_result = np.std(self.best_10_results)
        result_str = 'Mean best accuracy: {:.2f}±{:.2f}'.format(mean_best_result, std_best_result)
        result_file = osp.join(self.result_file_path, self.acc_result_file_name)
        fb = open(result_file, 'w', encoding='utf-8')
        fb.write(result_str)
        fb.close()

    def generate_test_loss_curve(self):
        curve_fig_file = osp.join(self.result_file_path, self.loss_curve_file_name)
        mean_acc = np.mean(self.test_accs, 0)
        mean_loss = np.mean(self.test_losses, 0)
        x = range(len(mean_loss))
        plt.figure(1)
        plt.style.use('ggplot')
        plt.subplot(2, 1, 1)
        plt.plot(x, mean_acc)
        plt.ylabel('Test accuracy')
        plt.title('Test accuracy vs. epoches')
        plt.subplot(2, 1, 2)
        plt.plot(x, mean_loss)
        plt.ylabel('Test loss')
        plt.xlabel('Test loss vs. epochs')
        plt.savefig(curve_fig_file)

    def generate_train_loss_curve(self):
        curve_fig_file = osp.join(self.result_file_path, self.train_loss_curve_file_name)
        mean_acc = np.mean(self.train_accs, 0)
        mean_loss = np.mean(self.train_losses, 0)
        x = range(len(mean_loss))
        plt.figure(2)
        plt.style.use('ggplot')
        plt.subplot(2, 1, 1)
        plt.plot(x, mean_acc)
        plt.ylabel('Train accuracy')
        plt.title('Train accuracy vs. epoches')
        plt.subplot(2, 1, 2)
        plt.plot(x, mean_loss)
        plt.ylabel('Train loss')
        plt.xlabel('Train loss vs. epochs')
        plt.savefig(curve_fig_file)


class Graph_data(object):
    def __init__(self, graphs, fold, seed):
        self.graphs = graphs
        # self.labels = labels
        self.seed = seed
        self.fold = fold
        self.sep_data()

    def sep_data(self):
        skf = StratifiedKFold(n_splits=self.fold, shuffle=True, random_state=self.seed)
        labels = [g.label for g in self.graphs]
        self.idx_list = list(skf.split(np.zeros(len(self.graphs)), labels))

    def use_fold_data(self, fold_idx):
        self.fold_idx = fold_idx
        train_idx, test_idx = self.idx_list[fold_idx]
        self.train_graphs = [self.graphs[i] for i in train_idx]
        self.test_graphs = [self.graphs[i] for i in test_idx]
        

class FileLoader(object):
    def __init__(self, dataset, config):
        self.dataset = dataset
        self.conf = config

    def get_graph_list(self):
        g_list = []
        label_dict = {}
        feat_dict = {}
        max_node_num = 0

        with open('./data/%s/%s.txt' % (self.dataset, self.dataset), 'r') as f:
            n_g = int(f.readline().strip()) # number of graph
            for i in tqdm(range(n_g), desc='Create graph', unit='graphs'):
                row = f.readline().strip().split()
                n, l = [int(w) for w in row] #node number & graph label
                if not l in label_dict:
                    mapped = len(label_dict)
                    label_dict[l] = mapped
                g = nx.Graph()
                node_tags = []
                node_features = []
                n_edges = 0
                max_node_num = max(max_node_num, n)
                for j in range(n):
                    g.add_node(j)
                    row = f.readline().strip().split()
                    tmp = int(row[1]) + 2
                    if tmp == len(row):
                        # no node attributes
                        row = [int(w) for w in row]
                        attr = None
                    else:
                        row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                    if not row[0] in feat_dict:
                        mapped = len(feat_dict)
                        feat_dict[row[0]] = mapped
                    node_tags.append(feat_dict[row[0]])

                    if tmp > len(row):
                        node_features.append(attr)
                    
                    n_edges += row[1]
                    for k in range(2, tmp):         # for k in range(2, len(row)):
                        g.add_edge(j, row[k])

                if node_features != []:
                    node_feature = np.stack(node_features)
                    node_feature_flag = True
                else:
                    node_feature = None
                    node_feature_flag = False
                    
                #assert len(g.edges()) * 2 == n_edges (some graphs in COLLAB have self-loops, ignored here)
                assert len(g) == n
                g_list.append(Hyper_Graph(g, l, node_tags, node_features))
        
        for g in g_list:
            g.label = label_dict[g.label]
        self.conf.num_class = len(label_dict)
        self.conf.feat_dim = len(feat_dict)  # maximum node label (tag)
        if node_feature_flag == True:
            self.conf.attr_dim = node_features.shape[1]  # dimension of node features (attributes)
        else:
            self.conf.attr_dim = 0
        self.conf.input_dim = self.conf.feat_dim + self.conf.attr_dim
        self.max_node_num = max_node_num

        return g_list

    def process_graph(self, graph):
        mask = torch.zeros(self.max_node_num)
        adj = torch.zeros(self.max_node_num, self.max_node_num)
        
        if graph.node_tags is not None:
            node_tag_flag = True
            node_tag = torch.zeros(self.max_node_num, self.conf.feat_dim)
        else:
            node_tag_flag = False

        if graph.node_features is not None and graph.node_features != []:
            node_feat_flag = True
            node_feat = torch.zeros(self.max_node_num, self.conf.attr_dim)
        else:
            node_feat_flag = False
        
        cur_node_num = graph.num_nodes

        # node tag
        if node_tag_flag == True:
            tmp_tag_idx = torch.LongTensor(graph.node_tags).view(-1, 1)
            tmp_node_tag = torch.zeros(cur_node_num, self.conf.feat_dim)
            tmp_node_tag.scatter_(1, tmp_tag_idx, 1)
            node_tag[:cur_node_num] = tmp_node_tag

        # node attribute feature
        if node_feat_flag == True:
            tmp_node_feat = torch.from_numpy(graph.node_features).type('torch.FloatTenosr')
            node_feat[:cur_node_num] = tmp_node_feat
        
        # adj
        adj[:cur_node_num, :cur_node_num] = graph.adj

        mask[:cur_node_num] = 1

        # combine the two kinds of node feature
        if node_feat_flag == True:
            node_feat = node_feat.clone()
        
        if node_feat_flag and node_tag_flag:
            # concatenate one-hot embedding of node tags (node labels) with continuous node features
            node_feat = torch.cat([node_tag.type_as(node_feat), node_feat], 1)
        elif node_feat_flag == False and node_tag_flag == True:
            node_feat = node_tag
        elif node_feat_flag == True and node_tag_flag == False:
            pass
        else:
            node_feat = torch.ones(self.max_node_num, 1)     # use all-one vector as node features

        return Graph(node_feat, adj, mask, graph.label)

    def load_data(self):
        print('Loading data ...')
        graph_list = self.get_graph_list()

        print('# ================== Dataset %s Information ==================' % self.dataset)
        print('# total classes: %d' % self.conf.num_class)
        print('# maximum node tag: %d' % self.conf.feat_dim)

        return Graph_data(graph_list, self.conf.fold, self.conf.seed)