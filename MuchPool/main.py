import torch
import time
import random
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from utils import Hyper_Graph
from utils import get_graph_list
from utils import PrepareFeatureLabel
from utils import FileLoader
from utils import Result_generator
from net import Net
from config import Config
from trainer import Trainer
import torch.optim as optim
import torch.nn.functional as F
import json
import os
import os.path as osp

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def generate_result_file_name(dataset, config):
    if config.local_topology and config.with_feature and config.global_topology:
        result_file_name = 'training_log_of_%d_epochs_%d_layer.json' % (config.epochs, config.hierarchical_num)
    elif not config.local_topology:
        result_file_name = 'training_log_of_%d_epochs_%d_layer_without_local_topology.json' % (config.epochs, config.hierarchical_num)
    elif not config.with_feature:
        result_file_name = 'training_log_of_%d_epochs_%d_layer_without_feature.json' % (config.epochs, config.hierarchical_num)
    elif not config.global_topology:
        result_file_name = 'training_log_of_%d_epochs_%d_layer_without_global_topology.json' % (config.epochs, config.hierarchical_num)
    full_name = osp.join(osp.dirname(osp.abspath(__file__)), 'result', dataset, result_file_name)
    return full_name

def check_dir(file_name=None):
    dir_name = osp.dirname(file_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def calculate_final_result(dataset, acc_file, convolution):
    fb = open(acc_file, 'r')
    line = fb.readline()
    results = []
    while line:
        line = line.strip()
        tmp_result = float(line.split('\t')[1])
        results.append(tmp_result)
        line = fb.readline()
    results = np.array(results) * 100
    avg_acc = np.mean(results)
    std = np.std(results)
    line = 'avg_acc±std: {:.2f}±{:.2f}'.format(avg_acc, std)
    final_result_file = osp.join(osp.dirname(osp.abspath(
        __file__)), 'result', convolution, dataset + '_final_result.txt')
    result_fb = open(final_result_file, 'w', encoding='utf-8')
    result_fb.write(line)
    fb.close()
    result_fb.close()

def app_run(config, G_data, fold_idx, acc_file, convolution_method):
    G_data.use_fold_data(fold_idx)
    model = Net(config, convolution_method)
    trainer = Trainer(config, model, G_data)
    trainer.train(acc_file, fold_idx)

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--dataset', default='NCI109', type=str, help='dataset')
    parse.add_argument('--convolution', default='GCN', type=str, help='GCN, GAT or GraphSage')
    args = parse.parse_args()
    config_file = osp.join(osp.dirname(osp.abspath(__file__)), 'config', '%s.ini' % args.dataset)
    config = Config(config_file)
    set_seed(config.seed)

    G_data = FileLoader(args.dataset, config).load_data()
    acc_file = osp.join(osp.dirname(osp.abspath(__file__)), 'result', args.convolution, args.dataset + '_result.txt')
    check_dir(acc_file)
    for fold_idx in range(config.fold):
        print('start training ------> fold', fold_idx+1)
        start = time.time()
        app_run(config, G_data, fold_idx, acc_file, args.convolution)
        print('Total time cost in this fold: {:.2f}s'.format(time.time() - start))
        print()

    calculate_final_result(args.dataset, acc_file, args.convolution)