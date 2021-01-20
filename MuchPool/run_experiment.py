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
import torch.optim as optim
import torch.nn.functional as F
import json
import os
import os.path as osp


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(g_list, classifier, sample_idxes, epoch, optimizer, args):
    total_loss = []
    total_iters = len(sample_idxes) // args.batch_size
    pred_correct_num = 0
    total_sample_num = 0
    avg_loss = 0.0
    classifier.train()

    pbar = tqdm(range(total_iters))
    for pos in pbar:
        pbar.set_description('Train epoch %i progress' % epoch)
        selected_idx = sample_idxes[pos * args.batch_size: (pos + 1) * args.batch_size]
        batch_graph = [g_list[idx] for idx in selected_idx]

        X, adjs, masks, labels = PrepareFeatureLabel(batch_graph, args.feat_dim, args.attr_dim)
        logits = classifier(X, adjs, masks)

        optimizer.zero_grad()
        loss = F.nll_loss(logits, labels)

        loss.backward()
        optimizer.step()

        labels_pred = logits.data.max(1)[1]
        cur_correct_num = labels_pred.eq(labels.data.view_as(labels_pred)).sum().item()
        pred_correct_num += cur_correct_num
        total_sample_num += labels.size()[0]
        loss = loss.data.item()
        cur_acc = cur_correct_num / labels.size()[0]
        avg_loss += loss

        pbar.set_postfix(loss='%.5f' % loss, acc='%.5f' % cur_acc, desc='mini-batch info')

    avg_loss = avg_loss / total_sample_num
    acc = pred_correct_num / total_sample_num
    return avg_loss, acc


def evaluate(g_list, classifier, sample_idxes, epoch, optimizer, args):
    total_loss = []
    total_iters = len(sample_idxes) // args.batch_size + 1
    pred_correct_num = 0
    total_sample_num = 0
    best_acc = 0
    best_epoch = 0
    avg_loss = 0.0
    classifier.eval()
    with torch.no_grad():
        pbar = tqdm(range(total_iters))
        for pos in pbar:
            pbar.set_description('Test epoch %i progress' % epoch)
            selected_idx = sample_idxes[pos * args.batch_size: (pos + 1) * args.batch_size]
            batch_graph = [g_list[idx] for idx in selected_idx]

            X, adjs, masks, labels = PrepareFeatureLabel(batch_graph, args.feat_dim, args.attr_dim)
            logits = classifier(X, adjs, masks)
            loss = F.nll_loss(logits, labels)

            labels_pred = logits.data.max(1)[1]
            cur_correct_num = labels_pred.eq(labels.data.view_as(labels_pred)).sum().item()
            pred_correct_num += cur_correct_num
            total_sample_num += labels.size()[0]
            loss = loss.data.item()
            cur_acc = cur_correct_num / labels.size()[0]
            avg_loss += loss

            pbar.set_postfix(loss='%.5f' % loss, acc='%.5f' % cur_acc, desc='mini-batch info')

    avg_loss = avg_loss / total_sample_num
    acc = pred_correct_num / total_sample_num
    return avg_loss, acc


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


def app_run(train_graphs, test_graphs, dataset, fold_idx, fb, config):
    train_idxes = list(range(len(train_graphs)))
    test_idxes = list(range(len(test_graphs)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(config)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    log = {'fold': 0, 'epoch': 0, 'train acc': 0, 'train loss': 0, 'test acc': 0, 'test loss': 0, 'best test acc': 0, 'best epoch': 0}
    best_test_result = {'acc': 0, 'epoch': 0, 'loss': 0.0}
    max_acc = 0.0

    random.shuffle(train_idxes)
    for epoch in range(1, config.epochs+1):
        start = time.time()
        avg_train_loss, train_acc = train(train_graphs, model, train_idxes, epoch, optimizer, config)
        end = time.time()
        print('Train epoch {} result: loss: {:.5f}   acc: {:.5f}   time cost: {:.2f}s'.format(epoch, avg_train_loss, train_acc, end-start))

        start = time.time()
        avg_test_loss, test_acc = evaluate(test_graphs, model, test_idxes, epoch, optimizer, config)
        end = time.time()
        max_acc = max(max_acc, test_acc)
        print('Test epoch {} result: loss: {:.5f}   acc: {:.5f}   max: {:.5f}   time cost: {:.2f}s'.format(epoch, avg_test_loss, test_acc, max_acc, end-start))
        if test_acc > best_test_result['acc'] - 1e-7:
            best_test_result['acc'] = test_acc
            best_test_result['loss'] = avg_test_loss
            best_test_result['epoch'] = epoch

        log['fold'] = fold_idx + 1
        log['epoch'] = epoch
        log['train acc'] = '%.5f' % train_acc
        log['train loss'] = '%.5f' % avg_train_loss
        log['test acc'] = '%.5f' % test_acc
        log['test loss'] = '%.5f' % avg_test_loss
        log['best test acc'] = '%.5f' % best_test_result['acc']
        log['best epoch'] = best_test_result['epoch']

        json_data = json.dumps(log, ensure_ascii=False)
        fb.write(json_data + '\n')


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--dataset', default='ENZYMES', type=str, help='dataset')
    args = parse.parse_args()
    config_file = osp.join(osp.dirname(osp.abspath(__file__)), 'config', '%s.ini' % args.dataset)
    config = Config(config_file)
    set_seed(config.seed)

    G_data = FileLoader(args.dataset, config).load_data()
    training_process_data_file = generate_result_file_name(args.dataset, config)
    check_dir(training_process_data_file)

    train_fb = open(training_process_data_file, 'a+', encoding='utf-8')
    for fold_idx in range(config.fold):
        G_data.use_fold_data(fold_idx)
        train_graphs, test_graphs = G_data.train_graphs, G_data.test_graphs
        print('start training ------> fold', fold_idx+1)
        print('train sample number: {}   test sample number: {}'.format(len(train_graphs), len(test_graphs)))
        app_run(train_graphs, test_graphs, args.dataset, fold_idx, train_fb, config)
        print()

    rg = Result_generator(training_process_data_file, args.dataset, config.hierarchical_num, config.local_topology, config.with_feature, config.global_topology)
    rg.generate_acc_std()
    rg.generate_train_loss_curve()
    rg.generate_test_loss_curve()