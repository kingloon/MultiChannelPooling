import torch
import time
import json
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from utils import PrepareFeatureLabel
from dataset import Graph_dataset

class Trainer(object):
    def __init__(self, conf, model, G_data):
        self.conf = conf
        self.model = model
        self.fold_idx = G_data.fold_idx
        self.init(G_data.train_graphs, G_data.test_graphs)
        if torch.cuda.is_available():
            self.model.cuda()

    def init(self, train_gs, test_gs):
        print('#train: %d, #test: %d' % (len(train_gs), len(test_gs)))
        self.train_graphs = train_gs
        self.test_graphs = test_gs
        train_dataset = Graph_dataset(train_gs, self.conf.feat_dim, self.conf.attr_dim)
        test_dataset = Graph_dataset(test_gs, self.conf.feat_dim, self.conf.attr_dim)
        self.train_data_loader = train_dataset.loader(self.conf.batch_size, True)
        self.test_data_loader = test_dataset.loader(self.conf.batch_size, False)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.conf.lr, weight_decay=self.conf.weight_decay)

    def to_cuda(self, gs):
        if torch.cuda.is_available():
            if type(gs) == list:
                return [g.cuda() for g in gs]
            return gs.cuda()
        return gs

    def run_epoch(self, epoch, data, model, optimizer, desc_str):
        pred_correct_num = 0
        total_sample_num = 0
        avg_loss = 0.0
        for batch in tqdm(data, desc=desc_str, unit='graphs'):
            xs, adjs, masks, labels = batch
            logits = model(xs, adjs, masks)
            loss = F.nll_loss(logits, labels)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            labels_pred = logits.data.max(1)[1]
            cur_correct_num = labels_pred.eq(labels.data.view_as(labels_pred)).sum().item()
            pred_correct_num += cur_correct_num
            total_sample_num += labels.size()[0]
            loss = loss.data.item()
            cur_acc = cur_correct_num / labels.size()[0]
            avg_loss += loss

        avg_loss = avg_loss / total_sample_num
        acc = pred_correct_num / total_sample_num
        return avg_loss, acc

    def train(self, acc_file, fold_idx):
        max_acc = 0.0
        for epoch in range(1, self.conf.epochs+1):
            start = time.time()
            avg_train_loss, train_acc = self.run_epoch(epoch, self.train_data_loader, self.model, self.optimizer, 'Training progress')
            end = time.time()
            print('Train epoch {} result: loss: {:.5f}   acc: {:.5f}   time cost: {:.2f}s'.format(epoch, avg_train_loss, train_acc, end-start))

            start = time.time()
            avg_test_loss, test_acc = self.run_epoch(epoch, self.test_data_loader, self.model, None, 'Testing progress')
            end = time.time()
            max_acc = max(max_acc, test_acc)
            print('\033[1;32mTest epoch {} result: loss: {:.5f}   acc: {:.5f}   max: {:.5f}   time cost: {:.2f}s\033[0m'.format(epoch, avg_test_loss, test_acc, max_acc, end-start))
            
        line_str = '%d:\t%.5f\n'
        with open(acc_file, 'a+', encoding='utf-8') as fb:
            fb.write(line_str % (fold_idx, max_acc))
        fb.close()    