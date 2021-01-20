import random
import torch
from utils import PrepareFeatureLabel

class Graph_dataset(object):
    def __init__(self, data, feat_dim, attr_dim):
        self.data = data
        self.idx = list(range(len(data)))
        self.pos = 0
        self.feat_dim = feat_dim
        self.attr_dim = attr_dim

    def __reset__(self):
        self.pos = 0
        if self.shuffle:
            random.shuffle(self.idx)
    
    def __len__(self):
        return len(self.data) // self.batch_size + 1

    def __getitem__(self, idx):
        return self.data[idx]

    def __iter__(self):
        return self

    def __next__(self):
        if self.pos >= len(self.data):
            self.__reset__()
            raise StopIteration

        cur_idx = self.idx[self.pos: self.pos + self.batch_size]
        batch_graphs = [self.__getitem__(idx) for idx in cur_idx]
        self.pos += len(cur_idx)
        # features, adjs, masks, labels = map(list, zip(*data))
        xs, adjs, masks, labels = PrepareFeatureLabel(batch_graphs, self.feat_dim, self.attr_dim)
        return xs, adjs, masks, labels

    def loader(self, batch_size, shuffle):
        self.batch_size = batch_size
        self.shuffle = shuffle
        if shuffle:
            random.shuffle(self.idx)
        return self