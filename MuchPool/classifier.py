import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_class, num_layers=2, dropout=0., indrop=0):
        super(MLPClassifier, self).__init__()

        self.num_layers = num_layers
        if self.num_layers == 2:
            self.h1_weights = nn.Linear(input_size, hidden_size)
            self.h2_weights = nn.Linear(hidden_size, num_class)
            torch.nn.init.xavier_normal_(self.h1_weights.weight.t())
            torch.nn.init.constant_(self.h1_weights.bias, 0)
            torch.nn.init.xavier_normal_(self.h2_weights.weight.t())
            torch.nn.init.constant_(self.h2_weights.bias, 0)
        elif self.num_layers == 1:
            self.h1_weights = nn.Linear(input_size, num_class)
            torch.nn.init.xavier_normal_(self.h1_weights.weight.t())
            torch.nn.init.constant_(self.h1_weights.bias, 0)
        self.dropout = dropout
        self.indrop = indrop
        if self.dropout > 0.001:
            self.dropout_layer = nn.Dropout(p = dropout)
        
    def forward(self, x, y = None):
        if self.indrop and self.dropout > 0.001:
            x = self.dropout_layer(x)
        if self.num_layers == 2:
            h1 = self.h1_weights(x)
            if self.dropout > 0.001:
                h1 = self.dropout_layer(h1)
            h1 = F.relu(h1)

            logits = self.h2_weights(h1)
        elif self.num_layers == 1:
            logits = self.h1_weights(x)
        
        softmax_logits = F.softmax(logits, dim=1)
        logits = F.log_softmax(logits, dim=1)

        if y is not None:
            loss = F.nll_loss(logits, y)

            pred = logits.data.max(1)[1]
            acc = pred.eq(y.data.view_as(pred)).cpu().sum().item() / float(y.size()[0])
            return logits, softmax_logits, loss, acc
        else:
            return logits