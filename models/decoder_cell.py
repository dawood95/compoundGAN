import torch
from torch import nn
from torch.nn import functional as F

class DecoderCell(nn.Module):

    def __init__(self, in_feats, out_feats, hidden_feats, num_layers=4, bias=True):
        super().__init__()

        self.input_fc = nn.Linear(in_feats, hidden_feats, bias)

        self.lstm_layers = nn.ModuleList([])
        for i in range(num_layers):
            self.lstm_layers.append(nn.LSTMCell(
                hidden_feats, hidden_feats, bias=bias
            ))

        self.output_fc = nn.Linear(hidden_feats, out_feats, bias)

        self.num_layers = num_layers
        self.hidden_feats = hidden_feats

        self.hidden = None

    def reset_hidden(self, batch_size):
        cuda = next(self.input_fc.parameters()).is_cuda
        hidden = []
        for l in range(self.num_layers):
            _hidden = []
            for _ in range(2):
                _hidden.append(torch.zeros(batch_size, self.hidden_feats))
            _hidden = [h.cuda() if cuda else h for h in _hidden]
            hidden.append(_hidden)
        self.hidden = hidden
        return

    def set_context(self, context):
        for i in range(self.num_layers):
            assert(self.hidden[i][1].shape == context.shape)
            self.hidden[i][1] = context

    def detach(self, mask):
        raise NotImplementedError
        return

    def forward(self, x):
        x = F.relu(self.input_fc(x))
        for i in range(self.num_layers):
            self.hidden[i] = self.lstm_layers[i](x, self.hidden[i])
            x = self.hidden[i][0]
        x = self.output_fc(x)
        return x

