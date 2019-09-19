import torch
from torch import nn
from torch.nn import functional as F

class DecoderCell(nn.Module):

    def __init__(self, in_feats, hidden_feats, num_layers=4, bias=True):
        super().__init__()

        self.lstm = nn.GRU(
            in_feats, hidden_feats, num_layers, bias
        )

        self.num_layers = num_layers
        self.hidden_feats = hidden_feats

        self.hidden = None

    def reset_hidden(self, batch_size):
        cuda = next(self.lstm.parameters()).is_cuda
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_feats)
        hidden = hidden.cuda() if cuda else hidden
        self.hidden = hidden
        return

    def set_context(self, context):
        for l in range(self.num_layers):
            assert(self.hidden[l].shape == context.shape)
            self.hidden[l] = context

    def detach(self):
        self.hidden = self.hidden.detach()
        return

    def forward(self):
        '''
        call the specific forward fn you want. For clarity
        '''
        raise NotImplementedError
    
    def forward_unit(self, x):
        s, h = self.lstm(x.unsqueeze(0), self.hidden)
        assert (s == h[-1]).all()
        self.hidden = h
        return s[0]

    def forward_seq(self, x):
        s, h = self.lstm(x, self.hidden)
        seq_len, batch_size, _ = s.shape
        self.hidden = h
        return s

