import torch
from torch import nn
from torch.nn import functional as F

class DecoderCell(nn.Module):

    def __init__(self, in_feats, out_feats, hidden_feats, num_layers=4, bias=True):
        super().__init__()

        self.lstm = nn.GRU(
            in_feats, hidden_feats, num_layers, bias
        )

        self.output_fc = nn.Linear(hidden_feats, out_feats, bias)

        self.num_layers = num_layers
        self.hidden_feats = hidden_feats

        self.hidden = None

    def reset_hidden(self, batch_size):
        cuda = next(self.output_fc.parameters()).is_cuda
        hidden = []
        for _ in range(1):
            hidden.append(torch.zeros(self.num_layers, batch_size, self.hidden_feats))
        hidden = [h.cuda() if cuda else h for h in hidden]
        self.hidden = hidden
        return

    def set_context(self, context):
        for l in range(self.num_layers):
            assert(self.hidden[-1][l].shape == context.shape)
            self.hidden[-1][l] = context

    def detach(self):
        self.hidden = [h.detach() for h in self.hidden]
        return

    def forward(self, x):
        _, h = self.lstm(x.unsqueeze(0), self.hidden[0])
        x = F.relu(self.output_fc(h[-1]))
        self.hidden[0] = h
        return x

