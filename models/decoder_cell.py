import torch
from torch import nn
from torch.nn import functional as F

from .lstm import script_lnlstm, LSTMState

class DecoderCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers=4, bias=True):
        super().__init__()

        '''
        self.lstm = script_lnlstm(input_dim, hidden_dim, num_layers, bias)
        '''

        self.lstm = nn.GRU(
            input_dim, hidden_dim, num_layers, bias, dropout=0.20
        )

        '''
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, bias, dropout=0.20)
        '''

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.state = None

    def reset_state(self, batch_size):
        device = next(self.lstm.parameters()).device

        state = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        state = state.to(device)

        '''
        state  = [
            (torch.zeros(batch_size, self.hidden_dim).to(device),
             torch.zeros(batch_size, self.hidden_dim).to(device))
            for i in range(self.num_layers)]

        state  = (
            torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device),
            torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        )
        '''
        self.state = state
        return

    def set_context(self, context):
        self.state = context
        return

        new_state = []
        for i in range(self.num_layers):
            # self.state[1][i] = context[i]
            self.state[i][1][:] = context[i][:]
            # new_state.append((self.state[i][0], context[i]))
        # self.state = new_state

    def set_hidden(self, hidden):
        self.state = hidden
        return

        new_state = []
        for i in range(self.num_layers):
            # self.state[0][i] = hidden[i]
            self.state[i][0][:] = hidden[i][:]
            # new_state.append((hidden[i], self.state[i][1]))
        # self.state = new_state

    def add_hidden(self, hidden):
        self.state = self.state + hidden
        return
        new_state = []
        for i in range(self.num_layers):
            self.state[i][0][:] += hidden[i][:]
            # self.state[0][i] += hidden[i]

    def get_hidden(self):
        return self.state
        # return self.state[0]
        return [s[0] for s in self.state]

    def detach(self):
        if self.state is None: return
        # self.state[0].detach_()
        # self.state[1].detach_()
        # return
        self.state.detach_()
        for s in self.state:
            s[0].detach_()
            s[1].detach_()
        return

    def forward(self):
        '''
        call the specific forward fn you want. For clarity
        '''
        raise NotImplementedError

    def forward_unit(self, x):
        y, new_state = self.lstm(x.unsqueeze(0), self.state)
        self.state = new_state
        return new_state.sum(0)
        # return sum([h for (h, c) in new_state])

    def forward_seq(self, x):
        y, new_state = self.lstm(x, self.state)
        self.state = new_state
        return y
