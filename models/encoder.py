import torch

from torch import nn
from torch.nn import functional as F
from torch.nn.utils import rnn

class Encoder(nn.Module):

    def __init__(self, input_dim,
                 hidden_dim=128, latent_dim=256,
                 num_layers=1, num_head=1, ff_dim=1024,
                 dropout=0.1, bias=True):

        super().__init__()

        self.inp_project = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=bias),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            hidden_dim, num_head, ff_dim, dropout
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers
        )

        self.pool = nn.GRU(hidden_dim, latent_dim, 2, bias)

        sin_freq = torch.arange(0, hidden_dim, 2.0) / (hidden_dim)
        sin_freq = 1 / (10_000 ** sin_freq)
        cos_freq = torch.arange(1, hidden_dim, 2.0) / (hidden_dim)
        cos_freq = 1 / (10_000 ** cos_freq)

        x = torch.arange(0, 500) # 100 = max 01, 32, 256tokens

        sin_emb = torch.sin(torch.einsum('i,d->id', x, sin_freq))
        cos_emb = torch.cos(torch.einsum('i,d->id', x, cos_freq))
        sin_emb = sin_emb.unsqueeze(1)
        cos_emb = cos_emb.unsqueeze(1)

        embedding = torch.zeros(len(x), 1, hidden_dim)
        embedding[:, :, 0:hidden_dim:2] = sin_emb
        embedding[:, :, 1:hidden_dim:2] = cos_emb

        self.pos_emb = embedding

        self.mu_fc     = nn.Linear(latent_dim, latent_dim, bias)
        self.logvar_fc = nn.Linear(latent_dim, latent_dim, bias)


    def get_pos_emb(self, batch_size, seq_length, pos=None):

        if pos is None:
            embedding = self.pos_emb[:seq_length]
        else:
            embedding = self.pos_emb[pos].unsqueeze(0)

        embedding = embedding.clone().detach()
        embedding = embedding.repeat_interleave(batch_size, dim=1)

        return embedding

    def forward(self, emb, mask=None):

        if mask is None:
            mask = torch.zeros((emb.shape[:-1])).to(emb).bool()
            mask[:] = True
            
        seq_length, batch_size = mask.shape

        # project to match dimensions
        # and use self-attention on embeddings
        token_emb = self.inp_project(emb)
        pos_emb   = self.get_pos_emb(batch_size, seq_length).to(emb)
        token_emb = token_emb + pos_emb
        enc_emb   = self.transformer(token_emb, None, ~mask.T)
        
        # Pack embeddings to run through GRU-pooling
        packed_emb = rnn.pack_padded_sequence(enc_emb, mask.sum(0),
                                              enforce_sorted=False)
        pooled_emb, _ = self.pool(packed_emb)
        padded_emb, lengths = rnn.pad_packed_sequence(pooled_emb)

        # Gather last timestep of RNN pooling, ignoring the padding value
        gather_idx = (lengths - 1).to(padded_emb).long().view(-1, 1)
        gather_idx = gather_idx.expand(len(lengths), padded_emb.shape[-1])
        gather_idx = gather_idx.unsqueeze(0)
        pooled_emb = padded_emb.gather(0, gather_idx).squeeze(0)

        # Calculate mu and logvar from per batch embedding
        mu_x     = self.mu_fc(pooled_emb)
        logvar_x = self.logvar_fc(pooled_emb)

        return mu_x, logvar_x
