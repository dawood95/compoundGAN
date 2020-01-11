import torch

from torch import nn
from torch.nn import functional as F

class Decoder(nn.Module):

    def __init__(self, latent_dim, output_dims,
                 num_layers=1, num_head=1, ff_dim=1024,
                 dropout=0.1, bias=True):

        super().__init__()

        hidden_dim = latent_dim

        self.token_project = nn.Sequential(
            nn.Linear(sum(output_dims), hidden_dim, bias=bias),
            nn.LayerNorm(hidden_dim),
            nn.SELU(True)
        )

        decoder_layer = nn.TransformerDecoderLayer(
            hidden_dim, num_head, ff_dim, dropout
        )

        self.transformer = nn.TransformerDecoder(
            decoder_layer, num_layers
        )

        self.classifiers = nn.ModuleList([
            nn.Linear(hidden_dim, feat_dim, bias)
            for feat_dim in output_dims
        ])

        sin_freq = torch.arange(0, hidden_dim, 2.0) / (hidden_dim)
        sin_freq = 1 / (1_000 ** sin_freq)
        cos_freq = torch.arange(1, hidden_dim, 2.0) / (hidden_dim)
        cos_freq = 1 / (1_000 ** cos_freq)

        x = torch.arange(0, 500) # 100 = max tokens

        sin_emb = torch.sin(torch.einsum('i,d->id', x, sin_freq))
        cos_emb = torch.cos(torch.einsum('i,d->id', x, cos_freq))
        sin_emb = sin_emb.unsqueeze(1)
        cos_emb = cos_emb.unsqueeze(1)

        embedding = torch.zeros(len(x), 1, hidden_dim)
        embedding[:, :, 0:hidden_dim:2] = sin_emb
        embedding[:, :, 1:hidden_dim:2] = cos_emb

        self.pos_emb = embedding

    def get_pos_emb(self, batch_size, seq_length, pos=None):
        if pos is None:
            embedding = self.pos_emb[:seq_length]
        else:
            embedding = self.pos_emb[pos].unsqueeze(0)
        embedding = embedding.clone().detach()
        embedding = embedding.repeat_interleave(batch_size, dim=1)
        return embedding

    def generate_square_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.float().masked_fill(mask == 1, float('-inf'))
        mask = mask.masked_fill(mask == 0, float(0.0))
        return mask

    def generate_diagonal_mask(self, sz):
        mask = torch.ones(sz, sz) * float('-inf')
        diag = torch.diagonal(mask)
        diag[:] = 0.0
        return mask

    def forward(self, z, token_x, token_y):

        batch_size = z.shape[0]
        seq_length = token_x.shape[0]

        token_x = self.token_project(token_x)
        token_x = token_x + self.get_pos_emb(batch_size, seq_length).to(z)

        diagonal_mask = self.generate_diagonal_mask(seq_length).to(z)
        square_mask   = self.generate_square_mask(seq_length).to(z)

        context = torch.cat([z.unsqueeze(0), token_x[:-1]], 0)

        token_emb = self.transformer(
            tgt = token_x,
            memory = context,
            tgt_mask = diagonal_mask,
            memory_mask = square_mask
            # tgt_key_padding_mask = (token_y[:, :, 0] == -1).T
        )

        token_preds = [c(token_emb) for c in self.classifiers]
        token_preds = [pred.view(-1, pred.shape[-1]) for pred in token_preds]
        token_tgt   = token_y.view(-1, token_y.shape[-1])

        total_loss = 0.
        for i in range(token_tgt.shape[-1]):
            loss = F.cross_entropy(token_preds[i], token_tgt[:, i],
                                   ignore_index=-1, reduction='mean')
            total_loss += loss

        return total_loss # / seq_length



