import torch
import torch.nn as nn

# from tqdm import tqdm

from . import layers as diffeq_layers

__all__ = ["ODEnet", "ODEfunc"]

def divergence_approx(f, y, e=None):
    # FFJORD trick to approximate trace
    e_dzdx   = torch.autograd.grad(f, y, e, create_graph=True)[0]
    e_dzdx_e = e_dzdx.mul(e)

    cnt = 0
    while not e_dzdx_e.requires_grad and cnt < 10:
        e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
        e_dzdx_e = e_dzdx * e
        cnt += 1

    approx_tr_dzdx = e_dzdx_e.sum(dim=-1)

    assert approx_tr_dzdx.requires_grad, \
        "(failed to add node to graph) f=%s %s, y(rgrad)=%s, e_dzdx:%s," \
        " e:%s, e_dzdx_e:%s cnt:%s" % (
            f.size(), f.requires_grad, y.requires_grad, e_dzdx.requires_grad,
            e.requires_grad, e_dzdx_e.requires_grad, cnt)

    return approx_tr_dzdx


class ODEnet(nn.Module):
    """
    Helper class to make neural nets for use in continuous normalizing flows
    size: latent --> hidden --> latent
    """
    def __init__(self, latent_dim, hidden_dims, context_dims=0):
        super(ODEnet, self).__init__()

        base_layer   = diffeq_layers.ConcatSquashLinear
        nonlinearity = nn.Softplus

        layers = nn.ModuleList()

        in_dim = latent_dim
        for dim in hidden_dims:
            layer = nn.Sequential(
                base_layer(in_dim, dim, context_dims),
                nonlinearity(),
            )
            layers.append(layer)
            in_dim = dim

        layers.append(nn.Sequential(
            base_layer(in_dim, latent_dim, context_dims),
        ))

        self.layers = layers

    def forward(self, y, context):
        for layer in self.layers:
            y = layer((y, context))
        return y


class ODEfunc(nn.Module):
    def __init__(self, diffeq):
        super(ODEfunc, self).__init__()
        self.diffeq = diffeq
        self.divergence_fn = divergence_approx
        self.register_buffer("num_evals", torch.tensor(0.))
        self.e = None

    def before_odeint(self):
        self.e = None
        self.num_evals.fill_(0)

    def forward(self, t, states):
        y = states[0]
        t = torch.ones(y.size(0), 1).to(y) * t
        # c = states[2]
        
        # is the clone detach required ?
        # Regardless of T being able to be trained
        # .clone().detach().requires_grad_(True).type_as(y)

        # just to be sure
        t.requires_grad_(True)
        y.requires_grad_(True)
        # c.requires_grad_(True)

        self.num_evals += 1

        # Sample and fix the noise.
        if self.e is None:
            self.e = torch.randn_like(y, requires_grad=True).to(y)

        # I guess need to set this for inference ?
        with torch.set_grad_enabled(True):
            context = t
            dy = self.diffeq(y, context)
            divergence = self.divergence_fn(dy, y, self.e).view(-1, 1)

        return dy, -divergence
