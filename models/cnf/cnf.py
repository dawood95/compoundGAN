import torch
import torch.nn as nn

from torchdiffeq import odeint_adjoint
from torchdiffeq import odeint as odeint_normal

class CNF(nn.Module):
    def __init__(self, odefunc, T=1.0, train_T=False,
                 solver='dopri5', atol=1e-5, rtol=1e-5, use_adjoint=True):
        super(CNF, self).__init__()

        self.train_T = train_T
        self.T = float(T)
        if train_T:
            self.register_parameter(
                "sqrt_end_time",
                nn.Parameter(torch.sqrt(torch.tensor(float(T))))
            )

        self.odefunc = odefunc
        self.odeint  = odeint_adjoint if use_adjoint else odeint_normal
        self.solver  = solver
        self.atol    = atol
        self.rtol    = rtol


    def forward(self, x, context=None, integration_times=None, reverse=False):

        if integration_times is None:
            if self.train_T:
                integration_times = torch.stack([
                    torch.tensor(0.0).to(x),
                    self.sqrt_end_time * self.sqrt_end_time
                ])
            else:
                integration_times = torch.tensor([0., self.T],
                                                 requires_grad=False)

        if reverse:
            integration_times = _flip(integration_times, 0)

        integration_times = integration_times.to(x)

        if context is None:
            context = torch.zeros(0).to(x)

        _logpx = torch.zeros((x.shape[0], 1)).to(x)

        # Refresh the odefunc statistics.
        self.odefunc.before_odeint()

        state_t = self.odeint(
            self.odefunc,
            (x, _logpx),
            integration_times,
            atol=self.atol,
            rtol=self.rtol,
            method=self.solver,
        )

        state_t = (s[-1] for s in state_t)
        z_t, logpz_t = state_t

        return z_t, logpz_t

    def num_evals(self):
        return self.odefunc.num_evals.item()


def _flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]
