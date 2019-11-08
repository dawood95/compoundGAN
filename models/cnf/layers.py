import torch
import torch.nn as nn

# Theoretically, these layers should be 1-lipschitz compliant ?
# Other than using smooth activations, quiet unsure how this is enforced

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class Lambda(nn.Module):
    def __init__(self, f):
        super(Lambda, self).__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)

class ConcatSquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_context):
        super(ConcatSquashLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1 + dim_context, dim_out, bias=False)
        self._hyper_gate = nn.Linear(1 + dim_context, dim_out)

    def forward(self, x):
        x, context = x
        # Context here is <time, context_vector>
        gate = torch.sigmoid(self._hyper_gate(context))
        bias = self._hyper_bias(context)
        ret  = self._layer(x) * gate + bias
        return ret
