import torch.nn as nn
import torch.nn.functional as F
from rlkit.torch import pytorch_util as ptu


class MLPBlock(nn.Module):
    def __init__(
        self,
        sizes,
        output_activation,
        hidden_activation=F.elu,
        hidden_init=ptu.fanin_init,
        b_init_value=0.1,
    ):
        super().__init__()
        self.output_activation = output_activation
        self.hidden_activation = hidden_activation
        self.hidden_init = hidden_init
        self.b_init_value = b_init_value
        self.sizes = sizes

        self.fcs = []
        in_size = sizes[0]
        for i, next_size in enumerate(sizes[1:]):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            self.hidden_init(fc.weight, scale=1.0)
            fc.bias.data.fill_(self.b_init_value)
            fc_name = f"fc{i}"
            self.__setattr__(fc_name, fc)
            self.fcs.append(fc)

    def forward(self, x):
        for fc in self.fcs[:-1]:
            res = x
            x = fc(x)
            if x.size() == res.size():
                x += res
            x = self.hidden_activation(x)
        x = self.fcs[-1](x)
        x = self.output_activation(x)

        return x
