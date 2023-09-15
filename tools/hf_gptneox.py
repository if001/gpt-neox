from transformers.models.gpt_neox import GPTNeoXPreTrainedModel, GPTNeoXModel, GPTNeoXLayer
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXMLP
from transformers.activations import ClassInstantier, ACT2CLS
from torch import Tensor, nn

from typing import Callable, Optional
import torch.nn.functional as F


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)

ACT2CLS['swiglu'] = SwiGLUFFN
ACT2FN = ClassInstantier(ACT2CLS)

class GPTNeoX2MLP(GPTNeoXMLP):
    def __init__(self, config):
        super().__init__()
        self.act = ACT2FN[config.hidden_act]

class GPTNeoX2Layer(GPTNeoXLayer):
    def __init__(self, config):
        super().__init__()
        self.mlp = GPTNeoX2MLP(config)

class GPTNeoX2Model(GPTNeoXModel):
    def __init__(self, config):
        _config = config.deepcopy()
        _config.hidden_act = "gelu"
        super().__init__(_config)
        self.layers = nn.ModuleList([GPTNeoX2Layer(config) for _ in range(config.num_hidden_layers)])

class GPTNeoX2ForCausalLM(GPTNeoXPreTrainedModel):
    _tied_weights_keys = ["embed_out.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.gpt_neox = GPTNeoX2Model(config)