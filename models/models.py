from typing import List

import mlx
import mlx.core as mx
import mlx.nn as nn

def layer_norm(a: mx.array, eps: float = 1e-7) -> mx.array:
    """
    LayerNorm without any trainable parameters 

    Args:
        a (mx.array): incoming activations
        eps (float): tolerance

    Returns:
        (mx.array): normalized activations
    """
    a_mean = mx.mean(a, axis=-1, keepdims=True)
    a_var = mx.var(a, axis=-1, keepdims=True)
    return (a - a_mean) / mx.sqrt(a_var + eps)


class MLP(nn.Module):

    def __init__(self, layer_dims: List[int]):
        super().__init__()
        self.layers = [nn.Linear(d_in, d_out) 
                       for d_in, d_out in zip(layer_dims[:-1], layer_dims[1:])]
        
    def __call__(self, x: mx.array, t: mx.array) -> mx.array:
        # x is (B, d = 1 or 2) and t is (B, 1)

        p = mx.concatenate([x,t], axis=-1)
        for layer in self.layers[:-1]:
            # p = nn.tanh(layer_norm(layer(p)))
            # p = nn.gelu(layer_norm(layer(p)))
            p = nn.tanh(layer(p))
            
        return self.layers[-1](p).squeeze() # (B,)
