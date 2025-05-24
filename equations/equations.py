from abc import ABC, abstractmethod

import mlx.core as mx
import mlx.nn as nn

class PDEEquation(ABC):
    @abstractmethod
    def residual(self, model: nn.Module, x: mx.array, t: mx.array) -> mx.array:
        """Return the PDE residual at input x using the current model."""
        pass


class Burgers1DEquation(PDEEquation):

    def __init__(self, nu: float):
        self.nu = nu

    def residual(self, model: nn.Module, x: mx.array, t: mx.array) -> mx.array:
        # x is (B, 1)
        # t is (B, 1)
        # to conform to grad conventions and array broadcasting, we will have to
        # squeeze arrays.

        u, (u_x, u_t) = mx.vmap(mx.value_and_grad(model, argnums=[0, 1]), in_axes=0)(x, t)

        
        dudx = mx.grad(model, argnums=0)
        _dudx = lambda a, b: dudx(a, b).squeeze()
        du2dx2 = mx.grad(_dudx, argnums=0)
        u_xx = mx.vmap(du2dx2, in_axes=0)(x, t)

        u, u_x, u_t, u_xx = map(mx.squeeze, (u, u_x, u_t, u_xx))

        return u_t + u * u_x - self.nu * u_xx


class Burgers2DEquation(PDEEquation):

    def __init__(self):
        raise NotImplementedError

    def residual(self, model: nn.Module, x: mx.array, t: mx.array) -> mx.array:
        raise NotImplementedError

