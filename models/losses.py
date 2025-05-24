from abc import ABC, abstractmethod
from typing import Callable, Optional

import mlx.core as mx
import mlx.nn as nn

from equations.equations import PDEEquation


class Loss(ABC):
    """
    Impose spatio-temporal constraints, such as boundary and initial conditions.
    """

    def __init__(self, weight: float = 1.0):
        self.weight = weight
        
    @abstractmethod
    def _loss(self, model: nn.Module, x: mx.array, t: mx.array) -> mx.array:
        """Compute boundary loss for the given model."""
        pass

    def __call__(self, model: nn.Module, x: mx.array, t: mx.array) -> mx.array:
        return self.weight * self._loss(model=model, x=x, t=t)



class PhysicsLoss(Loss):
    """
    Impose spatio-temporal constraints, such as boundary and initial conditions.
    """

    def __init__(self, equation: PDEEquation, weight: float = 1.0):
        super().__init__(weight=weight)
        self.equation = equation

    def _loss(self, model: nn.Module, x: mx.array, t: mx.array) -> mx.array:
        return mx.mean(self.equation.residual(model=model, x=x, t=t)**2)
    

class DirichletLoss(Loss):
    """
    Dirichlet Loss
    Used for initial and boundary conditions.
    """
    def __init__(self, target_fn: str, weight: float = 1.0):
        super().__init__(weight=weight)
        self.target_fn = lambda x: eval(target_fn)

    def _loss(self, model: nn.Module, x: mx.array, t: mx.array) -> mx.array:
        """
        For initial conditons, t = mx.zeros((x.shape[0], 1))
        For boundary condtions, constraint must hold for all t.
        """
        u_pred = model(x, t) #(B,)
        u_target = self.target_fn(x.squeeze()) #(B,)
        return mx.mean((u_pred - u_target) ** 2)