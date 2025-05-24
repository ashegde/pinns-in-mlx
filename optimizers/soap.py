"""
This module contains functionality for the SOAP optimization algorithm.
The motivation for this particular optimizer for PINNs is described in,

Wang, S., Bhartari, A. K., Li, B., & Perdikaris, P. (2025). 
Gradient Alignment in Physics-informed Neural Networks: A Second-Order Optimization Perspective. 
arXiv preprint arXiv:2502.00604.
"""

from typing import Union, Callable, List

import mlx.core as mx
from mlx.optimizers import AdamW


class SOAP(AdamW):
    r"""ShampoO with Adam in the Preconditioner's eigenbasis (SOAP) [1]

    Our implementation is based on [2], which identified the algorithm
    as well-suited for resolving gradient conflicts between individual terms
    in the the PINN Loss. Note that in our implementation, we follow [1] and use
    AdamW instead of Adam [2].

    [1] Vyas, N., et al. (2025) SOAP: Improving and Stabilizing Shampoo using Adam.
        arXiv:2409.11321. 

    [2] Wang, S., et al. (2025) Gradient Alignment in Physics-informed Neural Networks: 
        A Second-Order Optimization Perspective. arXiv:2502.00604. 

    .. math::
    
    For each layer's weight matrix, say W, and gradient G_t \in \mathbb{R}^{m \times n}
    at iteration t:

        L_t &= \beta_2 L_{t - 1} + (1 - \beta_2) G_t G_t.T
        R_t &= \beta_2 R_{t - 1} + (1 - \beta_2) G_t.T G_t

        Eigen-decomp: L_t = Q_L \Lambda_L Q_L.T
                      R_t = Q_R \Lambda_R Q_R.T
        
        Projection: \tilde{G}_t = Q_L.T G_t Q_R

        Adam: \tilde{W}_t = \tilde{W}_{t - 1} - \eta Adam(\tilde{G}_t)

        Return mapping: W_t = Q_L \tilde{W}_t Q_R.T

    Args:
        learning_rate (float or callable): The learning rate :math:`\alpha`.
        betas (Tuple[float, float], optional): The coefficients
          :math:`(\beta_1, \beta_2)` used for computing running averages of the
          gradient and its square. Default: ``(0.9, 0.999)``
        eps (float, optional): The term :math:`\epsilon` added to the
          denominator to improve numerical stability. Default: ``1e-8``
        weight_decay (float, optional): The weight decay :math:`\lambda`.
          Default: ``0``.
        bias_correction (bool, optional): If set to ``True``, bias correction
          is applied. Default: ``False``
        eig_frequency (int, optional): frequency of eigendecomposition updates.
          Default: ``1``.

    """

    def __init__(self,
                 learning_rate: Union[float, Callable[[mx.array], mx.array]],
                 betas: List[float] = [0.9, 0.999],
                 eps: float = 1e-8,
                 weight_decay: float = 0.01,
                 bias_correction: bool = False,
                 beta_soap: float = 0.99,
                 eig_update_freq: int = 1):
        super().__init__(learning_rate=learning_rate,
                         betas=betas,
                         eps=eps,
                         weight_decay=weight_decay,
                         bias_correction=bias_correction)
        self.beta_soap = beta_soap
        self.eig_update_freq = eig_update_freq

    def init_single(self, parameter: mx.array, state: dict) -> None:
        """Initialize optimizer state"""
        super().init_single(
            parameter=parameter,
            state=state,
        )
        if parameter.ndim == 2:
            state["L"] = mx.zeros(
                (parameter.shape[0], parameter.shape[0]),
                dtype=parameter.dtype,
            )
            state["R"] = mx.zeros(
                (parameter.shape[1], parameter.shape[1]),
                dtype=parameter.dtype,
            )
            state["Q_L"] = mx.zeros(
                (parameter.shape[0], parameter.shape[0]),
                dtype=parameter.dtype,
            )
            state["Q_R"] = mx.zeros(
                (parameter.shape[1], parameter.shape[1]),
                dtype=parameter.dtype,
            )

    def update_eigs(self, parameter: mx.array, state: dict) -> None:
        if parameter.ndim == 2:
            _, Q_L = mx.linalg.eigh(state['L'], stream=mx.Device(mx.cpu))
            state["Q_L"] = Q_L
            _, Q_R = mx.linalg.eigh(state['R'], stream=mx.Device(mx.cpu))
            state["Q_R"] = Q_R
    
    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict) -> mx.array:
        """Performs the SOAP Update"""
        if parameter.ndim == 2:
            b3 = self.beta_soap
            state["L"] = b3 * state["L"] + (1 - b3) * gradient @ gradient.T
            state["R"] = b3 * state["R"] + (1 - b3) * gradient.T @ gradient

            if self.step % self.eig_update_freq == 0:
                self.update_eigs(parameter, state)

            Q_L = state["Q_L"]
            Q_R = state["Q_R"]

            # rotate param and grad
            gradient = Q_L.T @ gradient @ Q_R
            parameter = Q_L.T @ parameter @ Q_R

        updated_parameter = super().apply_single(
            gradient=gradient,
            parameter=parameter,
            state=state,
        )

        if parameter.ndim == 2:
            updated_parameter = Q_L @ updated_parameter @ Q_R.T

        return updated_parameter