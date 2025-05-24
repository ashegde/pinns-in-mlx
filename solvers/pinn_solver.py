from typing import List
from functools import partial

import mlx.core as mx
import mlx.nn as nn
from mlx.optimizers import Optimizer

from configs.logger import Logger
from samplers.sampler import Sampler
from models.losses import Loss 

class PINNSolver:
    def __init__(self,
                 model: nn.Module,
                 physics_loss: Loss,
                 boundary_loss: Loss,
                 initial_loss: Loss,
                 optimizer: Optimizer,
                 sampler: Sampler,
                 logger: Logger
                 ):
        self.model = model
        self.physics_loss = physics_loss
        self.boundary_loss = boundary_loss
        self.initial_loss = initial_loss
        self.optimizer = optimizer
        self.sampler = sampler
        self.logger = logger

    def loss_fn(self,
                x: mx.array,
                t: mx.array,
                x_ic: mx.array,
                t_ic: mx.array,
                x_bc: mx.array,
                t_bc: mx.array) -> mx.array:
        physics_loss = self.physics_loss(self.model, x, t)
        bc_loss = self.boundary_loss(self.model, x_bc, t_bc)
        ic_loss = self.initial_loss(self.model, x_ic, t_ic)
        return physics_loss + bc_loss + ic_loss
    
    def train(self, n_iter: int):
        self.model.train()
        state = [self.model.state, self.optimizer.state]

        #@partial(mx.compile, inputs=state, outputs=state) # soap optimizer needs to be rewritten with this in mind
        def step(x: mx.array,
                 t: mx.array,
                 x_ic: mx.array,
                 t_ic: mx.array,
                 x_bc: mx.array,
                 t_bc: mx.array) -> mx.array:
            loss_and_grad_fn = nn.value_and_grad(self.model, self.loss_fn)
            loss, grads = loss_and_grad_fn(x, t, x_ic, t_ic, x_bc, t_bc)
            self.optimizer.update(self.model, grads)
            return loss

        for ii in range(n_iter):
            batch = self.sampler.get_batch()
            x, t = batch['solution']
            x_ic, t_ic = batch['initial']
            x_bc, t_bc = batch['boundary']
            loss = step(x, t, x_ic, t_ic, x_bc, t_bc)
            mx.eval(state)
            # Write to log file
            self.logger.log(f"{loss: 0.5e}")
            if self.logger.config.save_model:
                if ii % self.logger.config.checkpoint_freq == 0:
                    self.logger.checkpoint(self.model, f"checkpoint_{ii}.npz")