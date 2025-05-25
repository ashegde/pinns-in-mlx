"""
This module trains a PINN on the 1D Burgers equation,
in accordance with the specifications in configs.
"""

import matplotlib.pyplot as plt
import mlx.core as mx
import mlx.nn as nn

from configs.config_loader import load_config
from equations.equations import Burgers1DEquation
from models.models import MLP
from models.losses import PhysicsLoss, DirichletLoss 

from configs.logger import Logger
from optimizers.soap import SOAP
from samplers.sampler import UniformSampler
from solvers.pinn_solver import PINNSolver


path_to_config = "configs/config_burgers_1d.yaml"

config = load_config(path=path_to_config)
equation = Burgers1DEquation(nu=config.problem.viscosity)
physics_loss = PhysicsLoss(
    equation=equation,
    weight=config.optimizer.loss_weights.physics,
) 

initial_loss = DirichletLoss(
    target_fn=config.problem.initial_condition_expr,
    weight=config.optimizer.loss_weights.initial,
)

boundary_loss = DirichletLoss(
    target_fn=config.problem.boundary_condition_expr,
    weight=config.optimizer.loss_weights.boundary,
)

model = MLP(config.model.layers)

optimizer = SOAP(
    learning_rate=config.optimizer.learning_rate,
    betas=config.optimizer.betas_adamw,
    beta_soap=config.optimizer.beta_soap,
    weight_decay=config.optimizer.weight_decay,
    eig_update_freq=config.optimizer.update_freq,
)

# from mlx.optimizers import AdamW
# optimizer = AdamW(learning_rate=config.optimizer.learning_rate)

sampler = UniformSampler(
    batch_sizes=config.optimizer.batch_sizes,
    t_lims=config.problem.domain.t,
    x_lims=config.problem.domain.x,
    y_lims=config.problem.domain.y,
)

logger = Logger(config=config.logging)

# setup PINN solver and train model

solver = PINNSolver(
    model=model,
    physics_loss=physics_loss,
    boundary_loss=boundary_loss,
    initial_loss=initial_loss,
    optimizer=optimizer,
    sampler=sampler,
    logger=logger,
)
solver.train(config.optimizer.epochs)

## Plotting

# Loss

# Load log
losses = [float(line.strip()) for line in logger.load_log()]

plt.figure(figsize=(10, 6))
plt.plot(losses, label='Loss')
plt.yscale('log')  # Logarithmic scale for y-axis
plt.xlabel('Iteration')
plt.ylabel('Loss (log scale)')
plt.title('Training Loss')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.tight_layout()
plt.savefig(f"{logger.config.output_dir}/loss.png")

## Prediction
n_grid = 100
x_domain = mx.linspace(config.problem.domain.x[0], config.problem.domain.x[1], n_grid)  # 100 x-values from 0 to 1
t_domain = mx.linspace(config.problem.domain.t[0], config.problem.domain.t[1], n_grid)  # 100 t-values from 0 to 2

# Create meshgrid (note: x is vertical axis, so use meshgrid accordingly)
T, X = mx.meshgrid(t_domain, x_domain)  # shape: (n_grid, n_grid)

# Evaluate f on the grid
Z = model(X.reshape((-1,1)), T.reshape((-1,1))).reshape((n_grid, n_grid))
# Define domains
n_grid = 100
x_domain = mx.linspace(config.problem.domain.x[0], config.problem.domain.x[1], n_grid)  # 100 x-values from 0 to 1
t_domain = mx.linspace(config.problem.domain.t[0], config.problem.domain.t[1], n_grid)  # 100 t-values from 0 to 2

# Create meshgrid (note: x is vertical axis, so use meshgrid accordingly)
T, X = mx.meshgrid(t_domain, x_domain)  # shape: (n_grid, n_grid)

# Evaluate f on the grid
Z = model(X.reshape((-1,1)), T.reshape((-1,1))).reshape((n_grid, n_grid))

# Plotting
plt.figure(figsize=(8, 6))
contour = plt.contourf(T, X, Z, levels=50, cmap='magma')  # filled contour
plt.colorbar(contour, label='u(x, t)')
plt.xlabel('t')
plt.ylabel('x')
plt.title(f"Prediction (viscosity = {config.problem.viscosity: 0.3f})")
plt.tight_layout()
plt.savefig(f"{logger.config.output_dir}/prediction.png")