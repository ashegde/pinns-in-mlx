from typing import List

import matplotlib.pyplot as plt
import mlx.core as mx

from configs.schema import ConfigSchema
from models.models import MLP


def plot_field(model: MLP, config: ConfigSchema, save_name: str):
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
    plt.savefig(save_name)

def plot_losses(records: List[dict]):
    plt.figure(figsize=(10, 6))
    for r in records:
        rows = [line.split("\t") for line in r["logger"].load_log()]
        iterations = [int(row[0].strip()) for row in rows]
        losses = [float(row[1].strip()) for row in rows]
        plt.plot(iterations, losses, alpha=0.7, label=r["logger"].config.experiment_name)
    plt.yscale("log") 
    plt.xlabel("Iteration")
    plt.ylabel("Loss (log scale)")
    plt.title("Training Loss")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{records[0]["logger"].config.output_dir}/loss_combined.png")
    plt.close()
