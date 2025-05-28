import mlx.core as mx

from configs.schema import ConfigSchema
from equations.equations import Burgers1DEquation
from models.models import MLP
from models.losses import PhysicsLoss, DirichletLoss 
from configs.logger import Logger
from optimizers.load_optimizer import load_optimizer
from samplers.sampler import UniformSampler
from solvers.pinn_solver import PINNSolver

def run_experiment(config: ConfigSchema) -> tuple[MLP, Logger]:
    mx.random.seed(seed=config.problem.seed)
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

    optimizer = load_optimizer(optim_config=config.optimizer)

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

    print(f"\n{logger.config.experiment_name}\n")
    solver.train(config.optimizer.epochs)

    return model, logger