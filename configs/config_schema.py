from pydantic import BaseModel
from typing import Callable, Optional

class DomainConfig(BaseModel):
    x: list[float]
    t: list[float]
    y: Optional[list[float]] = None


class ProblemConfig(BaseModel):
    name: str
    viscosity: float
    domain: DomainConfig
    initial_condition_expr: str
    boundary_condition_expr: str
    seed: int


class ModelConfig(BaseModel):
    type: str
    layers: list[int]


class SchedulerConfig(BaseModel):
    type: str
    step_size: int
    gamma: float


class LossConfig(BaseModel):
    physics: float
    initial: float
    boundary: float


class BatchSizes(BaseModel):
    solution: int
    initial: int
    boundary: int


class Optimizer(BaseModel):
    learning_rate: float
    betas_adamw: list[float]
    beta_soap: float
    weight_decay: float
    update_freq: int
    epochs: int
    loss_weights: LossConfig
    batch_sizes: BatchSizes


class LogConfig(BaseModel):
    output_dir: str
    log_file: str
    checkpoint_freq: int
    save_model: bool


class ConfigSchema(BaseModel):
    problem: ProblemConfig
    model: ModelConfig
    optimizer: Optimizer
    logging: LogConfig
