"""
This module trains a PINN on the 1D Burgers equation,
in accordance with the specifications in configs.
"""

import glob
from typing import List

from runner import run_experiment
from configs.config_loader import load_config
from plotting.plotting import plot_field, plot_losses

path_to_experiment_configs = "configs/experiments/*.yaml"
config_files = glob.glob(pathname=path_to_experiment_configs)
records: List[dict] = []
for config_path in config_files:
    config = load_config(path=config_path)
    model, logger = run_experiment(config=config)
    records.append({"model": model, "logger": logger})
    plot_field(model=model, config=config, save_name=f"{logger.dir}/prediction.png")

plot_losses(records=records)