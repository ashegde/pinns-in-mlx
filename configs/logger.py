import os

import mlx.nn as nn

from configs.schema import LogConfig

class ImproperFileExtensionError(Exception): pass

class Logger:

    def __init__(self, config: LogConfig):
        self.config = config
        self.dir = f"{self.config.output_dir}/{self.config.experiment_name}/"
        os.makedirs(self.dir, exist_ok=True)

    def log(self, text: str):
        with open(f"{self.dir}/{self.config.log_file}", "a") as file:
            file.write(text+"\n")

    def checkpoint(self, model: nn.Module, file_name: str):
        if file_name.split(".")[-1] not in ("npz", "safetensors"):
            raise ImproperFileExtensionError("File extension must be .npz or .safetensors")
        model.save_weights(file=f"{self.dir}/{file_name}")

    def load_log(self) -> list[str]:
        log = []
        with open(f"{self.dir}/{self.config.log_file}", "r") as file:
            for line in file:
                log.append(line)
        return log