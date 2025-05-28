import yaml
from configs.schema import ConfigSchema

def load_config(path: str) -> ConfigSchema:
    with open(path, "r") as f:
        config_dict = yaml.safe_load(f)

    return ConfigSchema(**config_dict)
