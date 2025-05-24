import yaml
from configs.config_schema import ConfigSchema

def load_config(path: str) -> ConfigSchema:
    with open(path, "r") as f:
        config_dict = yaml.safe_load(f)

    # config_dict['problem']['initial_condition_expr'] = lambda x: eval(
    #     config_dict['problem']['initial_condition_expr']
    # )
    # config_dict['problem']['boundary_condition_expr'] = lambda x: eval(
    #     config_dict['problem']['boundary_condition_expr']
    # )
    return ConfigSchema(**config_dict)
