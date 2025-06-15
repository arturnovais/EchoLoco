import os
import yaml

def load_config():
    """
    Carrega o config.yaml que está na pasta src,
    independente de onde for chamada.
    """
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(src_dir, "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)