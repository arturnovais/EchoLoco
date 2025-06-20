"""
Script para carregar o arquivo de configuração.
Execute o script com o comando:
python -m src.utils.load_config

O arquivo de configuração é o config.yaml.
"""

import os
import yaml

def load_config():
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(src_dir, "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)