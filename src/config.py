from pathlib import Path

import yaml


def load_config(path):
    path = Path(path).resolve()
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["paths"] = {name: path.parent / value for name, value in config["paths"].items()}
    return config