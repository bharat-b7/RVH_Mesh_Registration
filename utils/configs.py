import yaml
from pathlib import Path


def load_config(config_path):
    with open(config_path, 'r') as fp:
        try:
            config = yaml.safe_load(fp)
        except yaml.YAMLError as exc:
            print(exc)

    config = {k: Path(v) if isinstance(v, str) else v for k, v in config.items()}

    return config