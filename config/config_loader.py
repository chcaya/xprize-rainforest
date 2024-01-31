import yaml


def load_yaml_config(path):
    return yaml.safe_load(open(path, "rb"))


DEFAULT_CONFIG = load_yaml_config('./config/config.yaml')
