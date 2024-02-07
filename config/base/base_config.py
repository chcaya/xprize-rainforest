import yaml


class BaseConfig:
    def __init__(self):
        pass

    @staticmethod
    def load_yaml_config(config_path):
        return yaml.safe_load(open(config_path, 'rb'))
