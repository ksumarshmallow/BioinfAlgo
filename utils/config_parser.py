import yaml
from os import path

class ConfigParser:
    @staticmethod
    def parse(file: str):
        if not path.exists(file):
            raise Exception(f'File "{file}" not exists')
        try:
            with open(file=file, mode="r") as config_file:
                config = yaml.load(config_file, Loader=yaml.SafeLoader)
        except Exception as config_parse_exception:
            raise Exception(f'Can not parse file "{file}"') from config_parse_exception
        return config
