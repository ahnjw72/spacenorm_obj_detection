# https://chatgpt.com/share/692c1309-c910-8009-8fae-3a945fc961c2

import json
import yaml
from types import SimpleNamespace

class ConfigManager:
    def __init__(self, cli_args):
        self.cli_args = cli_args
        self.config = SimpleNamespace()

        # Load order: JSON → YAML → CLI
        self.load_json(cli_args.common_config)
        # self.load_yaml(cli_args.model_cfg)
        self.apply_cli_overrides()

    def load_json(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        for k, v in data.items():
            setattr(self.config, k, v)

    def load_yaml(self, path):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        setattr(self.config, "yolo", data)

    def apply_cli_overrides(self):
        cli = vars(self.cli_args)
        for k, v in cli.items():
            if v is not None:
                setattr(self.config, k, v)

    def reload(self):
        print("[INFO] Reloading configs ...")
        self.load_json(self.cli_args.config)
        # self.load_yaml(self.cli_args.model_cfg)
        self.apply_cli_overrides()

    def __getattr__(self, item):
        return getattr(self.config, item)

