import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Union, Optional, Dict, Any, Tuple


class SpacenormConfigLoader:
    """
    Loads configuration with precedence:
      1) Common config file path from env: SPACENORM_DEFAULT_CFG_FILE (required unless fallback provided)
      2) Optional server override file path from env: SPACENORM_SERVER_CFG_FILE (if env set and file exists)
      3) CLI overrides (highest priority)

    Output:
      - Flattened cfg object: cfg.<param>
    """

    ENV_DEFAULT_CFG = "SPACENORM_DEFAULT_CFG_FILE" # env var for default behavior config file path
    ENV_SERVER_ID = "SPACENORM_SERVER_ID" 
    ENV_SERVER_CFG = "SPACENORM_SERVER_CFG_FILE" # env var for behavior config file path overriding default behavior
    ENV_DEVICEKEY_FILE = "SPACENORM_DEVICEKEY_FILE" 
    ENV_SENSOR_CFG = "SPACENORM_SENSOR_CFG_FILE" # env var for sensor (e.g. CCTV, environment sensors) config file path
    ENV_LOG_DIR = "SPACENORM_LOG_DIR"
    ENV_RECORD_DETECTION_RESULT_DIR = "SPACENORM_RECORD_DETECTION_RESULT_DIR"

    def __init__(self, fallback_common_cfg: Optional[Union[str, Path]] = None):
        self.fallback_common_cfg = fallback_common_cfg

        # Internal state (filled during load)
        self._cfg_tree: Optional[Dict[str, Any]] = None
        self._flat_defaults: Optional[Dict[str, Any]] = None
        self._flat_help: Optional[Dict[str, str]] = None

    # -------------------------
    # Public API
    # -------------------------
    def load(self) -> SimpleNamespace:
        """
        End-to-end load:
          - resolve common config path
          - load base config tree
          - optionally apply server overrides
          - flatten
          - parse CLI
          - apply CLI overrides
          - return cfg namespace
        """
        self._cfg_tree = self._load_base_then_server_override()
        self._flat_defaults, self._flat_help = self._flatten_config(self._cfg_tree)

        parser = argparse.ArgumentParser(description="Spacenorm service")
        self._add_cli_args(parser, self._flat_defaults, self._flat_help)
        args = parser.parse_args()

        return self._build_cfg(self._flat_defaults, args)

    # -------------------------
    # Core steps
    # -------------------------
    def _resolve_common_config_path(self) -> Path:
        env_path = os.environ.get(self.ENV_DEFAULT_CFG, "").strip()

        if env_path:
            path = Path(env_path)
        elif self.fallback_common_cfg is not None:
            path = Path(self.fallback_common_cfg)
        else:
            raise RuntimeError(
                f"Missing required environment variable {self.ENV_DEFAULT_CFG} "
                "and no fallback config path was provided."
            )

        if not path.is_file():
            raise FileNotFoundError(f"Common config file does not exist: {path}")

        return path

    def _load_json(self, path: Union[str, Path]) -> Dict[str, Any]:
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _load_base_then_server_override(self) -> Dict[str, Any]:
        common_cfg_path = self._resolve_common_config_path()
        base_cfg = self._load_json(common_cfg_path)

        # server_id = os.environ.get(self.ENV_SERVER_ID, "").strip()
        # if not server_id:
        #     return base_cfg

        server_cfg_path = os.environ.get(self.ENV_SERVER_CFG, "").strip()
        server_cfg_path = Path(server_cfg_path)
        if not server_cfg_path.is_file():
            return base_cfg

        server_cfg = self._load_json(server_cfg_path)
        self._apply_server_overrides_inplace(base_cfg, server_cfg)
        return base_cfg

    def _apply_server_overrides_inplace(self, base_cfg: Dict[str, Any], server_cfg: Dict[str, Any]) -> None:
        """
        Override base_cfg[category][param]['value'] using server_cfg[category][param]['value'].

        Assumption:
          - All (category, param) keys in server_cfg exist in base_cfg.
        If unexpected keys appear, raises immediately.
        """
        for category, params in server_cfg.items():
            if category not in base_cfg:
                raise KeyError(f"Unknown category in server config: {category}")

            if not isinstance(params, dict):
                raise TypeError(
                    f"Expected dict for category '{category}', got {type(params).__name__}"
                )

            for name, meta in params.items():
                if name not in base_cfg[category]:
                    raise KeyError(f"Unknown parameter in server config: {category}.{name}")

                if not isinstance(meta, dict) or "value" not in meta:
                    raise KeyError(
                        f"Server config entry must be a dict with 'value': {category}.{name}"
                    )

                base_cfg[category][name]["value"] = meta["value"]

    # -------------------------
    # Flatten + CLI
    # -------------------------
    def _flatten_config(self, cfg_tree: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """
        Flatten categorized config into:
          flat_defaults: {param_name: value}
          flat_help:     {param_name: help_text}

        Fails fast if duplicates appear across categories.
        """
        flat_defaults: Dict[str, Any] = {}
        flat_help: Dict[str, str] = {}
        seen_in: Dict[str, str] = {}

        for category, params in cfg_tree.items():
            if not isinstance(params, dict):
                raise TypeError(
                    f"Expected dict for category '{category}', got {type(params).__name__}"
                )

            for name, meta in params.items():
                if name in flat_defaults:
                    raise ValueError(
                        f"Duplicate parameter '{name}' in categories "
                        f"'{seen_in[name]}' and '{category}'. Flattening would be ambiguous."
                    )

                if not isinstance(meta, dict) or "value" not in meta:
                    raise KeyError(f"Invalid config entry (missing 'value'): {category}.{name}")

                flat_defaults[name] = meta["value"]
                flat_help[name] = meta.get("help", "")
                seen_in[name] = category

        return flat_defaults, flat_help

    def _add_cli_args(
        self,
        parser: argparse.ArgumentParser,
        flat_defaults: Dict[str, Any],
        flat_help: Dict[str, str],
    ) -> None:
        """
        Adds CLI args for each flattened parameter.

        Bool handling uses toggle-default semantics:
          - default False -> --flag sets True
          - default True  -> --flag sets False
        """
        for name, default in flat_defaults.items():
            flag = f"--{name}"
            help_text = flat_help.get(name, "")

            if isinstance(default, bool):
                if default is False:
                    parser.add_argument(flag, action="store_true", help=help_text)
                else:
                    parser.add_argument(flag, action="store_false", help=help_text)
            else:
                parser.add_argument(
                    flag,
                    type=type(default),
                    default=None,  # None => not provided, so no override
                    help=help_text,
                )

    def _build_cfg(self, flat_defaults: Dict[str, Any], args: argparse.Namespace) -> SimpleNamespace:
        """
        Build cfg.<param> object, applying CLI overrides last.
        """
        final: Dict[str, Any] = {}

        devicekey_file_path = os.environ.get(self.ENV_DEVICEKEY_FILE, "").strip()
        final["spacenorm_devicekey_file"] = devicekey_file_path if devicekey_file_path else ""

        sensor_cfg_file_path = os.environ.get(self.ENV_SENSOR_CFG, "").strip()
        final["spacenorm_sensor_cfg_file"] = sensor_cfg_file_path if sensor_cfg_file_path else ""

        log_dir = os.environ.get(self.ENV_LOG_DIR, "").strip()
        final["log_dir"] = log_dir if log_dir else ""

        record_detection_result_dir = os.environ.get(self.ENV_RECORD_DETECTION_RESULT_DIR, "").strip()
        final["record_detection_result_dir"] = record_detection_result_dir if record_detection_result_dir else ""

        for name, default in flat_defaults.items():
            cli_value = getattr(args, name, None)

            if isinstance(default, bool):
                # argparse yields bool always; treat as override only if differs from default
                final[name] = cli_value if cli_value != default else default
            else:
                final[name] = default if cli_value is None else cli_value

        return SimpleNamespace(**final)


if __name__ == "__main__":
    # Strict mode: will raise if SPACENORM_DEFAULT_CFG_FILE is not set.
    loader = SpacenormConfigLoader()
    cfg = loader.load()

    print(cfg.background_thresh1)
    print(cfg.log_dir)
    print(cfg.no_display)
