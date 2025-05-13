import hashlib
import json
from dataclasses import asdict

def generate_config_id(config, exclude_keys=None) -> str:
    """
    Generates a short SHA256 hash string from the config.
    Parameters:
        config (HyperParams): dataclass instance of your config.
        exclude_keys (list): list of config keys to ignore when generating ID.
    Returns:
        str: 10-character hash ID.
    """
    config_dict = asdict(config)
    if exclude_keys:
        config_dict = {k: v for k, v in config_dict.items() if k not in exclude_keys}

    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:10]
