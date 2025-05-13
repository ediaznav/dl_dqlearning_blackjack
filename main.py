import argparse
from config import HyperParams
from dataclasses import replace, asdict
from train import train
import json
import os
from datetime import datetime

from src.utils.tools import generate_config_id



def get_args():
    parser = argparse.ArgumentParser(description="Corriendo RL BlackJack training")
    parser.add_argument("--config", type=str, help="Path al YAML config file")

    # Optional CLI overrides
    parser.add_argument("--N_EPISODES", type=int)
    parser.add_argument("--INITIAL_BANKROLL", type=int)
    parser.add_argument("--GAMMA", type=float)
    parser.add_argument("--LR", type=float)
    parser.add_argument("--EPSILON_START", type=float)
    parser.add_argument("--EPSILON_END", type=float)
    parser.add_argument("--EPSILON_DECAY", type=float)
    parser.add_argument("--BATCH_SIZE", type=int)
    parser.add_argument("--MEMORY_SIZE", type=int)
    parser.add_argument("--TARGET_UPDATE_FREQ", type=int)
    parser.add_argument("--DUELING", type = bool)
    parser.add_argument("--RENDER", action="store_true")
    

    return parser.parse_args()

def build_config(args) -> HyperParams:
    if args.config:
        config = HyperParams.from_yaml(args.config)
    else:
        config = HyperParams()

    overrides = {k: v for k, v in vars(args).items() if v is not None and k != "config"}
    return replace(config, **overrides)

def log_config(config: HyperParams):
    log_dir = "logs/hyper_confs/"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_id = generate_config_id(config)
    path = os.path.join(log_dir, f"config_{config_id}.json")
    with open(path, "w") as f:
        json.dump(asdict(config), f, indent=4)
    print(f"[INFO] Configuración guardada en: {path}")

if __name__ == "__main__":
    args = get_args()
    config = build_config(args)
    print("[INFO] Configuración final cargada:")
    print(config)

    log_config(config)
    train(config)
