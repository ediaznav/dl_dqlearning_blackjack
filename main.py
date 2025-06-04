
# main.py

# Este es el código principal que permite el entrenamiento de las 
# redes neuronales para una configuración dada. 
#   


# Importamos las librerias a utilizar 
import argparse
from config import HyperParams
from dataclasses import replace, asdict
from train import train
import json
import os
from datetime import datetime

from src.utils.tools import generate_config_id


def get_args():
    """
    Procesa los argumentos de línea de comandos para configurar el entrenamiento.

    :return: Objeto argparse.Namespace con los argumentos proporcionados.
    """
    parser = argparse.ArgumentParser(description="Corriendo RL BlackJack training")

    # Ruta al archivo de configuración YAML
    parser.add_argument("--config", type=str, help="Path al YAML config file")

    # Sobrescrituras opcionales desde la CLI
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
    parser.add_argument("--DUELING", type=bool)
    parser.add_argument("--RENDER", action="store_true")

    return parser.parse_args()


def build_config(args) -> HyperParams:
    """
    Crea una configuración de entrenamiento combinando el archivo YAML
    con posibles sobrescrituras desde línea de comandos.

    :param args: Objeto Namespace con los argumentos CLI.
    :return: Instancia de HyperParams con la configuración final.
    """
    # Cargar configuración base desde archivo YAML o usar valores por defecto
    if args.config:
        config = HyperParams.from_yaml(args.config)
    else:
        config = HyperParams()

    # Reemplazar atributos definidos por el usuario vía CLI
    overrides = {
        k: v for k, v in vars(args).items()
        if v is not None and k != "config"
    }
    return replace(config, **overrides)


def log_config(config: HyperParams):
    """
    Guarda la configuración final usada para entrenamiento en un archivo JSON
    identificado por un hash único.

    :param config: Objeto de configuración HyperParams.
    :return: None
    """
    log_dir = "logs/hyper_confs/"
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_id = generate_config_id(config)
    path = os.path.join(log_dir, f"config_{config_id}.json")

    with open(path, "w") as f:
        json.dump(asdict(config), f, indent=4)

    print(f"[INFO] Configuración guardada en: {path}")


if __name__ == "__main__":
    # Procesar argumentos CLI
    args = get_args()

    # Construir configuración final
    config = build_config(args)

    print("[INFO] Configuración final cargada:")
    print(config)

    # Guardar configuración
    log_config(config)

    # Ejecutar entrenamiento principal
    train(config)
