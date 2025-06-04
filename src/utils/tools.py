# tools.py

# Diferentes funciones auxiliares a lo largo del proceso de 
# entrenamiento y de prueba. 
# 

# Importamos las librerias requeridas 
import hashlib
import json
from dataclasses import asdict


def generate_config_id(config, exclude_keys=None) -> str:
    """
    Genera un hash corto (SHA256 truncado) a partir de un diccionario
    de configuración, útil para identificar de forma única distintos
    experimentos o configuraciones de entrenamiento.

    :param config: Instancia de dataclass (por ejemplo, HyperParams).
    :param exclude_keys: Lista de claves que se excluirán del hash.
    :return: Cadena de 10 caracteres que representa el hash único.
    """
    # Convertir la configuración a diccionario
    config_dict = asdict(config)

    # Eliminar claves excluidas si se especifican
    if exclude_keys:
        config_dict = {
            k: v for k, v in config_dict.items()
            if k not in exclude_keys
        }

    # Serializar y ordenar el diccionario para asegurar consistencia
    config_str = json.dumps(config_dict, sort_keys=True)

    # Generar hash SHA256 y truncarlo a los primeros 10 caracteres
    return hashlib.sha256(config_str.encode()).hexdigest()[:10]
