# cleanup.py

# Se definen funciones para limpieza y manejo de datos.  
# 
#

# Importamos las librerías necesarias 
import shutil
import os

# Fiunción para reiniciar datos de entrenamiento y registro 
def reset_training_data(
    tensorboard_dir="runs",
    log_dir="logs",
    model_dir="models"
):
    """
    Elimina todos los datos generados durante el entrenamiento:
    registros de TensorBoard, logs en CSV y modelos guardados.
    Esta operación no se puede deshacer.

    :param tensorboard_dir: Ruta a la carpeta de logs de TensorBoard.
    :param log_dir: Ruta a la carpeta de logs en CSV.
    :param model_dir: Ruta a la carpeta de modelos guardados.
    :return: None
    """
    # Recorre cada ruta y elimina la carpeta si existe
    for path in [tensorboard_dir, log_dir, model_dir]:
        if os.path.exists(path):
            shutil.rmtree(path)  # Elimina el directorio completo
            print(f"Directorio eliminado: {path}")
        else:
            print(f"Directorio no encontrado (omitido): {path}")

