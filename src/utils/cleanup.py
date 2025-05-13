import shutil
import os

def reset_training_data(
    tensorboard_dir="runs",
    log_dir="logs",
    model_dir="models"
):
    """
    Deletes all training output: TensorBoard logs, CSV logs, and model files.
    Use with caution — this operation is irreversible.

    Args:
        tensorboard_dir (str): Directory containing TensorBoard logs.
        log_dir (str): Directory containing episode and step CSV logs.
        model_dir (str): Directory containing saved model checkpoints.
    """
    for path in [tensorboard_dir, log_dir, model_dir]:
        if os.path.exists(path):
            shutil.rmtree(path)
            print(f"Deleted directory: {path}")
        else:
            print(f"Directory not found (skipped): {path}")
