from dataclasses import dataclass
import yaml

@dataclass
class HyperParams:
    N_EPISODES: int = 10_000_000
    INITIAL_BANKROLL: int = 100
    GAMMA: float = 0.99
    LR: float = 1e-3
    EPSILON_START: float = 1.0
    EPSILON_END: float = 0.01
    EPSILON_DECAY: float = 0.999995
    BATCH_SIZE: int = 64
    MEMORY_SIZE: int = 100_000
    TARGET_UPDATE_FREQ: int = 100
    DUELING: bool = False
    RENDER: bool = False

    @staticmethod
    def from_yaml(path: str) -> "HyperParams":
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return HyperParams(**data)
