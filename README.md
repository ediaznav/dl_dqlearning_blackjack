# Entrenando un agente a jugar BlackJack con Reinforcement Deep Q-Learning

Descripción del proyecto 

## El entorno de simulación

Se simula  un entorno de BlackJack. 

### Custom Environment 


Definir el algoritmo de generaciónd de juegos. 


### OpenAI Gym

Definir y usar como Benchmark 



## El entrenamiento 

### Arquitectura de la Red Neuronal

Describir la arquitectura seleccionada


### Definir el proceso 


** Cómo se ejecuto el código  **

Se entreno el código ejecutando la siguiente línea para utilizar los hipér parámetros de un archivo de configuración. 

``` zsh
python main.py --config config.yaml
```

O para hacerlo modificando los parámetros en la línea de comandos:

``` zsh
python main.py --config config.yaml --LR 0.0002 --RENDER
```

O con el default: 

``` zsh
python main.py -
```

Y con los parametros dados por:

``` python
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
RENDER: bool = False
DUELING: bool = False
```

## Evaluación

Evaluamos los diferentes modelos para conocer su desempeño. 

## Testing 

Finalmente, de mejor modelo, evaluamos para tener métricas finales. 
