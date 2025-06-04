# training.py

# Este script continene el codigo necesario para ejecutar un entrenamiento
# de las redes dado una configuracion particualar de los hiper 
# parametros. 


### Librerías --------------------------------------------------------------

# Importamos las librerías necesarias 

# Sistema 
import os

# Manejo de datos
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Modelos ML
import torch
from torch.utils.tensorboard import SummaryWriter

# Librerías del entorno y utilidades
from src.envs.blackjack_env import BlackjackMultiPlayerEnv
from src.agents.dqn_agent import DQNAgent
from src.utils.logger import init_loggers, log_step, log_summary
from src.utils.tools import generate_config_id

# Configuración e hiperparámetros
from config import HyperParams


def train(config: HyperParams):
    """
    Entrena un agente DQN en el entorno de Blackjack usando configuración definida.

    :param config: Instancia de HyperParams con los valores para entrenamiento.
    :return: None
    """

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[TRAIN] Empezando el entrenamiento con:")
    print(config)

    # Se importa dinámicamente la red Q correspondiente (dueling o no)
    if config.DUELING:
        from src.agents.dqn_agent import DuelingQNetwork as QNetwork  
        DIRECTORY_LOGS = "logs/DQnet/"
    else:
        from src.agents.dqn_agent import QNetwork  
        DIRECTORY_LOGS = "logs/Qnet/"

    # Desempaquetar hiperparámetros desde la configuración
    N_EPISODES = config.N_EPISODES
    INITIAL_BANKROLL = config.INITIAL_BANKROLL
    GAMMA = config.GAMMA
    LR = config.LR
    EPSILON_START = config.EPSILON_START
    EPSILON_END = config.EPSILON_END
    EPSILON_DECAY = config.EPSILON_DECAY
    BATCH_SIZE = config.BATCH_SIZE
    MEMORY_SIZE = config.MEMORY_SIZE
    TARGET_UPDATE_FREQ = config.TARGET_UPDATE_FREQ
    RENDER = config.RENDER

    VERBOSE_FREQ = int(N_EPISODES / 100)

    # LOGGING -------------------------------------------------------------
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    DIRECTORY_TENSORBOARD = f"./runs/blackjack_{TIMESTAMP}"
    os.makedirs(DIRECTORY_LOGS, exist_ok=True)

    init_loggers(base_dir=DIRECTORY_LOGS)
    print(f"Generando logs en : {DIRECTORY_LOGS}")

    # ENTORNO Y AGENTE ----------------------------------------------------
    env = BlackjackMultiPlayerEnv(render_mode="rgb_array", initial_bankroll=INITIAL_BANKROLL)
    agent = DQNAgent(
        state_dim=3,
        action_dim=4,
        lr=LR,
        gamma=GAMMA,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay=EPSILON_DECAY,
        memory_size=MEMORY_SIZE,
        batch_size=BATCH_SIZE,
        target_update_freq=TARGET_UPDATE_FREQ
    ) 

    writer = SummaryWriter(log_dir=DIRECTORY_TENSORBOARD)
    reward_history = []

    # Bucle principal de episodios
    for episode in range(N_EPISODES):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        step_id = 0

        # Interacciones por episodio
        while not done:
            legal_actions = env.get_legal_actions()
            action = agent.act(state, legal_actions=legal_actions)

            next_state, reward, done, _, _ = env.step(action)

            # Verificar si se quebró el bankroll
            bankrupt = env.bankroll <= 0
            agent.remember((state, action, reward, next_state, done or bankrupt))
            agent.update()

            # Logging en TensorBoard (paso a paso)
            writer.add_scalar("Step/Reward", reward, episode * 100 + steps)
            writer.add_scalar("Step/Epsilon", agent.epsilon, episode * 100 + steps)
            writer.add_scalar("Step/Bankroll", env.bankroll, episode * 100 + steps)

            log_step(episode, step_id, 
                     state, action, reward, agent.epsilon, env.bankroll,
                     base_dir=DIRECTORY_LOGS)

            state = next_state
            episode_reward += reward
            step_id += 1
            steps += 1

            if RENDER:
                env.render()

        # Reiniciar bankroll si se ha quebrado
        if env.bankroll <= 0:
            env.bankroll = env.initial_bankroll

        reward_history.append(episode_reward)

        # Logging en TensorBoard (resumen por episodio)
        writer.add_scalar("Episode/Reward", episode_reward, episode)
        writer.add_scalar("Episode/Epsilon", agent.epsilon, episode)

        if episode % VERBOSE_FREQ == 0:
            avg_reward = np.mean(reward_history[-50:])
            print(f"Episode {episode} | AvgReward: {avg_reward:.2f} | Epsilon: {agent.epsilon:.3f}")
            writer.add_scalar("Episode/AvgReward_50", avg_reward, episode)

        # Log CSV resumen por episodio
        log_summary(episode, env.player, env.dealer, reward, env.bankroll, base_dir=DIRECTORY_LOGS)

    # Guardar modelo entrenado
    config_id = generate_config_id(config)
    if config.DUELING:
        model_path = f"models/q_network_dueling_{config_id}.pth"
    else:
        model_path = f"models/q_network_{config_id}.pth"

    torch.save(agent.policy_net.state_dict(), model_path)
    print(f"[INFO] Modelo guardado en: {model_path}")
    writer.close()
