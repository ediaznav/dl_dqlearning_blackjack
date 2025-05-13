# test.py


### Librerias --------------------------------------------------------------

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from src.envs.blackjack_env import BlackjackMultiPlayerEnv
from src.agents.dqn_agent import QNetwork
#from blackjack_ui import BlackjackUI

### EVALUATION CONFIG ------------------------------------------------------
MODEL_PATH = "models/q_network.pth"
N_EPISODES = 1000
RENDER = False
INITIAL_BANKROLL = 100

### ENVIRONMENT ------------------------------------------------------------
env = BlackjackMultiPlayerEnv(render_mode="human", initial_bankroll=INITIAL_BANKROLL)
ui = BlackjackUI() if RENDER else None

state_dim = 3
action_dim = 4

policy_net = QNetwork(state_dim, action_dim)
policy_net.load_state_dict(torch.load(MODEL_PATH))
policy_net.eval()


### EVALUACION -------------------------------------------------------------
# Métricas init
results = {"win": 0, "loss": 0, "draw": 0, "surrender": 0}
total_rewards = []
final_bankrolls = []

def interpret_result(player_val, dealer_val, reward):
    if reward == -0.5:
        return "surrender"
    elif player_val > 21:
        return "loss"
    elif dealer_val > 21:
        return "win"
    elif reward == 1:
        return "win"
    elif reward == 0:
        return "draw"
    elif reward == -1:
        return "loss"
    else:
        return "unknown"

# Evaluación
for episode in range(N_EPISODES):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = torch.argmax(policy_net(state_tensor)).item()

        next_state, reward, done, _, _ = env.step(action)
        state = next_state
        total_reward += reward

        if RENDER:
            ui.render(
                dealer_hand=env.dealer,
                player_hand=env.player,
                bankroll=env.bankroll,
                bet=env.current_bet,
                info_text=f"Episode {episode + 1}"
            )

    # Resultado
    player_val = sum(env.player) + 10 if (1 in env.player and sum(env.player) + 10 <= 21) else sum(env.player)
    dealer_val = sum(env.dealer) + 10 if (1 in env.dealer and sum(env.dealer) + 10 <= 21) else sum(env.dealer)
    result = interpret_result(player_val, dealer_val, reward)
    results[result] += 1
    total_rewards.append(total_reward)
    final_bankrolls.append(env.bankroll)


### RESUMEN ----------------------------------------------------------------
print("\n EVALUACIÓN FINAL")
print(f"Episodios evaluados: {N_EPISODES}")
for k, v in results.items():
    print(f"{k.capitalize()}: {v} ({v / N_EPISODES:.1%})")
print(f"Recompensa media por episodio: {np.mean(total_rewards):.2f}")
print(f"Banca final promedio: {np.mean(final_bankrolls):.2f}")

if ui:
    ui.close()