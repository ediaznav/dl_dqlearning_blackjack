# test.py

### Librerias --------------------------------------------------------------

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from src.utils.logger import init_loggers, log_step, log_summary


from src.envs.blackjack_env import BlackjackMultiPlayerEnv
from src.agents.dqn_agent import QNetwork
#from blackjack_ui import BlackjackUI

### EVALUATION CONFIG ------------------------------------------------------
MODEL_PATH = "models/q_network_dueling_b845945d4b.pth" #"models/q_network_a006637e49.pth"
RESUlTS_PATH = "./data/results/"
N_EPISODES = 1000
RENDER = False
INITIAL_BANKROLL = 100
DIRECTORY_LOGS = "logs/test/"

### ENVIRONMENT ------------------------------------------------------------
env = BlackjackMultiPlayerEnv(render_mode="human", initial_bankroll=INITIAL_BANKROLL)
#ui = BlackjackUI() if RENDER else None

state_dim = 3
action_dim = 4

policy_net = QNetwork(state_dim, action_dim)
policy_net.load_state_dict(torch.load(MODEL_PATH))
policy_net.eval()


### EVALUACION -------------------------------------------------------------
# Métricas init
results = {"win": 0, "loss": 0, "draw": 0, "surrender": 0}
eventos = {"dealer_hand": [], "player_hand": [],
            "result": [], "reward": [], "total_reward": [], 
            "bankroll":[]}

total_rewards = []
final_bankrolls = []
states_p = []
states_d = []
actions= []

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

        
        
        step_player_h, step_dealer_h, _ = state
        actions.append(action)
        states_p.append(step_player_h)
        states_d.append(step_dealer_h)

        next_state, reward, done, _, _ = env.step(action)
        state = next_state
        total_reward += reward

        """if RENDER:
            ui.render(
                dealer_hand=env.dealer,
                player_hand=env.player,
                bankroll=env.bankroll,
                bet=env.current_bet,
                info_text=f"Episode {episode + 1}"
            )"""

    # Resultado
    player_val = sum(env.player) + 10 if (1 in env.player and sum(env.player) + 10 <= 21) else sum(env.player)
    dealer_val = sum(env.dealer) + 10 if (1 in env.dealer and sum(env.dealer) + 10 <= 21) else sum(env.dealer)
    result = interpret_result(player_val, dealer_val, reward)
    results[result] += 1
    eventos["player_hand"].append(player_val)
    eventos["dealer_hand"].append(dealer_val)
    eventos["result"].append(result)
    eventos["reward"].append(reward)
    eventos["total_reward"].append(total_reward)
    eventos["bankroll"].append(env.bankroll)
    total_rewards.append(total_reward)
    final_bankrolls.append(env.bankroll)
### RESUMEN ----------------------------------------------------------------

pd.DataFrame(results, 
             index = [list(results.keys())]).to_csv(RESUlTS_PATH + "results_duel.csv",
                              index = False)

pd.DataFrame(eventos).to_csv(RESUlTS_PATH + "eventos_duel.csv",
                              index = False)

pd.DataFrame({
    "dealer_hand":states_d, "player_hand":states_p,
      "action":actions
}).to_csv(RESUlTS_PATH + "actions_duel.csv", index = False)

print("\n EVALUACIÓN FINAL")
print(f"Episodios evaluados: {N_EPISODES}")
for k, v in results.items():
    print(f"{k.capitalize()}: {v} ({v / N_EPISODES:.1%})")
print(f"Recompensa media por episodio: {np.mean(total_rewards):.2f}")
print(f"Banca final promedio: {np.mean(final_bankrolls):.2f}")

"""if ui:
    ui.close()"""