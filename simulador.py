# simulador.py

### Librerias --------------------------------------------------------------

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from src.utils.logger import init_loggers, log_step, log_summary
import streamlit as st
import time
import random 

from src.envs.blackjack_env import BlackjackMultiPlayerEnv
from src.agents.dqn_agent import QNetwork

plt.style.use("ggplot")

### EVALUATION CONFIG ------------------------------------------------------
MODEL_PATH = "models/q_network_a006637e49.pth"
MODEL_PATH_DUELING = "models/q_network_dueling_b845945d4b.pth" 
RESUlTS_PATH = "./data/results/"
N_EPISODES = 1000
RENDER = False
INITIAL_BANKROLL = 100
DIRECTORY_LOGS = "logs/test/"

### ENVIRONMENT ------------------------------------------------------------
env = BlackjackMultiPlayerEnv(render_mode="human", initial_bankroll=INITIAL_BANKROLL)

state_dim = 3
action_dim = 4



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
    

def card_display(number_list):
    card_symbols = {
        1: "🂡",
        2: "🂢",
        3: "🂣",
        4: "🂤",
        5: "🂥",
        6: "🂦",
        7: "🂧",
        8: "🂨",
        9: "🂩",
        10: "🂪",
        11: "🂫",
        12: "🂭",
        13: "🂮"
    }
    cartas = ""
    for num in number_list:
        emoji = card_symbols.get(num, "🂠")
        cartas += emoji

    return cartas




def run_simulation_live(n_simulations, model, velocidad):
    wins = 0
    losses = 0
    draws = 0
    surrenders = 0
    win_rates = []
    loss_rates = []
    surrender_rates = []
    draw_rates = []

    if model == "Vanilla":
        policy_net = QNetwork(state_dim, action_dim)
        policy_net.load_state_dict(torch.load(MODEL_PATH))
        policy_net.eval()
    else:
        policy_net = QNetwork(state_dim, action_dim)
        policy_net.load_state_dict(torch.load(MODEL_PATH_DUELING))
        policy_net.eval()

    st.markdown(f"Simulando {n_simulations} con el modelo '{model}'")

    # Create a 2-column layout
    col1, col2 = st.columns([1,1])  # Adjust widths as needed

    # Placeholders for left column (game state)
    with col1:
        game_number_placeholder = st.empty()
        dealer_hand_placeholder = st.empty()
        player_hand_placeholder = st.empty()
        final_action_placeholder = st.empty()
        result_placeholder = st.empty()

    # Right column: placeholder for evolving plot
    with col2:
        plot_placeholder = st.empty()

    for i in range(n_simulations):
        # Dummy logic – replace with your real blackjack RL logic
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            
            with torch.no_grad():
                action = torch.argmax(policy_net(state_tensor)).item()

            step_player_h, step_dealer_h, _ = state
            dealer_hand = step_dealer_h
            player_hand = step_player_h
            final_action = action

            next_state, reward, done, _, _ = env.step(action)
            state = next_state
            total_reward += reward

            
        player_val = sum(env.player) + 10 if (1 in env.player and sum(env.player) + 10 <= 21) else sum(env.player)
        dealer_val = sum(env.dealer) + 10 if (1 in env.dealer and sum(env.dealer) + 10 <= 21) else sum(env.dealer)
        result = interpret_result(player_val, dealer_val, reward)

        # Update left column
        with col1:
            game_number_placeholder.markdown(f"### Juego {i+1} / {n_simulations}")
            dealer_hand_placeholder.markdown(f"**Carta que muestra el Dealer:** {card_display(env.dealer)} ({dealer_hand})")
            player_hand_placeholder.markdown(f"**Mano del jugador:** {card_display(env.player)} ({player_hand})")
            final_action_placeholder.markdown(f"**Acción Final:** {final_action}")
            result_placeholder.markdown(f"**Resultado:** {result}")

        # Update win/loss counts
        if result == "win":
            wins += 1
        elif result == "surrender":
            surrenders += 1
        elif result == "draw":
            draws+=1
        else:
            losses +=1

        # Calculate evolving win rate
        win_rate = wins / (i + 1)
        win_rates.append(win_rate)
        surrenders_rate = surrenders / (i + 1)
        surrender_rates.append(surrenders_rate)
        loss_rate = losses / (i + 1)
        loss_rates.append(loss_rate)
        draw_rate = draws/ (i + 1)
        draw_rates.append(draw_rate)


        # Update right column plot
        with col2:
            fig, ax = plt.subplots()
            ax.plot(win_rates, color="darkgreen", label="Tasa de victorias")
            ax.plot(surrender_rates, color="steelblue", label="Tasa de abandono")
            ax.plot(loss_rates, color="salmon ", label="Tasa de derrota")
            ax.plot(draw_rates, color="gray", label="Tasa de empate")
            ax.set_xlim(0, n_simulations)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Juego")
            ax.set_ylabel("Rates")
            ax.set_title("Evolución del desempeño del agente.")
            ax.legend()
            plot_placeholder.pyplot(fig)

        time.sleep(int(velocidad))

    # Crea un DataFrame
    summary_data = {
        "Estadística": [
            "Victorias totales",
            "Derrotas totales",
            "Abandonos totales",
            "Empates totales",
            "Tasa de victoria",
            "Tasa de derrota",
            "Tasa de abandono",
            "Tasa de empates"
        ],
        "Valor": [
            wins,
            losses,
            surrenders,
            draws,
            f"{wins / n_simulations:.2%}",
            f"{losses / n_simulations:.2%}",
            f"{surrenders / n_simulations:.2%}",
            f"{draws / n_simulations:.2%}"
        ]
    }

    summary_df = pd.DataFrame(summary_data)
    summary_df.Valor = summary_df.Valor.astype("str")
    summary_df.set_index(summary_df["Estadística"], inplace=True)
    summary_df.drop(columns=["Estadística"], inplace = True)
    st.success("¡Simulación completada!")
    st.markdown(f"Resumen de la simulación")
    st.table(summary_df.T)


def main():
    st.title("Blackjack RL - Simulador Visual")
    st.header("MCD ITAM - Deep Learning")
    st.subheader("Proyecto Final")

    # Small text box with no line spacing
    st.markdown("""
    <div style='font-size:15px; line-height:1; border:1px solid #ccc; padding:5px; border-radius:4px;'>
    <strong>Instrucciones:</strong><br>
    1.- Selecciona un entero entre 1 a 1000.<br>
    2.- Selecciona el modelo.<br>
    3.- Selecciona la velocidad de partida (en segundos).<br>
    4.- Haz clic en el botón 'Jugar'.
    
    </div>
    """, unsafe_allow_html=True)

    pcol1, pcol2, pcol3= st.columns([4,4,1])
 
    with pcol1:
        n_simulations = st.number_input("Número de simulaciones:", min_value=1, max_value=1000, value=50)
    with pcol2:
        model = st.selectbox("Seleccionar modelo:", ["Dueling","Vanilla"])
    with pcol3:
        velocidad = st.selectbox("Velocidad:", ["1","2", "3", "4", "5"])
    
    if st.button("Jugar"):
            run_simulation_live(n_simulations, model, velocidad)

if __name__ == "__main__":
    main()
