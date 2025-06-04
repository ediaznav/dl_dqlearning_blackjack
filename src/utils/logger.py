
# logger.py

# Se definen funciones auxiliares para el manmejo de logs
# en el entorno de RL diseñado. 
# 


# Importamos las librarias necesarias 
import csv
import os


def init_loggers(base_dir="logs"):
    """
    Inicializa los archivos de log para el entrenamiento, incluyendo:
    - `episode_summary.csv`: resumen por episodio.
    - `step_trace.csv`: trazado paso a paso del agente.

    :param base_dir: Carpeta donde se guardarán los archivos de log.
    :return: None
    """
    os.makedirs(base_dir, exist_ok=True)

    # Inicializar resumen de episodios
    summary_path = os.path.join(base_dir, "episode_summary.csv")
    if not os.path.exists(summary_path):
        with open(summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode", "player_hand", "dealer_hand",
                "player_value", "dealer_value", "result",
                "reward", "final_bankroll"
            ])

    # Inicializar trazado paso a paso
    trace_path = os.path.join(base_dir, "step_trace.csv")
    if not os.path.exists(trace_path):
        with open(trace_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode", "step", "state", "action",
                "reward", "epsilon", "bankroll"
            ])


def log_step(episode, step, state, action, reward, epsilon, bankroll, base_dir="logs"):
    """
    Registra la información de un paso del agente en el archivo `step_trace.csv`.

    :param episode: Número del episodio actual.
    :param step: Número de paso dentro del episodio.
    :param state: Estado observado por el agente.
    :param action: Acción tomada.
    :param reward: Recompensa recibida.
    :param epsilon: Valor actual de epsilon (exploración).
    :param bankroll: Dinero actual del jugador.
    :param base_dir: Carpeta donde se encuentra el archivo de log.
    :return: None
    """
    trace_path = os.path.join(base_dir, "step_trace.csv")
    with open(trace_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            episode, step, list(state), action,
            reward, epsilon, bankroll
        ])


def log_summary(episode, player_hand, dealer_hand, reward, bankroll, base_dir="logs"):
    """
    Registra el resumen de un episodio completo en `episode_summary.csv`.

    :param episode: Número del episodio.
    :param player_hand: Mano final del jugador (lista de cartas).
    :param dealer_hand: Mano final del dealer.
    :param reward: Recompensa final del episodio.
    :param bankroll: Dinero total del jugador al finalizar el episodio.
    :param base_dir: Carpeta donde se encuentra el archivo de log.
    :return: None
    """

    def hand_val(hand):
        """
        Calcula el valor de una mano, considerando el as como 11 si no hay bust.

        :param hand: Lista de enteros representando cartas.
        :return: Valor entero de la mano.
        """
        return sum(hand) + 10 if (1 in hand and sum(hand) + 10 <= 21) else sum(hand)

    def result_text(player_val, dealer_val, reward):
        """
        Devuelve una etiqueta descriptiva del resultado del episodio.

        :param player_val: Valor final de la mano del jugador.
        :param dealer_val: Valor final de la mano del dealer.
        :param reward: Recompensa final recibida.
        :return: Cadena descriptiva del resultado.
        """
        if reward == -0.5:
            return "surrender"
        if player_val > 21:
            return "player_bust"
        if dealer_val > 21:
            return "dealer_bust"
        if reward == 1:
            return "win"
        elif reward == 0:
            return "draw"
        elif reward == -1:
            return "loss"
        return "unknown"

    # Calcular valores y resultado
    player_val = hand_val(player_hand)
    dealer_val = hand_val(dealer_hand)
    result = result_text(player_val, dealer_val, reward)

    # Escribir resumen del episodio
    summary_path = os.path.join(base_dir, "episode_summary.csv")
    with open(summary_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            episode, player_hand, dealer_hand,
            player_val, dealer_val, result,
            reward, bankroll
        ])
