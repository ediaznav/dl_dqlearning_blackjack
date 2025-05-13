import csv
import os

def init_loggers(base_dir="logs"):
    os.makedirs(base_dir, exist_ok=True)
    # Init summary log
    summary_path = os.path.join(base_dir, "episode_summary.csv")
    if not os.path.exists(summary_path):
        with open(summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode", "player_hand", "dealer_hand", "player_value", "dealer_value",
                "result", "reward", "final_bankroll"
            ])

    # Init step trace log
    trace_path = os.path.join(base_dir, "step_trace.csv")
    if not os.path.exists(trace_path):
        with open(trace_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode", "step", "state", "action", "reward", "epsilon", "bankroll"
            ])

def log_step(episode, step, state, action, reward, epsilon, bankroll, base_dir="logs"):
    trace_path = os.path.join(base_dir, "step_trace.csv")
    with open(trace_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            episode, step, list(state), action, reward, epsilon, bankroll
        ])

def log_summary(episode, player_hand, dealer_hand, reward, bankroll, base_dir="logs"):
    summary_path = os.path.join(base_dir, "episode_summary.csv")

    def hand_val(hand):
        return sum(hand) + 10 if (1 in hand and sum(hand) + 10 <= 21) else sum(hand)

    def result_text(player_val, dealer_val, reward):
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

    player_val = hand_val(player_hand)
    dealer_val = hand_val(dealer_hand)
    result = result_text(player_val, dealer_val, reward)

    with open(summary_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            episode, player_hand, dealer_hand, player_val, dealer_val,
            result, reward, bankroll
        ])
