import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import sys

# Blackjack helpers
def draw_card():
    card = random.randint(1, 13)
    return min(card, 10)

def draw_hand():
    return [draw_card(), draw_card()]

def usable_ace(hand):
    return 1 in hand and sum(hand) + 10 <= 21

def hand_value(hand):
    val = sum(hand)
    return val + 10 if usable_ace(hand) else val

def is_bust(hand):
    return hand_value(hand) > 21

def score(hand):
    return 0 if is_bust(hand) else hand_value(hand)

def cmp(a, b):
    return int(a > b) - int(a < b)


class BlackjackMultiPlayerEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode="human", initial_bankroll=100):
        self.render_mode = render_mode
        self.initial_bankroll = initial_bankroll
        self.bankroll = initial_bankroll
        self.bet = 5
        self.doubled = False
        self.done = False
        self.action_space = spaces.Discrete(4)  # 0: Hit, 1: Stand, 2: Double, 3: Surrender
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32),  # Player total
            spaces.Discrete(11),  # Dealer showing
            spaces.Discrete(2),   # Usable ace
        ))

        # Pygame settings
        self.display = None
        self.width, self.height = 640, 480
        self.bg_color = (0, 100, 0)
        self.font = None


    def get_legal_actions(self):
        legal = [0, 1]  # Hit, Stand always allowed
        if len(self.player) == 2:  # Only first turn
            legal.append(2)  # Double
            legal.append(3)  # Surrender
        return legal


    def _get_obs(self):
        return (hand_value(self.player), self.dealer[0], int(usable_ace(self.player)))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.dealer = draw_hand()
        self.player = draw_hand()
        self.doubled = False
        self.done = False
        self.current_bet = self.bet
        return self._get_obs(), {}

    def step(self, action):
        assert self.action_space.contains(action)
        if self.done:
            return self._get_obs(), 0.0, True, False, {}

        if action == 3:  # Surrender
            self.bankroll -= self.bet / 2
            self.done = True
            return self._get_obs(), -0.5, True, False, {}

        if action == 2:  # Double
            self.player.append(draw_card())
            self.current_bet = self.bet * 2
            self.doubled = True
            self.done = True
            return self._resolve_game()

        if action == 0:  # Hit
            self.player.append(draw_card())
            if is_bust(self.player):
                self.done = True
                self.bankroll -= self.current_bet
                return self._get_obs(), -1.0, True, False, {}
            return self._get_obs(), 0.0, False, False, {}

        if action == 1:  # Stand
            self.done = True
            return self._resolve_game()

    def _resolve_game(self):
        while hand_value(self.dealer) < 17:
            self.dealer.append(draw_card())

        p_score = score(self.player)
        d_score = score(self.dealer)
        result = cmp(p_score, d_score)

        if is_bust(self.player):
            reward = -1.0
        elif is_bust(self.dealer):
            reward = 1.0
        else:
            reward = result

        self.bankroll += self.current_bet * reward
        return self._get_obs(), reward, True, False, {}

   
