# blackjack_env.py

# Se utiliza la base del ambiente de de BlackJack de Gymnasium de 
# OpenAI para definir una versión con 4 acciones: 0-Hit, 1-Stand, 
# 2-Double y 3-Surrender.  


# Importamos las librerias a utilizar 
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import sys


def draw_card():
    """
    Extrae una carta aleatoria del mazo. Las cartas J, Q y K se tratan como 10.

    :return: Valor entero de la carta (1-10).
    """
    card = random.randint(1, 13)
    return min(card, 10)


def draw_hand():
    """
    Extrae una mano inicial de dos cartas.

    :return: Lista de dos enteros representando las cartas.
    """
    return [draw_card(), draw_card()]


def usable_ace(hand):
    """
    Determina si la mano contiene un as que puede usarse como 11 sin pasarse de 21.

    :param hand: Lista de enteros representando cartas.
    :return: True si hay un as usable, False si no.
    """
    return 1 in hand and sum(hand) + 10 <= 21


def hand_value(hand):
    """
    Calcula el valor total de la mano, considerando el as como 11 si es usable.

    :param hand: Lista de enteros representando cartas.
    :return: Valor entero de la mano.
    """
    val = sum(hand)
    return val + 10 if usable_ace(hand) else val


def is_bust(hand):
    """
    Determina si la mano ha superado los 21 puntos.

    :param hand: Lista de enteros representando cartas.
    :return: True si está en quiebra (bust), False si no.
    """
    return hand_value(hand) > 21


def score(hand):
    """
    Calcula el puntaje de la mano, considerando 0 si está en quiebra.

    :param hand: Lista de enteros representando cartas.
    :return: Puntaje entero.
    """
    return 0 if is_bust(hand) else hand_value(hand)


def cmp(a, b):
    """
    Compara dos puntajes y retorna:
    - 1 si a > b
    - 0 si a == b
    - -1 si a < b

    :param a: Puntaje del jugador.
    :param b: Puntaje del dealer.
    :return: -1, 0 o 1
    """
    return int(a > b) - int(a < b)


class BlackjackMultiPlayerEnv(gym.Env):
    """
    Entorno personalizado de Blackjack compatible con Gymnasium.
    Incluye 4 acciones: Hit, Stand, Double, Surrender.
    """

    def __init__(self, initial_bankroll=100):
        """
        Inicializa el entorno con configuración de apuestas y observación.

        :param initial_bankroll: Dinero inicial del jugador.
        """
        self.initial_bankroll = initial_bankroll
        self.bankroll = initial_bankroll
        self.bet = 5
        self.doubled = False
        self.done = False

        # Espacio de acciones: 0=Hit, 1=Stand, 2=Double, 3=Surrender
        self.action_space = spaces.Discrete(4)

        # Observación: total jugador, carta visible del dealer, as usable (0/1)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32),  # Total del jugador
            spaces.Discrete(11),  # Carta del dealer
            spaces.Discrete(2),   # As usable
        ))

    def get_legal_actions(self):
        """
        Devuelve las acciones válidas en el estado actual.

        :return: Lista de acciones legales disponibles.
        """
        legal = [0, 1]  # Hit y Stand siempre válidas
        if len(self.player) == 2:  # Solo en el primer turno
            legal += [2, 3]  # Double y Surrender
        return legal

    def _get_obs(self):
        """
        Devuelve la observación del estado actual del entorno.

        :return: Tupla con (total jugador, carta dealer, as usable).
        """
        return (
            hand_value(self.player),
            self.dealer[0],
            int(usable_ace(self.player))
        )

    def reset(self, seed=None, options=None):
        """
        Reinicia el entorno y devuelve la observación inicial.

        :param seed: Semilla de aleatoriedad (opcional).
        :param options: Opciones adicionales (no utilizadas).
        :return: Observación inicial, diccionario vacío.
        """
        super().reset(seed=seed)
        self.dealer = draw_hand()
        self.player = draw_hand()
        self.doubled = False
        self.done = False
        self.current_bet = self.bet
        return self._get_obs(), {}

    def step(self, action):
        """
        Ejecuta una acción en el entorno.

        :param action: Acción seleccionada (entero 0-3).
        :return: Tupla (observación, recompensa, done, truncated, info).
        """
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
        """
        Finaliza el episodio haciendo que el dealer juegue su turno,
        evalúa el resultado y actualiza el bankroll.

        :return: Tupla (observación, recompensa, done, truncated, info).
        """
        # Dealer roba hasta alcanzar al menos 17
        while hand_value(self.dealer) < 17:
            self.dealer.append(draw_card())

        # Evaluar puntajes
        p_score = score(self.player)
        d_score = score(self.dealer)
        result = cmp(p_score, d_score)

        # Determinar recompensa
        if is_bust(self.player):
            reward = -1.0
        elif is_bust(self.dealer):
            reward = 1.0
        else:
            reward = result  # 1: win, 0: empate, -1: pierde

        # Actualizar bankroll
        self.bankroll += self.current_bet * reward
        return self._get_obs(), reward, True, False, {}
