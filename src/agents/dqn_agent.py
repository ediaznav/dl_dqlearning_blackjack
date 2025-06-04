# dqn_agent.py

# En el presente scrip se definen la clase y funciones necesarias para 
# utilizar al agente (DQN) para los diferentes pasos en el proceso:
# training-testing.

# Importamos las librerias a utilizar 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque


class QNetwork(nn.Module):
    """
    Red neuronal simple para estimar la función Q(s, a).
    Utiliza capas densas, normalización, ReLU y dropout.
    """

    def __init__(self, state_dim, action_dim):
        """
        :param state_dim: Dimensión del espacio de estados.
        :param action_dim: Número de acciones posibles.
        """
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        """
        Propagación hacia adelante de la red.

        :param x: Tensor de entrada (estado).
        :return: Tensor de valores Q para cada acción.
        """
        return self.net(x)


class DuelingQNetwork(nn.Module):
    """
    Variante Dueling DQN donde se estiman por separado el valor del estado
    y la ventaja de cada acción.
    """

    def __init__(self, input_dim, output_dim):
        """
        :param input_dim: Dimensión de entrada (estado).
        :param output_dim: Número de acciones posibles.
        """
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
        )
        self.advantage = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        self.value = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        """
        Propagación hacia adelante combinando valor y ventaja.

        :param x: Tensor de entrada (estado).
        :return: Tensor de valores Q para cada acción.
        """
        shared_out = self.shared(x)
        A = self.advantage(shared_out)
        V = self.value(shared_out)
        Q = V + A - A.mean(dim=1, keepdim=True)
        return Q


class DQNAgent:
    """
    Agente de Aprendizaje por Diferencias Temporales (DQN).
    Usa una política epsilon-greedy y una red objetivo fija.
    """

    def __init__(
        self, state_dim, action_dim, lr=1e-3, gamma=0.99,
        epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
        memory_size=10000, batch_size=64, target_update_freq=10
    ):
        """
        :param state_dim: Dimensión del estado.
        :param action_dim: Número de acciones.
        :param lr: Tasa de aprendizaje.
        :param gamma: Factor de descuento.
        :param epsilon_start: Valor inicial de epsilon.
        :param epsilon_end: Valor mínimo de epsilon.
        :param epsilon_decay: Factor de decaimiento de epsilon.
        :param memory_size: Capacidad del buffer de memoria.
        :param batch_size: Tamaño del minibatch de entrenamiento.
        :param target_update_freq: Frecuencia de actualización de la red objetivo.
        """
        self.policy_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = deque(maxlen=memory_size)

        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.update_counter = 0
        self.target_update_freq = target_update_freq

        self.state_dim = state_dim
        self.action_dim = action_dim

    def act(self, state, legal_actions=None):
        """
        Selecciona una acción usando una política epsilon-greedy.

        :param state: Estado actual.
        :param legal_actions: Lista de acciones legales (opcional).
        :return: Acción seleccionada (int).
        """
        if legal_actions is None:
            legal_actions = list(range(self.action_dim))

        if random.random() < self.epsilon:
            return random.choice(legal_actions)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_vals = self.policy_net(state_tensor).squeeze()

        # Penalizar acciones ilegales con un valor bajo
        masked_q_vals = torch.full_like(q_vals, -1e9)
        for a in legal_actions:
            masked_q_vals[a] = q_vals[a]

        return torch.argmax(masked_q_vals).item()

    def remember(self, transition):
        """
        Guarda una transición en la memoria de experiencia.

        :param transition: Tupla (estado, acción, recompensa, siguiente_estado, done).
        :return: None
        """
        self.memory.append(transition)

    def update(self):
        """
        Realiza una actualización de la red Q usando un minibatch
        de transiciones almacenadas en la memoria.

        :return: None
        """
        if len(self.memory) < self.batch_size:
            return

        # Selección aleatoria de muestras
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Conversión a tensores
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones).unsqueeze(1)

        # Calcular Q(s,a) para las acciones tomadas
        q_values = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            # Calcular Q objetivo usando la red objetivo
            next_q = self.target_net(next_states).max(1, keepdim=True)[0]
            q_targets = rewards + self.gamma * next_q * (~dones)

        # Cálculo de la pérdida y paso de optimización
        loss = F.mse_loss(q_values, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Actualizar red objetivo periódicamente
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Aplicar decaimiento a epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
