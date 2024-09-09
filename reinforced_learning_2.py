import gym
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from gym import spaces


# Cargar y preparar los datos
data = pd.read_csv("heart_failure_clinical_records.csv")
columnas_a_eliminar = ['DEATH_EVENT', 'anaemia']
X = data.drop(columns=columnas_a_eliminar)
y = data['DEATH_EVENT']


# Dividir los datos en entrenamiento y pruebas
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Escalar datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


class HeartFailureEnv(gym.Env):
    def __init__(self, X, y):
        super(HeartFailureEnv, self).__init__()
        self.X = X
        self.y = y
        self.current_index = 0

        self.action_space = spaces.Discrete(2)  # Dos acciones: predecir 0 o 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(X.shape[1],), dtype=np.float32)

    def reset(self):
        self.current_index = 0
        return self.X[self.current_index]

    def step(self, action):
        correct = (action == self.y[self.current_index])
        reward = 1 if correct else -0.1
        self.current_index += 1

        done = (self.current_index >= len(self.X))
        next_state = self.X[self.current_index] if not done else np.zeros_like(self.X[0])

        return next_state, reward, done, {}


# Crear el entorno
env = HeartFailureEnv(X_train_scaled, y_train.values)

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.08, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.q_table = {}

    def _get_state_key(self, state):
        return tuple(state)

    def choose_action(self, state):
        state_key = self._get_state_key(state)
        if np.random.rand() < self.exploration_rate or state_key not in self.q_table:
            return np.random.choice(self.action_size)
        return np.argmax(self.q_table[state_key])

    def learn(self, state, action, reward, next_state):
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)

        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_size)

        best_next_action = np.argmax(self.q_table[next_state_key])
        td_target = reward + self.discount_factor * self.q_table[next_state_key][best_next_action]
        self.q_table[state_key][action] += self.learning_rate * (td_target - self.q_table[state_key][action])
        self.exploration_rate *= self.exploration_decay



# Inicializar el agente
agent = QLearningAgent(state_size=X_train_scaled.shape[1], action_size=2)

# Entrenar el agente
num_episodes = 2000
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False
    step = 0

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.learn(state, action, reward, next_state)
        state = next_state
        total_reward += reward
        step += 1


    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}: Total Reward: {total_reward}, Steps: {step}")


# Evaluar el agente
correct_predictions = 0
env = HeartFailureEnv(X_test_scaled, y_test.values)
state = env.reset()
done = False

while not done:
    action = agent.choose_action(state)
    next_state, reward, done, _ = env.step(action)
    correct_predictions += (reward == 1)
    state = next_state

accuracy = correct_predictions / len(X_test_scaled)
print("Accuracy del agente en el conjunto de prueba:", accuracy)



# Predicción
def predict_with_agent(agent, state):
    action = agent.choose_action(state)
    return action


sample_state = [82,1,379,0,50,0,47000,1.3,136,1,0,13]
predicted_action = predict_with_agent(agent, sample_state)
print("Predicción de acción:", predicted_action)




