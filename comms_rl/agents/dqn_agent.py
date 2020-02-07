# deep Q network agent
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)   # buffer memory
        self.gamma = 0.9                    # discount factor
        self.epsilon = 1.0                  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.01
        self.model = self._build_model()

    # creating a deep neural Q-network model
    def _build_model(self):
        model = Sequential()
        model.add(Dense(5, input_dim=self.state_size, activation='relu'))
        model.add(Dense(5, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        model.summary()

        return model

    # defining buffer memory
    def add_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # the act function return action value for the state
    def act(self, state, reward, done):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))

            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)



