import gym
import numpy as np
from gym import spaces


class TradingEnvironment(gym.Env):
    def __init__(self, data, initial_balance=1000):
        super(TradingEnvironment, self).__init__()

        self.data = data
        self.current_step = 0
        self.initial_balance = initial_balance
        self.shares_held = 0
        self.total_shares_sold = 0
        self.total_profit  = 0

        self.action_space =  spaces.Discrete(3) # 0: держать, 1: купить, 2: продать
        self.observation_space = spaces.Box(low=np.inf,high=np.inf, shape=(data.shape[1],), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_shares_sold = 0
        self.total_profit = 0
        return self.data[self.current_step]

    def step(self, action):
        self._take_action(action)
        self.current_step += 1

        if self.current_step >= len(self.data) - 1:
            done = True
        else:
            done = False

        reward = self.balance + (self.shares_held * self.data[self.current_step, 0] - self.initial_balance)
        obs = self.data[self.current_step]
        return obs, reward, done

    def _take_action(self, action):
        current_price = self.data[self.current_step, 0]

        if action == 1:
            shares_bought = self.balance // current_price
            self.balance -= shares_bought * current_price
            self.shares_held += shares_bought
        elif action == 2:
            self.balance += self.shares_held * current_price
            self.total_shares_sold += self.shares_held
            self.shares_held = 0

    def render(self, mode='human', close=False):
        profit = self.balance + (self.shares_held * self.data[self.current_step, 0]) - self.initial_balance
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Shares held: {self.shares_held}')
        print(f'Total shares sold: {self.total_shares_sold}')
        print(f'Total profit: {profit}')

