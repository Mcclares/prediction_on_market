import gym
import numpy as np
from gym import spaces


class MultiStockTradingEnvironment(gym.Env):
    def __init__(self, data, initial_balance=1000):
        super(MultiStockTradingEnvironment, self).__init__()

        self.data = data
        self.num_stocks = data.shape[1]
        self.current_step = 0
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.shares_held = np.zeros(self.num_stocks)
        self.total_shares_sold = np.zeros(self.num_stocks)
        self.total_profit = 0

        self.action_space = spaces.MultiDiscrete(
            [3] * self.num_stocks)  # 0: держать, 1: купить, 2: продать для каждой акции
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(data.shape[1],), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = np.zeros(self.num_stocks)
        self.total_shares_sold = np.zeros(self.num_stocks)
        self.total_profit = 0
        return self.data[self.current_step]

    def step(self, actions):
        self._take_action(actions)
        self.current_step += 1

        if self.current_step >= len(self.data) - 1:
            done = True
        else:
            done = False

        reward = self.balance + np.sum(self.shares_held * self.data[self.current_step, :]) - self.initial_balance
        obs = self.data[self.current_step]
        return obs, reward, done, {}

    def _take_action(self, actions):
        current_prices = self.data[self.current_step, :]

        for i, action in enumerate(actions):
            if action == 1:  # Buy
                shares_bought = self.balance // current_prices[i]
                self.balance -= shares_bought * current_prices[i]
                self.shares_held[i] += shares_bought

            elif action == 2:  # Sell
                self.balance += self.shares_held[i] * current_prices[i]
                self.total_shares_sold[i] += self.shares_held[i]
                self.shares_held[i] = 0

    def render(self, mode='human', close=False):
        profit = self.balance + np.sum(self.shares_held * self.data[self.current_step, :]) - self.initial_balance
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Shares held: {self.shares_held}')
        print(f'Total shares sold: {self.total_shares_sold}')
        print(f'Total profit: {profit}')