import nes_py
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import numpy as np
import matplotlib.pyplot as plt
from IPython import display as ipythondisplay
from tqdm import tqdm
from pyvirtualdisplay import Display

class SuperMarioBros_Dataset:

    def __init__(self, level, version):
        # add level selection
        self.env = gym_super_mario_bros.make('SuperMarioBros-{}-{}'.format(level, version))

        self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT)
        self.obs = None
        self.reward = None
        self.done = None
        self.info = None

    def reset_env(self):
        self.env.reset()

    def get_state(self, attribute):
        if attribute == None:
          return self.info
        else:
          return self.info[attribute]

    def get_reward(self):
        return self.reward

    def get_action_space(self):
        return self.env.action_space

    def get_gym_env(self):
        return self.env

    def step(self, action):
        self.obs, self.reward, self.done, self.info = self.env.step(action)

    def simulate_steps(self, step_count):
        display = Display(visible = 0, size = (400, 300))
        display.start()
        action_meanings = self.env.get_action_meanings()
        self.env.reset()

        for i in tqdm(range(step_count)):
          action = 2 # selecting A and right random: env.step(env.action_space.sample())
          obs, reward, done, info = self.env.step(action)
          #print(info)
          print("\nAction: %s" % (action_meanings[action]))
          print("Reward: {:.2f}".format(reward))
          screen = self.env.render(mode = 'rgb_array')

          plt.imshow(screen)
          ipythondisplay.clear_output(wait = True)
          ipythondisplay.display(plt.gcf())

          if done:
              break
        
        ipythondisplay.clear_output(wait = True)