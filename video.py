from gym.wrappers import Monitor

class Video:

  def generate_video(environment, episodes, model=None):
      environment = Monitor(environment, './video', force=True)
      for ep in range(episodes):
        state = environment.reset()
        done = False
        while not done:
          environment.render(mode = 'rgb_array')
          action = environment.action_space.sample()
          state, reward, done, info = environment.step(action)
