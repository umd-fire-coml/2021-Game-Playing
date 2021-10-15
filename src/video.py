from gym.wrappers.monitoring import video_recorder

class Video:

  def generate_video(environment, episodes, filename, model=None):
      vid = video_recorder.VideoRecorder(environment, path=filename)
      for ep in range(episodes):
        state = environment.reset()
        done = False
        while not done:
          environment.render(mode = 'rgb_array')
          vid.capture_frame()
          action = environment.action_space.sample()
          state, reward, done, info = environment.step(action)
      environment.close()
      vid.close()
