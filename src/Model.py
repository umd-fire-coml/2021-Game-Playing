import tensorflow as tf
import importlib
from DQN_Atari import play
from DQN_Atari import train
from DQN_Atari import test
import numpy as np
import matplotlib.pyplot as plt
from IPython import display as ipythondisplay
from tqdm import tqdm
from pyvirtualdisplay import Display
from DQN_Atari.utils.network import DeepQNetwork
from DQN_Atari.train import get_train_args
from DQN_Atari.test import get_test_args
from DQN_Atari.play import get_play_args
from DQN_Atari.utils.state_buffer import StateBuffer
from DQN_Atari.utils.utils import preprocess_image, reset_env_and_state_buffer
import gym
from gym.wrappers.monitoring import video_recorder

class Model:

  def __init__(self, env, level, version):
    importlib.reload(test)
    importlib.reload(train)
    importlib.reload(play)

    game_id = 'SuperMarioBros-{}-{}'.format(level, version)
    self.train_args = get_train_args(['--env', game_id, '--frame_width', '240', '--frame_height', '256'])
    self.test_args = get_test_args(self.train_args, ['--env', game_id])
    self.play_args = get_play_args(self.train_args, ['--env', game_id])

    self.env = env
    num_actions = env.action_space.n

    # Define input placeholders
    # state_ph = tf.placeholder(tf.uint8, (None, play_args.frame_height,
    #                                       play_args.frame_width,
    #                                       play_args.frames_per_state))

    # Instantiate DQN network
    state_ph = tf.placeholder(tf.uint8, (None, 256, 240, self.play_args.frames_per_state))
    action_ph = tf.placeholder(tf.int32, (None))
    target_ph = tf.placeholder(tf.float32, (None))
    learning_rate_ph = 0.00025

    DQN = DeepQNetwork(num_actions, state_ph, action_ph, target_ph, learning_rate_ph, scope='DQN_main')

    # DQN = DeepQNetwork(num_actions, state_ph, scope='DQN_main')
    self.DQN_predict_op = DQN.predict()

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    self.sess = tf.Session(config=config)

    self.sess.run(tf.global_variables_initializer())
    tf.reset_default_graph()

  def load_ckpt(self):
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth=True
    sess=tf.Session(config=config)
    sess.run(tf.global_variables_initializer())


    loader=tf.train.Saver()
    if self.play_args.ckpt_file is not None:
      ckpt = self.play_args.ckpt_dir + '/' + self.play_args.ckpt_file
    else:
      ckpt = tf.train.latest_checkpoint(self.play_args.ckpt_dir)
    loader.restore(sess, ckpt)
    print('%s restored.\n\n' % ckpt)

  def run_model(self, filename):

    state_buf = StateBuffer(self.play_args)
    screens = []
    rewards = []
    state_ph = tf.placeholder(tf.uint8, (None, 256, 240, self.play_args.frames_per_state))
    action_ph = tf.placeholder(tf.int32, (None))
    target_ph = tf.placeholder(tf.float32, (None))
    learning_rate_ph = 0.00025

    vid = video_recorder.VideoRecorder(self.env, path=filename)

    for ep in range(0, self.play_args.num_eps):
      reset_env_and_state_buffer(self.env, state_buf, self.play_args)
      ep_done = False
      initial_steps = np.random.randint(1, self.play_args.max_initial_random_steps + 1)
      reward = 0
      for step in tqdm(range(self.play_args.max_ep_length)):
        screen = self.env.render(mode = 'rgb_array')
        vid.capture_frame()
        screens.append(screen)
        if step < initial_steps:
          action = self.env.action_space.sample()
        else:
          state = np.expand_dims(state_buf.get_state(), 0)
          action = self.sess.run(self.DQN_predict_op, {state_ph:state})[0]
        #print(action)
        frame, r, ep_terminal, _ = self.env.step(action)
        frame = preprocess_image(frame, 240,
                                256)
        state_buf.add(frame)
        reward += r
        if ep_terminal:
          break
        
        rewards.append(reward)
    print("\nAverage Reward {} +- {}".format(np.mean(rewards), np.std(rewards)))
    vid.close()

    def train(self, ckpt):
      train_args = get_train_args(['--env', 'SuperMarioBros-1-1-v0',
                             '--num_steps_train', '10000',
                             '--save_ckpt_step', '1000',
                             '--ckpt_dir', './ckpts',
                             '--log_dir', './logs/train',
                             '--initial_replay_mem_size', '10000',
                             '--frame_width', '240',
                             '--frame_height', '256',
                             '--batch_size', '16',
                             '--epsilon_step_end', '5000',
                             '--ckpt_file', ckpt])
      tf.reset_default_graph()

      train(train_args)