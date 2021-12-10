import pytest
from src.dataset import SuperMarioBros_Dataset
from src.model import Model

def test_model_init():
  env = SuperMarioBros_Dataset("1-1", "v0")
  model = Model(env, "1-1", "v0")

def test_train():
  env = SuperMarioBros_Dataset("1-1", "v0")
  model = Model(env, "1-1", "v0")
  model.train(['--env', 'SuperMarioBros-1-1-v0',
                            '--num_steps_train', '10000',
                            '--save_ckpt_step', '1000',
                            '--ckpt_dir', './ckpts',
                            '--log_dir', './logs/train',
                            '--initial_replay_mem_size', '10000',
                            '--frame_width', '240',
                            '--frame_height', '256',
                            '--batch_size', '16',
                            '--epsilon_step_end', '5000',
                            '--replay_mem_size', '20000'])

def run_model():
  env = SuperMarioBros_Dataset("1-1", "v0")
  model = Model(env, "1-1", "v0")
