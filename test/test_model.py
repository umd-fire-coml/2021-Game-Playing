import os
os.system('ls')
from src.model import Model
from src.dataset import SuperMarioBros_Dataset

def test_model_init():
  env = SuperMarioBros_Dataset("1-1", "v0")
  model = Model(env, "1-1", "v0")

def test_load_ckpt():
  env = SuperMarioBros_Dataset("1-1", "v0")
  model = Model(env, "1-1", "v0")
  model.load_ckpt('model.ckpt-4000')

def test_train():
  env = SuperMarioBros_Dataset("1-1", "v0")
  model = Model(env, "1-1", "v0")
  model.train('model.ckpt-4000', '500')

def run_model():
  env = SuperMarioBros_Dataset("1-1", "v0")
  model = Model(env, "1-1", "v0")
