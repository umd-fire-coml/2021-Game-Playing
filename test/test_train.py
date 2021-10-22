import os

from src.Model import Model
from src.train import Train

def test_train_init():
  model = Model(100,50)
  trainer = Train(env='SuperMarioBros-1-1-v0')
  assert(trainer.env == 'SuperMarioBros-1-1-v0')
  assert(trainer.frame_width == 240)
  assert(trainer.frame_height == 256)

def test_train_train():
  model = Model(100,50)
  trainer = Train(env='SuperMarioBros-1-1-v0')
  trainer.train()
  assert(os.path.exists('./ckpts'))

  