import pytest
from src.dataset import SuperMarioBros_Dataset
from src.model import Model

def test_evaluation():
    env = SuperMarioBros_Dataset("1-1", "v0")
    model = Model(env, "1-1", "v0")
    model.evaluation([])