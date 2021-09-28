from src.Model import Model

def test_model():
  assert(Model.evaluate()[0] == 0.4)
