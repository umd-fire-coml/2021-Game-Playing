from src.Model import Model

def test_model():
  assert(Model.evaluate(Model)[0] == 0.4)
