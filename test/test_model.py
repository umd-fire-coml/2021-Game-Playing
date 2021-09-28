from src.Model import Model

def test_model_init():
  model = Model(100,50)
  assert(model.height == 100)
  assert(model.width == 50)

def test_model_evaluate():
  model = Model(100,50)
  simple_movement = Model.evaluate(Model, [0,0])
  assert(len(simple_movement) == 7)
  for i in range(7):
    assert(simple_movement[i] >= 0 and simple_movement[i] <= 1)
