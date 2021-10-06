from src.utils import SuperMarioBros_Dataset

def test_env_setup():
    preprocessor = SuperMarioBros_Dataset("1-1", "v0")
    preprocessor.reset_env()
    preprocessor.take_step(1)
    assert(preprocessor.get_state('coins') == 0)
    assert(preprocessor.get_state('stage') == 1)
    assert(preprocessor.get_state('world') == 1)

    preprocessor2 = SuperMarioBros_Dataset("2-2", "v0")
    preprocessor2.reset_env()
    preprocessor2.take_step(1)
    assert(preprocessor2.get_state('coins') == 0)
    assert(preprocessor2.get_state('stage') == 2)
    assert(preprocessor2.get_state('world') == 2)
