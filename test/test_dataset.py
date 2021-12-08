from src.dataset import SuperMarioBros_Dataset

def test_env_setup():
    env = SuperMarioBros_Dataset("1-1", "v0")
    env.reset_env()
    env.take_step(1)
    assert(env.get_state('coins') == 0)
    assert(env.get_state('stage') == 1)
    assert(env.get_state('world') == 1)

    env2 = SuperMarioBros_Dataset("2-2", "v0")
    env2.reset_env()
    env2.take_step(1)
    assert(env2.get_state('coins') == 0)
    assert(env2.get_state('stage') == 2)
    assert(env2.get_state('world') == 2)
