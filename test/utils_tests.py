from src.utils import Utils

def test_boilerplate():
    preprocessor = Utils()
    preprocessor.reset_env()
    preprocessor.take_step()
    assert(preprocessor.get_state() == "environment state")