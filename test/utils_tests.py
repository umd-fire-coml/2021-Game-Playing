from src.utils import Utils

def test_boilerplate():
    preprocessor = Utils.__init__
    preprocessor.reset_env()
    preprocessor.take_step()
    assert(preprocessor.get_state() == "environment state")