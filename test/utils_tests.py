from src.utils import Utils

def test_boilerplate():
    preprocessor = Utils()
    assert(preprocessor.get_state() == "environment state")