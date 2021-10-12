from preprocessor_benchmark import preprocessor_benchmark  
import pytest

def test_benchmark():
    assert(isinstance(preprocessor_benchmark(), float))
