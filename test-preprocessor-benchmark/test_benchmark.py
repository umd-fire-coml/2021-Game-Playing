from preprocessor_benchmark import preprocessor_benchmark  
import pytest

def test_benchmark():
    assert(preprocessor_benchmark() >= 1.22 and preprocessor_benchmark() <= 1.25)
