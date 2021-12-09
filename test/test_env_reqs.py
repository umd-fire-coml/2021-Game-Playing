import pytest
import tensorflow
import numpy
import matplotlib

def test_tensorflow():
    assert(tensorflow.__version__ == "1.15.0")

def test_numpy():
    assert(numpy.__version__ == "1.19.5")

def test_matplotlib():
    assert(matplotlib.__version__ == "3.4.3")
