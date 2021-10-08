import pytest

def test_tensorflow():
    assert(tensorflow.__version__ == "2.6.0")

def test_numpy():
    print(numpy.__version__)
    assert(numpy.__version__ == "?")

def test_matplotlib():
    print(matplotlib.__version__)
    assert(matplotlib.__version__ == ">")

def test_tensorflow():
    print(tqdm.__version__)
    assert(tqdm.__version__ == "?")

def test_numpy():
    print(pyvirtualdisplay.__version__)
    assert(pyvirtualdisplay.__version__ == "?")

