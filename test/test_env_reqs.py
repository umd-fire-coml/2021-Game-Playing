import requirements.txt
import test_requirements.txt

def test_tensorflow():
    assert(tensorflow.__version__() == "2.6.0")

def test_numpy():
    array = numpy.array([1,2,3])
    assert(len(array) == 3)