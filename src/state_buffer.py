import numpy as np

class StateBuffer:
    def __init__(self, height=240, width=256):
        self.frames_per_state = 4
        self.dims = (height, width)
        self.buffer = np.zeros((height, width, self.frames_per_state), dtype = np.uint8)

    def add(self, frame):
        assert frame.shape == self.dims
        self.buffer[..., :-1] = self.buffer[..., 1:]
        self.buffer[..., -1] = frame
        
    def reset(self):
        self.buffer *= 0
        
    def get_state(self):
        return self.buffer
