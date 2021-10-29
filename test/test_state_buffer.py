from src.state_buffer import StateBuffer
import numpy as np

def test_state_buffer():
    buf = StateBuffer()
    
    #Populate experience buffer
    for i in range(0,9):
        frame = np.random.randint(255, size=(240, 256))
        action = np.random.randint(4)
        reward = np.random.randint(2)
        terminal = np.random.choice(a=[False, False, False, False, False, False, False, False, True])
    
        buf.add(frame)
        
        state = buf.get_state()

        if i == 5:
            buf.reset()
    assert(state.shape == (240, 256, 4))
