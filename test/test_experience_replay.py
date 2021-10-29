from src.experience_replay import ReplayMemory
import numpy as np
import random
    
#Populate experience buffer
def test_experience_replay():
    mem = ReplayMemory(10000, 500)
    for i in range(0,500):
        frame = np.random.randint(255, size=(240, 256))
        action = np.random.randint(4)
        reward = np.random.randint(2)
        terminal = np.random.choice(a=[False, False, False, False, False, False, False, False, True])
        
        mem.add(action, reward, frame, terminal)
    batch = mem.getMinibatch()
    assert(len(batch) == 5)
    assert(batch[0].shape == (32, 240, 256, 4))
    assert(batch[0][0].shape == (240, 256, 4))
    assert(batch[0][0][0].shape == (256, 4))
    assert(batch[0][0][0][0].shape == (4,))
