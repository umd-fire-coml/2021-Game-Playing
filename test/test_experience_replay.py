from src.experience_replay import ReplayMemory
import numpy as np
import random
    
#Populate experience buffer
def test_experience_replay():
    mem = ReplayMemory()
    for i in range(0,256):
        frame = np.random.randint(255, size=(240, 256))
        action = np.random.randint(4)
        reward = np.random.randint(2)
        terminal = np.random.choice(a=[False, False, False, False, False, False, False, False, True])
        
        assert(mem.add(action, reward, frame, terminal))
    assert(mem.getMinibatch())
