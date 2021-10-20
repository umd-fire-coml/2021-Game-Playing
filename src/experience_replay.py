import numpy as np
import random

class ReplayMemory:
    def __init__(self, buffer=1000000, min_buffer=50000):
        self.buffer_size = buffer
        self.min_buffer_size = min_buffer
        # preallocate memory
        self.actions = np.empty(self.buffer_size, dtype = np.uint8)
        self.rewards = np.empty(self.buffer_size, dtype = np.integer)
        self.frames = np.empty((240, 256, self.buffer_size), dtype = np.uint8)
        self.terminals = np.empty(self.buffer_size, dtype = np.bool)
        self.frames_per_state = 4
        self.dims = (240, 256)
        self.batch_size = 32
        self.count = 0
        self.current = 0
        
        self.states = np.empty((self.batch_size, 240, 256, self.frames_per_state), dtype = np.uint8)
        self.next_states = np.empty((self.batch_size, 240, 256, self.frames_per_state), dtype = np.uint8)
    

    def add(self, action, reward, frame, terminal):
        assert frame.shape == self.dims
        # NB! frame is post-state, after action and reward
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.frames[..., self.current] = frame
        self.terminals[self.current] = terminal
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.buffer_size
        
  
    def getState(self, index):
        # Takes the frame at position 'index' and returns a state consisting of this frame and the previous 3 frames.
        return self.frames[..., (index - (self.frames_per_state - 1)):(index + 1)]
         

    def getMinibatch(self):
        # memory must include next_state, current state and (frames_per_state-1) previous states 
        assert self.count > self.frames_per_state, "Replay memory must contain more frames than the desired number of frames per state"
        # memory should be initially populated with random actions up to 'min_buffer_size'
        print(self.count, self.min_buffer_size, self.frames_per_state)
        assert self.count >= self.min_buffer_size, "Replay memory does not contain enough samples to start learning, take random actions to populate replay memory"
        
        # sample random indexes
        indexes = []
        # do until we have a full batch of states
        while len(indexes) < self.batch_size:
            # find random index 
            while True:
                # sample one index
                index = random.randint(self.frames_per_state, self.count - 1)
                # check index is ok
                # if wraps over current pointer, then get new one (as subsequent samples from current pointer position will not be from same episode)
                if index >= self.current and index - self.frames_per_state < self.current:
                    continue
                # if wraps over episode end (terminal state), then get new one (note that last frame can be terminal)
                if self.terminals[(index - self.frames_per_state):index].any():
                    continue
                # index is ok to use
                break
            
            # Populate states and next_states with selected state and next_state (consisting of a 4 frame sequence)
            # NB! having index first is fastest in C-order matrices
            self.states[len(indexes), ...] = self.getState(index - 1)
            self.next_states[len(indexes), ...] = self.getState(index)
            indexes.append(index)
        
        actions = self.actions[indexes]
        rewards = self.rewards[indexes]
        terminals = self.terminals[indexes]
        return self.states, actions, rewards, self.next_states, terminals
