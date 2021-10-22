'''
## Train ##
# Adapted from code to train Deep Q Network on OpenAI Gym environments
@author: Mark Sinton (msinto93@gmail.com) 
'''

import os
import sys
import argparse
import gym_super_mario_bros
import tensorflow as tf
import numpy as np
import time
import random
import utils
import ReplayMemory # doesn't exist ---------------------------------------------------------
import StateBuffer
import Model
    
class Train():

    def __init__(env='SuperMarioBros-1-1-v0', render=False, random_seed=1234, frame_width=240, frame_height=256, 
                 frames_per_state=4, num_steps_train = 50000000, train_frequency=4, max_ep_steps=2000, batch_size=32,
                 learning_rate=.00025, replay_mem_size=1000000, intitial_replay_mem_size=50000, epsilon_start=1.0,
                 epsilon_end=0.1, epsilon_step_end=1000000, discount_rate=0.99, update_target_step=10000, 
                 save_ckpt_step=250000, save_log_step=1000, ckpt_dir='./ckpts', ckpt_file=None, log_dir='./logs/train'):
        self.env = env
        self.render = render
        self.random_seed = random_seed
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frames_per_state = frames_per_state
        self.num_steps_train = num_steps_train
        self.train_frequency = train_frequency
        self.max_ep_steps = max_ep_steps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.replay_mem_size = replay_mem_size
        self.initial_replay_mem_size = intitial_replay_mem_size
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_step_end = epsilon_step_end
        self.discount_rate = discount_rate
        self.update_target_step = update_target_step
        self.save_ckpt_step = save_ckpt_step
        self.save_log_step = save_log_step
        self.ckpt_dir = ckpt_dir
        self.ckpt_file = ckpt_file
        self.log_dir = log_dir
         
    def train(self):
        
        # Function to return exploration rate based on current step
        def exploration_rate(current_step, exp_rate_start, exp_rate_end, exp_step_end):
            if current_step < exp_step_end:
                exploration_rate = current_step * ((exp_rate_end-exp_rate_start)/(float(exp_step_end))) + 1
            else:
                exploration_rate = exp_rate_end
                
            return exploration_rate
        
        # Function to update target network parameters with main network parameters
        def update_target_network(from_scope, to_scope):    
            from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)    
            to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)
        
            op_holder = []
            
            # Update old network parameters with new network parameters
            for from_var,to_var in zip(from_vars,to_vars):
                op_holder.append(to_var.assign(from_var))
            
            return op_holder
        
        
        # Create environment
        env = gym_super_mario_bros.make(self.env)
        num_actions = 7
        
        # Initialise replay memory and state buffer
        replay_mem = ReplayMemory(self) # don't exist yet ----------------------------------
        state_buf = StateBuffer(self)
        
        # Define input placeholders    
        state_ph = tf.placeholder(tf.uint8, (None, self.frame_height, self.frame_width, self.frames_per_state))
        action_ph = tf.placeholder(tf.int32, (None))
        target_ph = tf.placeholder(tf.float32, (None))
        
        # Instantiate DQN network
        #DQN = DeepQNetwork(num_actions, state_ph, action_ph, target_ph, self.learning_rate, scope='DQN_main')   # Note: One scope cannot be the prefix of another scope (e.g. cannot name this scope 'DQN' and   
                                                                                                                # target network scope 'DQN_target', as a search for vars in 'DQN' scope will return both networks' vars)
        DQN = Model(240, 256)
        DQN_predict_op = DQN.predict() #talk to rithvik --------------------------------------------------
        DQN_train_step_op = DQN.train_step()
        
        # Instantiate DQN target network
        #DQN_target = DeepQNetwork(num_actions, state_ph, scope='DQN_target')
        
        #update_target_op = update_target_network('DQN_main', 'DQN_target')
            
        # Create session
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)       
            
        # Add summaries for Tensorboard visualisation
        tf.summary.scalar('Loss', DQN.loss)  
        reward_var = tf.Variable(0.0, trainable=False)
        tf.summary.scalar("Episode Reward", reward_var)
        epsilon_var = tf.Variable(self.epsilon_start, trainable=False)
        tf.summary.scalar("Exploration Rate", epsilon_var)
        summary_op = tf.summary.merge_all() 
            
        # Define saver for saving model ckpts
        model_name = 'model.ckpt'
        checkpoint_path = os.path.join(self.ckpt_dir, model_name)        
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        saver = tf.train.Saver(max_to_keep=201)  
        
        # Create summary writer to write summaries to disk
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)
        
        # Load ckpt file if given
        if self.ckpt_file is not None:
            loader = tf.train.Saver()   #Restore all variables from ckpt
            ckpt = self.ckpt_dir + '/' + self.ckpt_file
            ckpt_split = ckpt.split('-')
            step_str = ckpt_split[-1]
            start_step = int(step_str)    
            loader.restore(sess, ckpt)
        else:
            start_step = 0
            sess.run(tf.global_variables_initializer())
            sess.run(update_target_op)

            
        ## Begin training 
                        
        env.reset()
        
        ep_steps = 0
        episode_reward = 0
        episode_rewards = []
        duration_values = []

        # Initially populate replay memory by taking random actions       
        
        for random_step in range(1, self.initial_replay_mem_size+1):
            
            if self.render:
                env.render()
            else:
                env.render(mode='rgb_array')
            
            action = env.action_space.sample() #get an action ------------------------------------------------------
            frame, reward, terminal, _ = env.step(action)
            frame = preprocess_image(frame, self.frame_width, self.frame_height) #should be function from utils ---------------
            replay_mem.add(action, reward, frame, terminal)
            
            if terminal:
                env.reset()
                            
            #sys.stdout.write('\x1b[2K\rStep {:d}/{:d}'.format(random_step, self.initial_replay_mem_size))
            #sys.stdout.flush() 
        
        # Begin training process         
        reset_env_and_state_buffer(env, state_buf, None) #should be function from utils ----------------------------- REPLACE NONE LATER
        #sys.stdout.write('\n\nTraining...\n\n')   
        #sys.stdout.flush()
        
        for train_step in range(start_step+1, self.num_steps_train+1):      
            start_time = time.time()  
            # Run 'train_frequency' iterations in the game for every training step       
            for _ in range(0, self.train_frequency):
                ep_steps += 1
                
                if self.render:
                    env.render()
                else:
                    env.render(mode='rgb_array')
                
                # Use an epsilon-greedy policy to select action
                epsilon = exploration_rate(train_step, self.epsilon_start, self.epsilon_end, self.epsilon_step_end)
                if random.random() < epsilon:
                    #Choose random action
                    action = env.action_space.sample()
                else:
                    #Choose action with highest Q-value according to network's current policy
                    current_state = np.expand_dims(state_buf.get_state(), 0)
                    action = sess.run(DQN_predict_op, {state_ph:current_state})
                    
                # Take action and store experience
                frame, reward, terminal, _ = env.step(action)
                frame = preprocess_image(frame, self.frame_width, self.frame_height) # again utils -------------------
                state_buf.add(frame)
                replay_mem.add(action, reward, frame, terminal) 
                episode_reward += reward     
                
                if terminal or ep_steps == self.max_ep_steps:  
                    # Collect total reward of episode              
                    episode_rewards.append(episode_reward)
                    # Reset episode reward and episode steps counters
                    episode_reward = 0
                    ep_steps = 0
                    # Reset environment and state buffer for next episode
                    reset_env_and_state_buffer(env, state_buf, self)          #utilsssss   --------------------   
            
            ## Training step    
            # Get minibatch from replay mem
            states_batch, actions_batch, rewards_batch, next_states_batch, terminals_batch = replay_mem.getMinibatch()
            # Calculate target by passing next states through the target network and finding max future Q
            #future_Q = sess.run(DQN_target.output, {state_ph:next_states_batch})
            max_future_Q = np.max(future_Q, axis=1) #actually this one i don't know if should be utils or not ------------------
            # Q values of the terminal states is 0 by definition
            max_future_Q[terminals_batch] = 0
            targets = rewards_batch + (max_future_Q*self.discount_rate)
            
            # Execute training step
            if train_step % self.save_log_step == 0:
                # Train and save logs
                average_reward = sum(episode_rewards)/len(episode_rewards)
                summary_str, _ = sess.run([summary_op, DQN_train_step_op], {state_ph:states_batch, action_ph:actions_batch, target_ph:targets, reward_var: average_reward, epsilon_var: epsilon})
                summary_writer.add_summary(summary_str, train_step)
                # Reset rewards buffer
                episode_rewards = []
            else:
                # Just train
                _ = sess.run(DQN_train_step_op, {state_ph:states_batch, action_ph:actions_batch, target_ph:targets})
            
            # Update target networks    
            if train_step % self.update_target_step == 0:
                sess.run(update_target_op) # i'm not sure where this comes from--------------------------
            
            # Calculate time per step and display progress to console   
            duration = time.time() - start_time
            duration_values.append(duration)
            ave_duration = sum(duration_values)/float(len(duration_values))
            
            #sys.stdout.write('\x1b[2K\rStep {:d}/{:d} \t ({:.3f} s/step)'.format(train_step, self.num_steps_train, ave_duration))
            #sys.stdout.flush()       
            
            # Save checkpoint       
            if train_step % self.save_ckpt_step == 0:
                saver.save(sess, checkpoint_path, global_step=train_step)
                sys.stdout.write('\n Checkpoint saved\n')   
                sys.stdout.flush() 
                
                # Reset time calculation
                duration_values = []
                       