# 2021-Game-Playing

**Product Description**

  Our model learns to play any level of Super Mario Bros. Its architecture is based off of the DQN research paper's model architecture. More specifically; however, this architecture connects a reinforcement model to a deep neural network. For our RGB input for our model, we took a 256x240 pixel screen capture from our Super Mario Bros emulator and produced a simple action vector of the q values of 7 possible movements for Mario to choose (ex: run right, jump, duck, etc.). Our model then chooses the action with the highest score and turns to the next frame, and continues onward through the level as such. Each action's q score is calculated based on projected reward, where positive factors include getting further into the level, living, and shorter time taken and negative factors include dying or going the wrong way over the x axis; hence, reinforcement learning. Our model then trains on the runs that had the best rewards.
 

**Video Demonstration**

  (*Jonathan to insert link*)


**Colab Notebook**

  https://colab.research.google.com/drive/17guySUGyHN8m4oEF4klH0jBNofHHN1yZ?usp=sharing


**Directory Guide**

 - .github folder: This folder has the python file that runs all the tests for each github commit.
  
 - gym-super-mario-bros folder: This folder has all of the code for the Super Mario Bros environment we used through its emulator.
  
 - src folder: This folder contains all of the source code for building, training, and testing our model as well as setting up the environment.
  
 - src/Model.py: This file contains the code for the Model. The Model has its constructor, where it creates the model with the right environment, level, and version. It also has three functions. The run_model function simulates the model for a certain number of steps. The load_checkpoint function loads a given checkpoint into the model for training purposes. The train function calls the training code on the model, given the correct arguments.
  
 - src/data_utils.py: This file benchmarks how long certain runs of the model might take and returns its time.
  
 - src/experience_replay.py: This file sets up an experience replay for our model, meaning that for each time the model trains it remembers the past several frames of input and learns not just off of one RGB image, but multiple.
  
 - src/state_buffer.py: This file sets up the state buffer, which remembers past states of the model and rewards made off of those decision paths and compares the current decision to such previous outcomes.
  
 - src/utils.py: This file contains the code for the utils file. The utils file contains the environment.
  
 - src/video.py: This file contains the code for giving our model a video output representation of the decisions it chooses to make.
  
 - src/train.py: This file provides the training script code for the model to train it over a certain number of epochs and save checkpoints throughout given the proper arguments.
  
 - src/test.py: This file provides the testing script code for our model that takes its weights and runs some tests and examines the rewards of such actions.
  
 - test folder: This folder contains various tests for each of the files in the src folder, to ensure each file works properly and as intended.
  
 - .gitignore: This file contains a list of commands to ignore.
  
 - requirements.txt: This text file contains a list of the necessary requirements for our repository.
  
 - setup.sh: This setup file installs the proper requirements for our repository.
  
 - test-requirements.txt: This text file contains a list of necessary requirements for testing, which happens just to be pytest.


**Environment Setup Instructions**

  The environment for our environment can be setup through running our shell file setup.sh, or manually by installing the required modules from both requirements.txt and test-requirements.txt. To run manually,

   1. Run the following lines of code in a shell:
      
      - pip install -r requirements.txt
      
      - pip install -r requirements-test.txt


**Dataset Setup Instructions**

  The "dataset" for our repository is just the precreated environment for our Super Mario Bros game. To install this,

   1. Download the Super Mario Bros gym environment with the command:
      pip install gym-super-mario-bros

   2. Include this python line in your file:
      import gym_super_mario_bros
   

**Training Instructions**

  A list of step by step instructions to get the training started and get the trained weights.
  
  1. In your main python file that you will run, make sure all the neccessary files are imported from src 
  
  2. Instantiate a model object and run model.train(). Inside of train will be several parameters that need to be passed in
     as a string. Parameters include:
     
Parameters  | Usage
------------- | -------------
env  | Environment to use (must be formatted as SuperMarioBros-{world}-{level}-{version})
render  | Whether or not to display the environment on the screen during training
random_seed | Random seed for reproducability
frames_per_state | Sequence of frames which constitutes a single state
num_steps_train | Number of steps to train for
train_frequency | Perform training step every N game steps
max_ep_steps | Maximum number of steps per episode
batch_size | Maximum size of replay memory buffer
learning_rate | Model learning rate
replay_mem_size | Maximum size of replay memory buffer
initial_replay_mem_size | Initial size of replay memory (populated by random actions) before learning can start
epsilon_start | Exploration rate at the beginning of training
epsilon_end | Exploration rate at the end of decay
epsilon_step_end | After how many steps to stop decaying the exploration rate
discount_rate | Discount rate (gamma) for future rewards.
update_target_step | Copy current network parameters to target network every N steps
save_ckpt_step | Save checkpoint every N steps
save_log_step | Save logs every N steps
ckpt_dir | Directory for saving/loading checkpoints
ckpt_file | Checkpoint file to load and resume training from (if None, train from scratch)
log_dir | Directory for saving logs

All parameters need to be put in a list. Each parameter should have two elements, with the parameter name preceded by two dashes and then a second element with the parameter value in a string. For example a valid call would be: 

model.train(['--env', 'SuperMarioBros-1-1-v0',
                            '--num_steps_train', '100',
                            '--save_ckpt_step', '1000',
                            '--ckpt_dir', './ckpts',
                            '--log_dir', './logs/train',
                            '--initial_replay_mem_size', '1000',
                            '--batch_size', '16',
                            '--epsilon_step_end', '5000',
                            '--replay_mem_size', '2000'])


**Testing Instructions**

  A list of step by step instructions to test your trained model and get the predicted results.
  
  1. Make sure you have your checkpoint files in your checkpoint directory from training. The default directory would be a folder called "ckpts".
  
  2. In your main python file that you will run, make sure all the neccessary files are imported from src 
  
  3. Instantiate a model object and run model.evaluation(ckpt). The ckpt parameter should be the name of your checkpoint file. For example, a valid call to this 
     would be model.evaluation('model.ckpt-40000')
     
  4. This should provide you with information about your average reward for your model over a series of test runs


https://user-images.githubusercontent.com/17547415/145637570-4f3892bc-12bb-48a6-93f0-f2a28cf597ee.mp4


**References:**

    [1]Christian Kauten. 2018. Super Mario Bros for OpenAI Gym. GitHub. Retrieved from https://github.com/Kautenja/gym-super-mario-bros 
  
    [2]Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin Riedmiller. 2013. Playing Atari with Deep Reinforcement Learning.
