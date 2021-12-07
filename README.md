# 2021-Game-Playing

**Product Description**

  Our model learns to play any level of Super Mario Bros. Its architecture is based off of the DQN research paper's model architecture. More specifically; however, this architecture connects a reinforcement model to a deep neural network. For our RGB input for our model, we took a 256x240 pixel screen capture from our Super Mario Bros emulator and produced a simple action vector of the q values of 7 possible movements for Mario to choose (ex: run right, jump, duck, etc.). Our model then chooses the action with the highest score and turns to the next frame, and continues onward through the level as such. Each action's q score is calculated based on projected reward, where positive factors include getting further into the level, living, and shorter time taken and negative factors include dying or going the wrong way over the x axis; hence, reinforcement learning. Our model then trains on the runs that had the best rewards.
 

**Video Demonstration**

  (*Jonathan to insert link*)


**Colab Notebook**

  https://colab.research.google.com/drive/17guySUGyHN8m4oEF4klH0jBNofHHN1yZ?usp=sharing


**Directory Guide**

  .github folder: This folder has the python file that runs all the tests for each github commit.
  
  gym-super-mario-bros folder: This folder has all of the code for the Super Mario Bros environment we used through its emulator.
  
  src folder: This folder contains all of the source code for building, training, and testing our model as well as setting up the environment.
  
  src/Model.py: This file contains the code for the Model. The Model has its constructor, where it creates the model with the right environment, level, and version. It also has three functions. The run_model function simulates the model for a certain number of steps. The load_checkpoint function loads a given checkpoint into the model for training purposes. The train function calls the training code on the model, given the correct arguments.
  
  src/data_utils.py: This file benchmarks how long certain runs of the model might take and returns its time.
  
  src/experience_replay.py: This file sets up an experience replay for our model, meaning that for each time the model trains it remembers the past several frames of input and learns not just off of one RGB image, but multiple.
  
  src/state_buffer.py: This file sets up the state buffer, which remembers past states of the model and rewards made off of those decision paths and compares the current decision to such previous outcomes.
  
  src/utils.py: This file contains the code for the utils file. The utils file contains the environment.
  
  src/video.py: This file contains the code for giving our model a video output representation of the decisions it chooses to make.
  
  src/train.py: This file provides the training script code for the model to train it over a certain number of epochs and save checkpoints throughout given the proper arguments.
  
  src/test.py: This file provides the testing script code for our model that takes its weights and runs some tests and examines the rewards of such actions.
  
  test folder: This folder contains various tests for each of the files in the src folder, to ensure each file works properly and as intended.
  
  .gitignore: This file contains a list of commands to ignore.
  
  requirements.txt: This text file contains a list of the necessary requirements for our repository.
  
  setup.sh: This setup file installs the proper requirements for our repository.
  
  test-requirements.txt: This text file contains a list of necessary requirements for testing, which happens just to be pytest.


**Environment Setup Instructions**

  A list of step by step instructions to install the environment.


**Dataset Setup Instructions**

  The "dataset" for our repository is just the precreated environment for our Super Mario Bros game. To install this,

  1. Download the Super Mario Bros gym environment with the command:
    pip install gym-super-mario-bros

  2. Include this python line in your file:
    import gym_super_mario_bros
   

**Training Instructions**

  A list of step by step instructions to get the training started and get the trained weights.


**Testing Instructions**

  A list of step by step instructions to test the model and get the predicted results.


**References:**

    [1]Christian Kauten. 2018. Super Mario Bros for OpenAI Gym. GitHub. Retrieved from https://github.com/Kautenja/gym-super-mario-bros 
  
    [2]Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin Riedmiller. 2013. Playing Atari with Deep Reinforcement Learning.
