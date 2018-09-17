# Deep Q-Network (DQN) Reinforcement Learning using PyTorch and Unity ML-Agents
A simple example of how to implement vector based DQN using PyTorch and Unity ML-Agents applications.

The repository includes the following files:
- dqn_agent.py : dqn-agent implementation
- replay_memory.py : dqn-agent's replay buffer implementation
- model.py ; example PyTorch neural network for vector based DQN learning
- train.py : initializes and implements the training processes for a DQN-agent.
- test.py : testes a trained DQN-agent

The repository also includes Mac/Linux/Windows versions of a simple Unity environment, *Banana*, for testing.
This Unity application and testing environment was developed using ML-Agents Beta v0.4


## Example Unity Environment - Banana's
The example uses a modified version of the Unity ML-Agents Banana Collection Example Environment.
The environment includes a single agent, who can turn left or right and move forward or backward.
The agent's task is to collect yellow bananas (reward of +1) that are scattered around a square
game area, while avoiding purple bananas (reward of -1). For the version of Bananas employed here,
the environment is considered solved when the average score over the last 100 episodes > 13. 

### Action Space
The simulation contains a single agent that navigates a large environment.  
At each time step, it can perform four possible actions:
- `0` - walk forward 
- `1` - walk backward
- `2` - turn left
- `3` - turn right

### State Spaces 
The agent is trained from vector input (not pixel input)
The state space has `37` dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  A reward of `+1` is provided for collecting a yellow banana, and a reward of `-1` is provided for collecting a purple banana. 

## Installation and Dependencies
1. Anaconda Python 3.6: Download and installation instructions here: https://www.anaconda.com/download/

2. Create (and activate) a new conda, virtual environment with Python 3.6. For example:
	- Linux or Mac:
	
		`conda create --name yourenvironamehere python=3.6`
	
		`source activate yourenvironamehere`

	- Windows:
	
		`conda create --name _yourenvironamehere_ python=3.6`
	
		`activate yourenvironamehere`

3. Download or clone this repository.

4. To install required dependencies (torch, ML-Agents trainers (v.4), etc...)
	- Change to the *yourpath/BananaNavigationProject/python/* subdirectory and run from the command line:
	
		`pip3 install .`

## Training
 - active the conda environment you created above
 - change the directory to the *yourpath/BananaNavigationProject* directory.
 - open *train.py*, find STEP 2 (lines 55 to 65) and set the relevant version of Banana to match your operating system.
 - run the following command
 	
	`python train.py`
	
 - training will complete once the agent reaches *solved_score* in train.py.
 - after training a *dqnAgent_Trained_Model_datetime.path* file will be saved with the trained model weights
 - a *dqnAgent_scores_datetime.csv* file will also be saved with the scores received during training. You can use this file to plot or assess training performance (see below figure).
 - It is recommended that you train multiple agents and test different hyperparameter settings in train.py and dqn_agent.py.
 - For more information about the DQN training algorithm and the training hyperparameters see the included Report.pdf file.

 ![Example of agent performance (score) as a function of training episodes](media/exampleTrainingScoresGraph.jpg)


## Testing
- active the conda environment you created above
 - change the directory to the *yourpath/BananaNavigationProject* directory.
 - run the following command
 
 	`python test.py`
	
 - An example model weights file is included in the repository (*dqnAgent_Trained_Model.pth*).
 - A different model weights file can be tested by changing the model file name defined in *test.py* on line 109.
