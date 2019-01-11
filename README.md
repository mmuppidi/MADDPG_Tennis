# Project 3: Collaboration and Competition

The goal of this project is to train 2 agents to play tennis against each other from Unity ML-Agents toolkit Tennis environment.

![Trained agent](./assets/tennis.gif)

### Environment:

The environment has two agents playing tennis against each other. If the agent misses to hit the ball the other agent gets a point. The goal of each agent is to hit the ball as many times as possible. At every step the agents have access to their own local observation. The size of state space per agent is 24. States observed by the agent are position and velocities of ball and racket.

The actions each agent can take are continuous actions, one to move the racket towards or away from nets and the other to jump. The size of action space for each agent is 2. If an agent hits the ball with racket it recievs '+0.1' reward and receives '-0.01' reward if it misses the ball. The task is episodic, and in order to solve the environment, agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,
    - After each episode, we add up the rewards that each agent received (without discounting), to get a score for each     agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
    - This yields a single score for each episode.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the root directory of this repository, and unzip (or decompress) the file. 

### Training the agent

In order to train the agent, open Tennis.ipynb and run all the cells. 

### Run a trained agent

In order to run a pretrained agent, run all the cells in the Report.ipynb.

####  For Jupyter notebook newbies

To open the Report.ipynb use the following command from the root of this repository

```
jupyter notebook
```

A webpage will be opened on your browser, click on Report.ipynb, which opens a new page. From here you should be able to run a cell or run everything at once.

#### Reference