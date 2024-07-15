This repository ccontains a bunch of notes and code that I have implemented through self-learning. Most of these resources are either through a free course online or through a textbook that matched the content that I'm interested in.

# Contents
## Reinforcement Learning Code (RLCode)
### Dynamic Programming
- [Policy Iteration using iterative policy evaluation on FrozenLake-v1](https://github.com/jasoon-chen/handwrittenNotes/blob/main/RLCode/DynamicProgramming/policyIteration.ipynb)
- [Value Iteration on FrozenLake-v1](https://github.com/jasoon-chen/handwrittenNotes/blob/main/RLCode/DynamicProgramming/valueIteration.ipynb)
### Monte-Carlo Methods
- [First-visit MC Prediciton on FrozenLake-v1](https://github.com/jasoon-chen/handwrittenNotes/blob/main/RLCode/MonteCarlo/firstVisitMonteCarlo.ipynb)
- [Monte Carlo Exploring Starts on FrozenLake-v1](https://github.com/jasoon-chen/handwrittenNotes/blob/main/RLCode/MonteCarlo/onPolicyFirstVisitMonteCarlo.ipynb)
- [REINFORCE: Monte-Carlo Policy-Gradient Control on MountainCar-v0](https://github.com/jasoon-chen/handwrittenNotes/blob/main/RLCode/MonteCarlo/reinforceMonteCarloPolicyGradientControl.ipynb)
- [REINFORCE with Baseline: Monte-Carlo Policy-Gradient Control on MountainCar-v0](https://github.com/jasoon-chen/handwrittenNotes/blob/main/RLCode/MonteCarlo/reinforceMonteCarloPolicyGradientControlwithBaseline.ipynb)
### Temporal Difference
- [Tabular TD(0) on FrozenLake-v1](https://github.com/jasoon-chen/handwrittenNotes/blob/main/RLCode/TemporalDifference/tabularTD0.ipynb)
- [Sarsa on FrozenLake-v1](https://github.com/jasoon-chen/handwrittenNotes/blob/main/RLCode/TemporalDifference/Sarsa.ipynb)
- [Q Learning on FrozenLake-v1](https://github.com/jasoon-chen/handwrittenNotes/blob/main/RLCode/TemporalDifference/qLearning.ipynb)
- [Double Q Learning on FrozenLake-v1](https://github.com/jasoon-chen/handwrittenNotes/blob/main/RLCode/TemporalDifference/doubleQLearning.ipynb)
- [N-Step TD Learning on FrozenLake-v1](https://github.com/jasoon-chen/handwrittenNotes/blob/main/RLCode/TemporalDifference/nStepTDLearning.ipynb)
- [N-Steo Sarsa Learning on FrozenLake-v1](https://github.com/jasoon-chen/handwrittenNotes/blob/main/RLCode/TemporalDifference/nStepSarsa.ipynb)
- [Episodic Semi-Gradient Sarsa on MountainCar-v0](https://github.com/jasoon-chen/handwrittenNotes/blob/main/RLCode/TemporalDifference/episodicSemiGradientSarsa.ipynb)
### Tabular Methods
- [Tabular Dyna-Q on FrozenLake-v1](https://github.com/jasoon-chen/handwrittenNotes/blob/main/RLCode/TabularMethods/tabularDynaQ.ipynb)

FrozenLake is a "game" environment from Gym. This is part of there Toy Text environments, in which they have multiple other environments such as MuJoCo, Atari, Classic Control, etc. There is not a lot of detailed documentation and example, but this is from there [official website](https://www.gymlibrary.dev/environments/toy_text/frozen_lake/).

MountainCar is a "game" environment from Gym. This is part of there Classic Control environments, in which they have multiple other environments such as MuJoCo, Atari, Classic Control, etc. There is not a lot of detailed documentation and example, but this is from there [official website]([https://www.gymlibrary.dev/environments/toy_text/frozen_lake/]).

To run the code on your own computer, you need to install [GymLibrary](https://www.gymlibrary.dev/content/basic_usage/). There really isn't any good tutorial on how to run this except for this one that I found on [here](https://www.youtube.com/watch?v=e3DyCg0fgx0). I'm running all of the code on `conda version 24.5.0` with `python version 3.9.19`.

The psuedo-code for all of the RL Code is from `An introduction to Reinforcement Learning, Sutton and Barto, Second Edition`


