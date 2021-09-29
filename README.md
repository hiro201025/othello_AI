## othello_AI
  
  
1. Code Language and library
  - coding by python
  - machine learning using pytorch
  
  
2. Contents
  - surprising learning policy network (SLpn), reinforcement learning (RLpn) by DQN, Value network and MonteCarlo tree search (MCTS)
  - othello enviroment made by self
  
  
3. Description of the models
  - othello AI refer to Alpha Go 
  - these codes refer to many web sites.
  - SLpn: learning by 6 millions othello states in online
  - RLpn: learning by DQN based SLpn parameters
  - Value network: learning by RLpn playing data
  - MCTS: using SLpn and Value network to search deeper and faster 

4. Results
  - MCTS is as strong as SLpn.
  - I think RLpn is not good.
  - my PC don't have performance enough to train RLpn. (AlphaGo: 176 GPU, 48 TPU)
