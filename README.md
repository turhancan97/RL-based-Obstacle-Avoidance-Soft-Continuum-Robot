# Obstacle Avoidance with Reinforcement Learning for a Soft Continuum Robot

# How to use

1. Create the environment from the `environment.yml` file:
    * `conda env create -f environment.yml`
2. Activate the new environment: `conda activate continuum-rl`
3. git clone `https://github.com/turhancan97/RL-based-Obstacle-Avoidance-Soft-Continuum-Robot.git`
4. Run `cd RL-based-Obstacle-Avoidance-Soft-Continuum-Robot/Tests` on the terminal
5. Run `python DDPG_keras_test.py` or `python DDPG_pytorch_test.py` on the terminal

**Note:** Another way is to create your own environment, activate the environment and run `pip install -r requirements.txt`. Then continue from step 3 above.

# About the Project

Please check the main repository for detailed project information on applying the reinforcement learning algorithm to control the soft continuum robot without obstacle avoidance. The main repository can be found [here](https://github.com/turhancan97/RL-based-Control-of-a-Soft-Continuum-Robot)

This repository extends the main project by adding an obstacle avoidance mechanism to the soft continuum robot. The robot is trained with the Deep Deterministic Policy Gradient (DDPG) algorithm to avoid obstacles while reaching the target point.
