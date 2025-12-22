import os
from stable_baselines3 import PPO


# Path to the agent's models
agent_left_name = "left_model.zip"
agent_right_name = "right_model.zip"


def load_right_agent():
    '''
    Loads and returns the right agent model.
    Returns:
        agent: The loaded left agent model.
    '''
    agent = PPO.load(os.path.join(os.path.dirname(__file__), agent_right_name))

    return agent


def load_left_agent():
    '''
    Loads and returns the left agent model.
    Returns:
        agent: The loaded left agent model.
    '''
    agent = PPO.load(os.path.join(os.path.dirname(__file__), agent_left_name))

    return agent
