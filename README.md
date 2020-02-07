## Overview
This Open-AI Gym compliant environment simulates a OFDM resource allocation task, where a limited number of frequency
resources are to be allocated to a large number of User Equipments (UEs) over time.
An agent interacting with this Reinforcement Learning (RL) environment plays the role of the MAC scheduler. On each time
step, the agent must allocate one frequency resource to one of a large number of UEs. The agent gets rewarded for these
resource allocation decisions. The reward increases with the number of UEs, whose traffic requirements are satisfied.
The traffic requirements for each UE are expressed in terms of their Guaranteed Bit Rate (if any) and their Packet
Delay Budget (PDP).

## Challenge
You are invited to develop a new agent that interacts with this environment and takes effective resource allocation
decisions. Such agent must use a Reinforcement Learning algorithm and improve its performance through interaction with
the environment.
Four sample agents are provided for reference in the *comms_rl/agents* folder:
-Random agent
-Round Robin agent
-Round Robin IfTraffic agent
-Proportional Fair agent

## Evaluation
The script *scripts/launch_agent.py* runs 16 episodes with a maximum of 65536 time steps each, and collects the reward
obtained by the agent on each time step. The result is calculated as the average reward obtained in all time steps on
all episodes.
The performance obtained by the default agents on the default environment configuration is:

Random                       -69590
Round Robin                  -69638
Round Robin IfTraffic        -3284
Proportional Fair            -9595

Note that the above average rewards are negative values. The best performing agent is thus the Round Robin IfTraffic.

## How to run it
The script *scripts/launch_agent.py* is used to run the interaction between the agent and the environment for a
customizable number of episodes.
The code has been tested to work on Python 3.7 under Windows 10.

## References
1. [Open AI Gym Documentation](http://gym.openai.com/docs/)
2. [How to create new environments for Gym](https://github.com/openai/gym/blob/master/docs/creating-environments.md)
3. [Sacred Documentation](https://sacred.readthedocs.io/en/stable/index.html)
