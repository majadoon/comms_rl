import gym
import numpy as np
import json

from sacred import Experiment
from sacred.observers import MongoObserver

from comms_rl.agents.dqn_agent import *
from comms_rl.agents.random_agent import RandomAgent
from comms_rl.agents.round_robin_agent import *
from comms_rl.agents.proportional_fair import *
path_abs = "../../"

# Load agent parameters
with open(path_abs + 'config/config_agent.json') as f:
    ac = json.load(f)

# Configure experiment
with open(path_abs + 'config/config_sacred.json') as f:
    sc = json.load(f)   # Sacred Configuration
    ns = sc["sacred"]["n_metrics_points"]  # Number of points per episode to log in Sacred
    ex = Experiment(ac["agent"]["agent_type"])
    ex.add_package_dependency("comms_rl", "0.1")
    ex.add_config(sc)
    ex.add_config(ac)
#mongo_db_url = f'mongodb://{sc["sacred"]["sacred_user"]}:{sc["sacred"]["sacred_pwd"]}@' +\
#               f'{sc["sacred"]["sacred_host"]}:{sc["sacred"]["sacred_port"]}/{sc["sacred"]["sacred_db"]}'
# ex.observers.append(MongoObserver(url=mongo_db_url, db_name=sc["sacred"]["sacred_db"]))  # Uncomment to save to DB

# Load environment parameters
with open(path_abs + 'config/config_environment.json') as f:
    ec = json.load(f)
    ex.add_config(ec)


@ex.automain
def main(_run):
    n_eps = _run.config["agent"]["n_episodes"]
    t_max = _run.config['agent']['t_max']
    n_sf = t_max//_run.config['env']['n_prbs']  # Number of complete subframes to run per episode
    log_period_t = max(1, (n_sf//ns)*_run.config['env']['n_prbs'])  # Only log rwd on last step of each subframe

    rwd = np.zeros((n_eps, t_max))  # Memory allocation

    # Simulate
    for ep in range(n_eps):  # Run episodes
        env = gym.make('CommsRLTimeFreqResourceAllocation-v0', n_ues=_run.config['env']['n_ues'],
                       n_prbs=_run.config['env']['n_prbs'], buffer_max_size=_run.config['env']['buffer_max_size'],
                       eirp_dbm=_run.config['env']['eirp_dbm'], f_carrier_mhz=_run.config['env']['f_carrier_mhz'],
                       max_pkt_size_bits=_run.config['env']['max_pkt_size_bits'],
                       it=_run.config['env']['non_gbr_traffic_mean_interarrival_time_ttis'])  # Init environment
        env.seed(seed=_run.config['seed'] + ep)

        # Init agent
        if ac["agent"]["agent_type"] == "random":
            agent = RandomAgent(env.action_space)
            agent.seed(seed=_run.config['seed'] + ep)
        elif ac["agent"]["agent_type"] == "round robin":
            agent = RoundRobinAgent(env.action_space, env.K, env.L)
        elif ac["agent"]["agent_type"] == "round robin iftraffic":
            agent = RoundRobinIfTrafficAgent(env.action_space, env.K, env.L)
        elif ac["agent"]["agent_type"] == "proportional fair":
            agent = ProportionalFairAgent(env.action_space, env.K, env.L)
        elif ac["agent"]["agent_type"] == "DQN":
            agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
        else:
            raise NotImplemented

        reward = 0
        done = False
        batch_size = _run.config["agent"]["batch_size"]
        state = env.reset()
        state = np.reshape(state, [1, env.observation_space.shape[0]])
        print(np.shape(state))
        for t in range(t_max):  # Run one episode
            # Collect progress
            #if t_max < ns or (t > 0 and (t+1) % log_period_t == 0):  # If it's time to log
            #    s = np.reshape(state[env.K:env.K * (1 + env.L)], (env.K, env.L))
            #    qi_ohe = np.reshape(state[env.K+2*env.K*env.L:5*env.K + 2*env.K*env.L], (env.K, 4))
            #    qi = [np.where(r == 1)[0][0] for r in qi_ohe]  # Decode One-Hot-Encoded QIs
            #    for u in range(0, env.K, env.K//2):  # Log KPIs for some UEs
            #        _run.log_scalar(f"Episode {ep}. UE {u}. CQI vs time step", state[u], t)
            #        _run.log_scalar(f"Episode {ep}. UE {u}. Buffer occupancy [bits] vs time step", np.sum(s[u, :]), t)
            #        _run.log_scalar(f"Episode {ep}. UE {u}. QoS Identifier vs time step", qi[u], t)

            action = agent.act(state, reward, done)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, env.observation_space.shape[0]])
            agent.add_memory(state, action, reward, next_state, done)
            state = next_state

            # Collect progress
            if t_max < ns or (t > 0 and (t+1) % log_period_t == 0):
                _run.log_scalar(f"Episode {ep}. Rwd vs time step", reward, t)

            rwd[ep, t] = reward
            if done:
                break

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

            if (ep*t_max + t) % log_period_t == 0:
                print(f"{(ep*t_max + t)*100/(n_eps*t_max):3.0f}% completed.")
        env.close()

    if n_eps > 1:
        rwd_avg = np.mean(rwd, axis=0)
        for t in range(t_max):
            if t_max < ns or (t > 0 and (t+1) % log_period_t == 0):  # If it's time to log
                _run.log_scalar(f"Mean rwd vs time step", rwd_avg[t], t)

    result = np.mean(rwd)  # Save experiment result
    print(f"Result: {result}")
    return result
