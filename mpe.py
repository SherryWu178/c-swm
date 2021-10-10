from os.path import join, exists
import random
import numpy as np
from pettingzoo.mpe import simple_adversary_v2
import utils

def generate_data(episode_count=1000, data_dir="data/mpe", agents=2, episode_length=1000):
    assert exists(data_dir), f"{data_dir} does not exist. Create a new directory {data_dir}."
    env = simple_adversary_v2.env(N=agents, max_cycles=episode_length - 1, continuous_actions=False)
    env.reset()
    
    agent_idx = {}
    for i, agent in enumerate(env.agents):
        agent_idx[agent] = i
    
    replay_buffer = []
    
    for i in range(episode_count):

        env.reset()
        ob_rollout = [[] for _ in range(agents + 1)]
        prev_ob_rollout = [[] for _ in range(agents + 1)]
        r_rollout = [[] for _ in range(agents + 1)]
        d_rollout = [[] for _ in range(agents + 1)]
        a_rollout = [[] for _ in range(agents + 1)]
        
        prev = None
        ob = None
        first = 0
        while True:
            for agent in env.agent_iter():
                idx = agent_idx[agent]
                ob, reward, done, info = env.last()
                
                # action = agent.act(ob, reward, done) if not done else None
                action = random.randint(0,4)if not done else None

                env.step(action)

                if first == 0:
                    first == 1
                else:
                    if idx == 0:
                        ob = [*pre_ob , *[0, 0]]
                        prev_ob_rollout[idx].append(prev)
                        ob_rollout[idx].append(ob)
                    else:
                        prev_ob_rollout[idx].append(prev)
                        ob_rollout[idx].append(ob)

                prev = ob
                
                r_rollout[idx].append(reward)
                d_rollout[idx].append(done)
                a_rollout[idx].append(action)
                
            if done:
                break
        # print(len(ob_rollout),len(ob_rollout[0]),len(ob_rollout[1]))
        # print(len(ob_rollout),len(ob_rollout[0][0]),len(ob_rollout[0][1]))
        replay_buffer.append({
            'prev_obs':  np.array(prev_ob_rollout).astype(np.float64),
            'obs': np.array(ob_rollout).astype(np.float64),
            'action': np.array(a_rollout).astype(np.float64),
            'reward': r_rollout,
        })
        
        if i % 10 == 0:
            print("iter "+str(i))

    env.close()

    fname = data_dir + '/data.h5'
    # Save replay buffer to disk.
    utils.save_list_dict_h5py(replay_buffer, fname)

generate_data(episode_count=100, data_dir='data/mpe', agents=2, episode_length=1000)