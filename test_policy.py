

from PPO import PPO
import json
from env4 import *

def test():
    print("============================================================================================")

    ################## hyperparameters ##################

    has_continuous_action_space = True
    action_std = 0.000000001            # set same std for action distribution which was used while saving

    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor

    lr_actor = 0.0003           # learning rate for actor
    lr_critic = 0.001           # learning rate for critic

    #####################################################

    cfg = {
        "trainSet":"C:\\polixirFiles\\PowerPlant\\南泉水源厂\\codes\\data\\data_to_haien_train.csv",
        "valSet": "C:\\polixirFiles\\PowerPlant\\南泉水源厂\\codes\\data\\data_to_haien_val.csv",
        # self.cfg["single_flow"]["graph"]["input"]
        "flow":{"input_titles":["1号单泵频率","2号单泵频率","3号单泵频率","4号单泵频率",
                                "5号单泵频率","6号单泵频率","7号单泵频率","8号单泵频率"]},
        "power":{"input_titles":["1号单泵频率","2号单泵频率","3号单泵频率","4号单泵频率",
                                "5号单泵频率","6号单泵频率","7号单泵频率","8号单泵频率"]},

        "flow_input_num":8,
        "flow_output_num":1,
        "spower_input_num":8,
        "power_output_num":1,

        "flow_model":"./best_total_flow.pkl",
        "flow_maxMinSave_path": "./maxMinVal_total_flow.json",

        "power_model": "./best_total_power.pkl",
        "power_maxMinSave_path": "./maxMinVal_total_power.json",

        "static_var": ["1号吸水井液位标高值","2号吸水井液位标高值"],

        "action_mode":"sigmoid",
    }

    env = WaterPlantVenv()

    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    state = env.reset()
    #env.setState([19500])
    action = ppo_agent.select_action(state)
    action = np.clip(action, env.action_space.low, env.action_space.high)
    state, reward, done, info = env.step(action)
    print(state, reward, done, info)


if __name__=="__main__":
    test()