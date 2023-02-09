import os
import glob
import time
from datetime import datetime

import torch
import numpy as np

import gym
# import roboschool

from PPO import PPO
from env666 import *
import pprint

def showAction(action):
    showStr = ""
    for i in range(8):
        if action[i*2+1] < 0.5:
            showStr+="泵"+str(i)+":"+"关,open value:0\n"
        else:
            showStr+="泵"+str(i)+":"+"开,open value:"+str(action[i*2])+"\n"
    print(showStr)

def showAction2(action):
    showStr = ""
    for i in range(8):
        if action[i] < 0.5:
            showStr+="泵"+str(i)+":"+"关,open value:0\n"
        else:
            showStr+="泵"+str(i)+":"+"开,open value:"+str(action[i])+"\n"
    print(showStr)


def isActionAvailable(action):
    existed_com = {
    '00000001': 87,
    '00000011': 27,
    '00001010': 142,
    '00010001': 1798,
    '00010010': 1664,
    '00010011': 605,
    '00010101': 59,
    '00010110': 137,
    '00011001': 47,
    '00011010': 191,
    '01100001': 68,
    '01100010': 58,
    '01100011': 5,
    '01100110': 72,
    '01101001': 201,
    '01101010': 253,
    '01101100': 1,
    '01101110': 1,
    '01110001': 370,
    '01110010': 726,
    '01110011': 269,
    '01111010': 1,
    '10000001': 1259,
    '10000010': 1002,
    '10000011': 6304,
    '10000101': 334,
    '10000110': 484,
    '10000111': 5061,
    '10001000': 139,
    '10001001': 337,
    '10001010': 646,
    '10001011': 6567,
    '10001100': 3557,
    '10001101': 863,
    '10001110': 836,
    '10001111': 363,
    '10010001': 1206,
    '10010010': 1170,
    '10010011': 11349,
    '10010100': 121,
    '10010101': 32,
    '10010110': 7,
    '10010111': 147,
    '10011000': 344,
    '10011001': 73,
    '10011010': 93,
    '10011011': 186,
    '10011111': 68,
    '11100000': 331,
    '11100001': 49,
    '11100010': 5,
    '11100011': 652,
    '11100100': 275,
    '11100101': 1,
    '11100110': 7,
    '11100111': 44,
    '11101000': 339,
    '11101001': 18,
    '11101010': 50,
    '11101011': 194,
    '11101111': 298,
    '11110000': 61,
    '11110001': 30,
    '11110010': 85,
    '11110011': 776,
    '11110111': 1,
    '11111010': 1,
    '11111011': 99,
    '11111111': 1
    }
    action = np.where(action>0.5,np.ones_like(action),0)
    action = action.tolist()
    # print("hai************",action)
    action = list(map(int,action))
    action = list(map(str,action))
    action_str = "".join(action)
    if action_str in list(existed_com.keys()):
        return True
    return False
################################### Training ###################################
def train(target_flow):
    print("============================================================================================")
    env_name = "WaterPlantEnv666_" + str(target_flow)
    ####### initialize environment hyperparameters ######
    has_continuous_action_space = True  # continuous action space; else discrete

    max_ep_len = 1                   # max timesteps in one episode
    max_training_timesteps = 60_0000   # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * 1000        # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2           # log avg reward in the interval (in num timesteps)
    save_model_freq = int(1e3)          # save model frequency (in num timesteps)

    action_std = 0.4                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = 10_0000  # action_std decay frequency (in num timesteps)
    show_action_freq = 2000
    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    update_timestep = max_ep_len * 3000      # update policy every n timesteps
    K_epochs = 80               # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network

    random_seed = 0         # set random seed if required (0 = no random seed)
    #####################################################

    cfg = {
        "trainSet":"",
        "valSet": "",
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
    env = WaterPlantVenv(cfg)

    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten
    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    #### create new log file for each run
    log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)
    #####################################################

    ################### checkpointing ###################
    run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder

    directory = "PPO_preTrained"
    if not os.path.exists(directory):
          os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
          os.makedirs(directory)


    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("save checkpoint path : " + checkpoint_path)
    #####################################################


    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)
    print("--------------------------------------------------------------------------------------------")
    if has_continuous_action_space:
        print("Initializing a continuous action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("starting std of action distribution : ", action_std)
        print("decay rate of std of action distribution : ", action_std_decay_rate)
        print("minimum std of action distribution : ", min_action_std)
        print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")
    else:
        print("Initializing a discrete action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    #####################################################

    print("============================================================================================")

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # logging file
    log_f = open(log_f_name,"w+")
    log_f.write('episode,timestep,reward\n')

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    # training loop
    while time_step <= max_training_timesteps:

        state = env.reset(target_flow)
        current_ep_reward = 0
        cnt = 0
        # while(cnt<1):
        for t in range(1, max_ep_len+1):
            # select action with policy
            action = ppo_agent.select_action(state)
            # low, high = env.action_space.low, env.action_space.high
            # action = low + (0.5 * (action + 1.0) * (high - low))
            action = np.clip(action, env.action_space.low, env.action_space.high)
            # if not isActionAvailable(action):
            #     ppo_agent.buffer.states.pop()
            #     ppo_agent.buffer.actions.pop()
            #     ppo_agent.buffer.logprobs.pop()
            #     continue
            # cnt+=1
            # print("action:",action)
            state, reward, done, info = env.step(action)
            if time_step%show_action_freq == 0:
                showAction2(action.tolist())
                pprint.pprint({"#####state#####:":state,"#####reward#####":reward,"#####info#####":info})
            # saving reward and is_terminals

            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step +=1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                print("begin update......")
                ppo_agent.update()

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # log in logging file
            if time_step % log_freq == 0:

                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % print_freq == 0:

                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # break; if the episode is over
            if done:
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

    log_f.close()
    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':
    for i in range(12000,30001,500):
        train(i)
    
    
    
    
    
    
    
