from operator import index
from pprint import pprint
import gym
from typing import Dict,Any,Tuple
import random
import numpy as np
import yaml
import json
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pump_model import new_Model


class WaterPlantVenv(gym.Env):
    def __init__(self,cfg:dict) -> None:
        super().__init__()
        self.cfg = cfg

        self.initFlowModel()
        '''
        action定义:[1#泵频率,2#泵频率,3#泵频率,4#泵频率,5#泵频率,6#泵频率,7#泵频率,8#泵频率]
                  该值为缩放到0-1 或者-1-1之间的值
        '''
        if self.cfg["action_mode"]=="sigmoid":
            action_min = np.array([0,0,0,0,0,0,0,0])
            action_max = np.array([1,1,1,1,1,1,1,1])
        else:
            action_min = np.array([-1,-1,-1,-1,-1,-1,-1,-1])
            action_max = np.array([1, 1, 1, 1, 1, 1, 1, 1])
        self.action_space = gym.spaces.Box(low=action_min,high=action_max,dtype=np.float32)
        '''
        state定义:[调度水量指令,1#水位,2#水位]
        # state定义:[调度水量指令,1#水位,2#水位,1#泵瞬时流量,2#泵瞬时流量,3#泵瞬时流量,4#泵瞬时流量,5#泵瞬时流量,6#泵瞬时流量,7#泵瞬时流量,8#泵瞬时流量,
        #             1#出口流量,2#出口流量,出口总流量]
        '''
        state_min = np.array([11272.393555])
        state_max = np.array([35320.148438])
        # state_min = np.array([12000,0,0,0,    0,    0,    0,    0,    0,    0,    0,    0,    0,     0])
        # state_max = np.array([30000,5,5,10000,10000,10000,10000,10000,10000,10000,10000,30000,30000, 30000])

        self.observation_space = gym.spaces.Box(low=state_min, high=state_max, dtype=np.float32)
        self.targetFlowList = list(range(12000,30000,500))
        self.data = None
        self.init_data_sets = None
        self.choose = index
        self.state = np.zeros_like(state_max)

    def initFlowModel(self):
        pump_power_coeffs = [
            [-0.000141903038949108, 0.019485504716210703, -0.9927746787532502, 22.225388069100852, -180.76628504559372],
            [-0.00017102489599191102, 0.023313946939290432, -1.1800111946697542, 26.250313688255012, -212.3655323499383],
            [-5.734998084753995e-05, 0.007484178408261394, -0.357115873315225, 7.326996886401661, -49.67190902306175],
            [-9.008077327541694e-05, 0.01217267461674565, -0.6097379442173119, 13.3862108477156, -104.58309111456406],
            [-0.000149227011175575, 0.019655155965341037, -0.960662264581789, 20.606435994770376, -159.489837148042],
            [-0.0002318718708196963, 0.030980679869216505, -1.5386151544324047, 33.62158141565299, -268.29736852863624],
            [-4.6599986197801176e-05, 0.006486949293711152, -0.3334777944466911, 7.492854504867879, -58.454633697426594],
            [-6.27070007983656e-05, 0.008582246014171775, -0.43380287028825804, 9.582593739253443, -74.49638489597946]
        ]
        self.pump_power_models = {}
        for i in range(8):
            self.pump_power_models[i] = np.poly1d(pump_power_coeffs[i])
    '''
    def anti_normalization(self,x,output_max,output_min):
        if self.cfg["action_mode"]=="sigmoid":
            x = x*(output_max - output_min)+output_min
        else:
            x = (x+1)*0.5*(output_max - output_min)+output_min
        return x

    def build_input_x(self,action):
        self.flow_input_x = np.zeros(self.cfg["flow_input_num"])
        for idx in range(8):
            if self.cfg["action_mode"]=="sigmoid":
                if action[idx] < 0.5:
                    self.flow_input_x[idx] = 0
                else:
                    self.flow_input_x[idx] = (action[idx]-0.5)*(38-28)*2 + 28
            else:
                if action[idx] < 0:
                    self.flow_input_x[idx] = 0
                else:
                    self.flow_input_x[idx] = (action[idx]-0)*(38-28) + 28
        self.check_action = self.flow_input_x
        self.flow_input_x = ((self.flow_input_x - self.flow_input_min) / (self.flow_input_max - self.flow_input_min))*2-1
        self.power_input_x = self.flow_input_x
    '''

    def step(self, action):
        # self.build_input_x(action)
        self.check_action = []
        self.single_flow_output = []
        for idx in range(8):
            if self.cfg["action_mode"]=="sigmoid":
                if action[idx] < 0.615:
                    self.check_action.append(0)
                    self.single_flow_output.append(0)
                else:
                    tmp = (action[idx]-0)*(41-0) + 0
                    self.check_action.append(tmp)
                    self.single_flow_output.append(self.pump_flow_models[idx](tmp))
            else:
                if action[idx] < 0.615:
                    self.check_action.append(0)
                    self.single_flow_output.append(0)
                else:
                    tmp = (action[idx]-0)*(41-0) + 0
                    self.check_action.append(tmp)
                    self.single_flow_output.append(self.pump_flow_models[idx](tmp))
        #self.total_flow = sum(self.single_flow_output)
        
        self.input_action = np.where(action>0.615, action, 0).tolist()
        self.total_flow, self.single_flow_output = self.new_model_flow(self.input_action)
        self.total_flow = self.total_flow + 900
        
        self.power_output,self.available_pump_num = [],0
        for idx in range(8):
            if self.check_action[idx] <= 0:
                self.power_output.append(0)
            else:
                self.power_output.append(self.pump_power_models[idx](self.check_action[idx]))
                self.available_pump_num+=1

        self.mean_power = None
        if self.available_pump_num != 0:
            self.mean_power = sum(self.power_output)/self.available_pump_num
        self.getReward()
        return self.state,self.reward,True,{
                            "input_action":self.input_action,
                            "check_action":self.check_action,
                            "single_flow_output":self.single_flow_output,
                            "total_flow": self.total_flow,
                            "single_power_output":self.power_output,
                            "mean_power":self.mean_power,
                            "power_reward":self.power_reward,
                            "flow_reward": self.flow_reward
                            }

    def getReward(self):
        if self.available_pump_num == 0:
            self.power_reward = -10000
        else:
            self.power_reward = -400*self.mean_power
        
        if self.total_flow - self.targetFlow < -1000:
            self.flow_reward = 2*(self.total_flow - self.targetFlow)
        # elif self.total_flow - self.targetFlow > 1000:
        #     self.flow_reward = -10000
        else:
            self.flow_reward = -abs(self.total_flow - self.targetFlow)
        self.reward = self.flow_reward + self.power_reward

    def reset(self, target_flow) -> Any:
        # length = self.init_data_sets.shape[0]
        # self.initIndex = random.randint(0,length-1)
        # self.water_pos_state = self.init_data_sets[self.initIndex]
        # self.water_pos_state = self.init_data_sets[100]

        # self.targetFlow = random.choice(self.targetFlowList)
        self.targetFlow = target_flow
        self.state = [self.targetFlow]
        return np.array(self.state)
    
    def setState(self,state):
        self.targetFlow = state[0]
        # self.water_pos_state = np.array(state[1:])
        self.state = state

    def render(self):
        pass
    
    
    @staticmethod
    def new_model_flow(action, water_state = [0.5, 0.66, 0.66]):
        model = new_Model()
        model = torch.load('new_Model_3')
        with torch.no_grad():
            test_data_2 = [action, [1,1,1,1,1,1,1,1,1], water_state]
            for i in range(301):
                test_x_2, _2 = collate_fc(test_data_2)
                flow_data = model(test_x_2.view(-1,8), _2[1].view(-1,3))
                flow_data = flow_data[0].numpy()
                flow_data[0] = flow_data[0]*(8732.071289 - 0.0) + 0.0
                flow_data[1] = flow_data[1]*(5773.861816 - 0.0) + 0.0
                flow_data[2] = flow_data[2]*(5908.125977 - 0.0) + 0.0
                flow_data[3] = flow_data[3]*(5742.946289 - 0.0) + 0.0
                flow_data[4] = flow_data[4]*(6260.272461 - 0.0) + 0.0
                flow_data[5] = flow_data[5]*(5529.479492 - 0.0) + 0.0
                flow_data[6] = flow_data[6]*(8361.963867 - 0.0) + 0.0
                flow_data[7] = flow_data[7]*(8364.025391 - 0.0) + 0.0

                total_flow = flow_data.sum()
                total_flow = total_flow + 1144
                Q = (total_flow - 11272.393555)/(35320.148438 - 11272.393555)
                test_data_2[2][0] = Q
            
        return total_flow, flow_data

def collate_fc(samples):
    x = samples[0]
    labels = samples[1]
    z = samples[2]
    return  torch.tensor(x, dtype=torch.float), [torch.tensor(labels, dtype=torch.float), torch.tensor(z, dtype=torch.float)]
    
if __name__=="__main__":
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
    env.reset()
    action = np.array([1, 0.6, 0, 0, 0.3, 0, 0, 0])
    print("action:",action)
    ttt = env.step(action)
    print(ttt)



