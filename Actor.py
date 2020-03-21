from __future__ import print_function
import torch
import numpy as np
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue
import time
from Model import QNet, PolicyNet
import gym
from utils import *


class Actor():
    def __init__(self, args, shared_queue, shared_value,share_net, lock, i):
        super(Actor, self).__init__()
        self.agent_id = i
        seed = args.seed + np.int64(self.agent_id)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.actor_params = {
            'obs_size': 128,  # screen size of cv2 window
            'dt': 0.1,  # time interval between two frames
            'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
            'port': 2000,  # connection port
            'task_mode': 'Straight',  # mode of the task, [random, roundabout (only for Town03)]
            'code_mode': 'train',
            'max_time_episode': 1000,  # maximum timesteps per episode
            'desired_speed': 8,  # desired speed (m/s)
            'max_ego_spawn_times': 100,  # maximum times to spawn ego vehicle
        }
        print(self.actor_params['port'])
        self.counter = shared_value[0]
        self.stop_sign = shared_value[1]
        self.lock = lock
        self.env = gym.make(args.env_name, params=self.actor_params)
        self.args = args
        self.experience_in_queue = []
        for i in range(args.num_buffers):
            self.experience_in_queue.append(shared_queue[0][i])

        self.device = torch.device("cpu")
        self.actor = PolicyNet(args).to(self.device)
        self.Q_net1 = QNet(args).to(self.device)

        #share_net = [Q_net1,Q_net1_target,Q_net2,Q_net2_target,actor,actor_target,log_alpha]
        #share_optimizer=[Q_net1_optimizer,Q_net2_optimizer,actor_optimizer,alpha_optimizer]
        self.Q_net1_share = share_net[1]
        self.actor_share = share_net[0]


    def put_data(self):
        if not self.stop_sign.value:
            index = np.random.randint(0, self.args.num_buffers)
            if self.experience_in_queue[index].full():
                #print("agent", self.agent_id, "is waiting queue space")
                time.sleep(0.5)
                self.put_data()
            else:
                # self.experience_in_queue[index].put((self.last_state, self.last_u, [self.reward*self.args.reward_scale], self.state, [self.done], self.TD.detach().cpu().numpy().squeeze()))
                self.experience_in_queue[index].put((self.last_state, self.last_info, self.last_u, [self.reward*self.args.reward_scale], self.state, self.info, [self.done], self.TD.detach().cpu().numpy().squeeze()))
        else:
            pass

    def run(self):
        time_init = time.time()
        step = 0
        while not self.stop_sign.value:
            self.state, self.ego_info = self.env.reset()  # shape(256,256,3)
            self.episode_step = 0
            self.info = info_dict_to_array(self.ego_info)  # shape(10,)

            state_tensor = torch.FloatTensor(self.state.copy()).float().to(self.device)  # 256,256,3
            info_tensor = torch.FloatTensor(self.info.copy()).float().to(self.device)

            if self.args.NN_type == "CNN":
                state_tensor = state_tensor.permute(2, 0, 1)  # 3, 256, 256
            self.u, _ = self.actor.get_action(state_tensor.unsqueeze(0), info_tensor.unsqueeze(0), False)
            #q_1 = self.Q_net1(state_tensor.unsqueeze(0), torch.FloatTensor(self.u).to(self.device))[0]
            self.u = self.u.squeeze(0)
            self.last_state = self.state.copy()
            self.last_info = self.info.copy()
            self.last_u = self.u.copy()
            #last_q_1 = q_1
            for i in range(self.args.max_step-1):
                self.state, self.reward, self.done, self.ego_info = self.env.step(self.u)
                self.info = info_dict_to_array(self.ego_info)

                state_tensor = torch.FloatTensor(self.state.copy()).float().to(self.device)
                info_tensor = torch.FloatTensor(self.info.copy()).float().to(self.device)

                if self.args.NN_type == "CNN":
                    state_tensor = state_tensor.permute(2, 0, 1)
                self.u, _ = self.actor.get_action(state_tensor.unsqueeze(0), info_tensor.unsqueeze(0), False)
                #q_1 = self.Q_net1(state_tensor.unsqueeze(0), torch.FloatTensor(self.u).to(self.device))[0]
                self.u = self.u.squeeze(0)

                # TODO
                # print(self.u)

                self.TD = torch.zeros(1) #self.reward + (1 - self.done) * self.args.gamma * q_1 - last_q_1
                self.put_data()

                self.last_state = self.state.copy()
                self.last_info = self.info.copy()
                self.last_u = self.u.copy()
                #last_q_1 = q_1

                with self.lock:
                    self.counter.value += 1

                if self.done == True:
                    # TODO
                    # print("Time steps:", self.env.time_step)
                    break

                if step % self.args.load_param_period == 0:
                    #self.Q_net1.load_state_dict(self.Q_net1_share.state_dict())
                    self.actor.load_state_dict(self.actor_share.state_dict())
                step += 1
                self.episode_step += 1

def test():
    def xxxx():
        time.sleep(1)
        print("!!!!!!")
        xxxx()
    xxxx()


if __name__ == "__main__":
    test()



