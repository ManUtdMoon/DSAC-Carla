from __future__ import print_function
import torch
import numpy as np
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue
# from torch.utils.tensorboard import SummaryWriter
import time
from Model import PolicyNet,QNet
import gym
import matplotlib.pyplot as plt



class Simulation():
    def __init__(self, args,shared_value):
        super(Simulation, self).__init__()
        seed = args.seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        simu_params = {
            'number_of_vehicles': 0,
            'number_of_walkers': 0,
            'obs_size': (160, 100),  # screen size of cv2 window
            'dt': 0.1,  # time interval between two frames
            'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
            'port': 2000,  # connection port
            'task_mode': 'Straight',  # mode of the task, [random, roundabout (only for Town03)]
            'code_mode': 'test',
            'max_time_episode': 500,  # maximum timesteps per episode
            'desired_speed': 8,  # desired speed (m/s)
            'max_ego_spawn_times': 100,  # maximum times to spawn ego vehicle
        }

        self.stop_sign = shared_value[1]
        self.args = args
        self.env = gym.make(args.env_name, params=simu_params)
        self.device = torch.device("cpu")
        self.load_index = self.args.max_train

        self.actor = PolicyNet(args).to(self.device)
        self.actor.load_state_dict(torch.load('./'+self.args.env_name+'/method_' + str(self.args.method) + '/model/policy1_' + str(self.load_index) + '.pkl',map_location='cpu'))

        self.Q_net1 = QNet(args).to(self.device)
        self.Q_net1.load_state_dict(torch.load('./'+self.args.env_name+'/method_' + str(self.args.method) + '/model/Q1_' + str(self.load_index) + '.pkl',map_location='cpu'))

        if self.args.double_Q:
            self.Q_net2 = QNet(args).to(self.device)
            self.Q_net2.load_state_dict(torch.load('./'+self.args.env_name+'/method_' + str(self.args.method) + '/model/Q2_' + str(self.load_index) + '.pkl',map_location='cpu'))


        self.test_step = 0
        self.save_interval = 10000
        self.iteration = 0
        self.reward_history = []
        self.entropy_history = []
        self.epoch_history =[]
        self.done_history = []
        self.Q_real_history = []
        self.Q_history =[]
        self.Q_std_history = []


    def run(self):
        alpha = 0.004

        step = 0
        while True:
            self.state, _ = self.env.reset()
            self.episode_step = 0
            state_tensor = torch.FloatTensor(self.state.copy()).float().to(self.device)
            if self.args.NN_type == "CNN":
                state_tensor = state_tensor.permute(2, 0, 1)
            self.u, log_prob = self.actor.get_action(state_tensor.unsqueeze(0), True)


            for i in range(600):
                q = self.Q_net1(state_tensor.unsqueeze(0), torch.FloatTensor(self.u).to(self.device))[0]
                if self.args.double_Q:
                    q = torch.min(
                        self.Q_net1(state_tensor.unsqueeze(0), torch.FloatTensor(self.u).to(self.device))[0],
                        self.Q_net2(state_tensor.unsqueeze(0), torch.FloatTensor(self.u).to(self.device))[0])


                self.Q_history.append(q.detach().item())

                self.u = self.u.squeeze(0)

                # TODO
                with SummaryWriter(log_dir='./logs') as writer:
                    writer.add_scalar('accel', self.u[0], i)
                    writer.add_scalar('steer', self.u[1], i)
                    # writer.add_scalar('random', np.random.randint(0, 10), i)
                    v = self.env.ego.get_velocity()
                    v = np.array([v.x, v.y, v.z])
                    writer.add_scalar('velocity', np.linalg.norm(v), i)

                self.state, self.reward, self.done, _ = self.env.step(self.u)

                self.reward_history.append(self.reward)
                self.done_history.append(self.done)
                self.entropy_history.append(log_prob)


                if step%10000 >=0 and step%10000 <=9999:
                    self.env.render(mode='human')
                state_tensor = torch.FloatTensor(self.state.copy()).float().to(self.device)
                if self.args.NN_type == "CNN":
                    state_tensor = state_tensor.permute(2, 0, 1)
                self.u, log_prob = self.actor.get_action(state_tensor.unsqueeze(0), True)



                if self.done == True:
                    time.sleep(1)
                    print("Episode Done!")
                    return
                    break
                step += 1
                self.episode_step += 1

            if self.done == True:
                pass
                #break


        print(self.reward_history)
        for i in range(len(self.Q_history)):
            a = 0
            for j in range(i, len(self.Q_history), 1):
                a += pow(self.args.gamma, j-i)*self.reward_history[j]
            for z in range(i+1, len(self.Q_history), 1):
                a -= alpha * pow(self.args.gamma, z-i) * self.entropy_history[z]
            self.Q_real_history.append(a)


        plt.figure()
        x = np.arange(0,len(self.Q_history),1)
        plt.plot(x, np.array(self.Q_history), 'r', linewidth=2.0)
        plt.plot(x, np.array(self.Q_real_history), 'k', linewidth=2.0)

        plt.show()




def test():
    a = np.arange(0,10,1)
    print(a)


if __name__ == "__main__":
    test()



