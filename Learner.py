from __future__ import print_function
import torch
import numpy as np
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue
import time
from Model import QNet, ValueNet, PolicyNet
from utils import *
import torch.nn as nn
from torch.distributions import Normal
from torch.optim import lr_scheduler

from gym.utils import seeding

class Learner():
    def __init__(self, args, shared_queue,shared_value,share_net,share_optimizer,device,lock,i):
        super(Learner,self).__init__()
        self.args = args
        seed = self.args.seed
        self.init_time = self.args.init_time
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.agent_id = i

        self.experience_out_queue = []
        for i in range(args.num_buffers):
            self.experience_out_queue.append(shared_queue[1][i])

        self.stop_sign = shared_value[1]
        self.iteration_counter  = shared_value[2]
        self.iteration = self.iteration_counter.value

        self.device = device
        if self.device == torch.device("cpu"):
            self.gpu = False
        else:
            self.gpu = True
        self.lock = lock

        self.Q_net1_share, self.Q_net1_target_share, self.Q_net2_share, self.Q_net2_target_share, self.actor1_share, \
                                        self.actor1_target_share, self.actor2_share, self.actor2_target_share, self.log_alpha_share = share_net
        self.Q_net1_optimizer, self.Q_net2_optimizer, self.actor1_optimizer, self.actor2_optimizer, self.alpha_optimizer = share_optimizer


        self.Q_net1 = QNet(args).to(self.device)
        self.scheduler_Q_net1 = lr_scheduler.CosineAnnealingLR(self.Q_net1_optimizer, T_max=self.args.decay_T_max, eta_min=self.args.end_lr, last_epoch=-1)
        self.Q_net1.train()

        self.Q_net1_target = QNet(args).to(self.device)
        self.Q_net1_target.train()

        self.Q_net2 = QNet(args).to(self.device)
        self.scheduler_Q_net2 = lr_scheduler.CosineAnnealingLR(self.Q_net2_optimizer, T_max=self.args.decay_T_max, eta_min=self.args.end_lr, last_epoch=-1)
        self.Q_net2.train()

        self.Q_net2_target = QNet(args).to(self.device)
        self.Q_net2_target.train()

        self.actor1 = PolicyNet(args).to(self.device)
        self.scheduler_actor1 = lr_scheduler.CosineAnnealingLR(self.actor1_optimizer, T_max=self.args.decay_T_max,eta_min=self.args.end_lr, last_epoch=-1)
        self.actor1.train()

        self.actor1_target = PolicyNet(args).to(self.device)
        self.actor1_target.train()

        self.actor2 = PolicyNet(args).to(self.device)
        self.scheduler_actor2 = lr_scheduler.CosineAnnealingLR(self.actor2_optimizer, T_max=self.args.decay_T_max, eta_min=self.args.end_lr, last_epoch=-1)
        self.actor2.train()

        self.actor2_target = PolicyNet(args).to(self.device)
        self.actor2_target.train()

        self.scheduler_alpha = lr_scheduler.CosineAnnealingLR(self.alpha_optimizer, T_max=self.args.decay_T_max, eta_min=self.args.end_lr, last_epoch=-1)

        if self.args.alpha == 'auto':
            self.target_entropy = args.target_entropy
        else:
            self.alpha = torch.tensor(self.args.alpha)


    def get_qloss(self, q, q_std, target_q, target_q_bound):
        if self.args.distributional_Q:
            # loss = -Normal(q, q_std).log_prob(target_q).mean()
            # loss = torch.mean(-Normal(q, q_std).log_prob(target_q_bound)*self.weight \
            #                   + self.weight.logical_not()*torch.pow(q-target_q,2))
            loss = torch.mean(torch.pow(q-target_q,2)/(2*torch.pow(q_std.detach(),2)) \
                   + torch.pow(q.detach()-target_q_bound,2)/(2*torch.pow(q_std,2))\
                   + torch.log(q_std))
        else:
            criterion = nn.MSELoss()
            loss = criterion(q, target_q)
        return loss

    def get_policyloss(self, q, log_prob_a_new):
        loss = (self.alpha.detach() * log_prob_a_new - q).mean()
        return loss

    def update_net(self, loss, optimizer, net, net_share, scheduler):
        optimizer.zero_grad()
        if self.gpu:
            if self.args.alpha == 'auto':
                if net is not self.log_alpha:
                    net.zero_grad()
            else:
                net.zero_grad()
        loss.backward()
        if self.args.alpha == 'auto':
            if net is self.log_alpha:
                if self.log_alpha_share.grad is None or self.log_alpha_share.grad == 0:
                    self.log_alpha_share._grad = self.log_alpha.grad
            else:
                ensure_shared_grads(model=net, shared_model=net_share, gpu=self.gpu)
        else:
            ensure_shared_grads(model=net, shared_model=net_share, gpu=self.gpu)
        optimizer.step()
        scheduler.step(self.iteration)

    def target_q(self,r,done, q, q_std, q_next,log_prob_a_next):
        target_q = r + (1 - done) * self.args.gamma * (q_next - self.alpha.detach() * log_prob_a_next)
        if self.args.distributional_Q:
            if self.args.adaptive_bound:
                target_max = q + 3 * q_std
                target_min = q - 3 * q_std
                target_q = torch.min(target_q, target_max)
                target_q = torch.max(target_q, target_min)
            difference = torch.clamp(target_q - q, -self.args.TD_bound, self.args.TD_bound)
            target_q_bound = q + difference
            self.weight = torch.le(torch.abs(target_q - q), self.args.TD_bound).detach()
        else:
            target_q_bound = target_q
        return target_q.detach(), target_q_bound.detach()

    def send_to_device(self, s, info, a, r, s_next, info_next, done, device):
        s = s.to(device)
        info = info.to(device)
        a = a.to(device)
        r = r.to(device)
        s_next = s_next.to(device)
        info_next = info_next.to(device)
        done = done.to(device)
        return s, info, a, r, s_next, info_next, done

    def run(self):
        local_iteration = 0
        index = np.random.randint(0, self.args.num_buffers)
        while self.experience_out_queue[index].empty() and not self.stop_sign.value:
            index = np.random.randint(0, self.args.num_buffers)
            time.sleep(0.1)

        while not self.stop_sign.value:
            self.iteration = self.iteration_counter.value
            self.Q_net1.load_state_dict(self.Q_net1_share.state_dict())
            self.Q_net1_target.load_state_dict(self.Q_net1_target_share.state_dict())
            self.Q_net2.load_state_dict(self.Q_net2_share.state_dict())
            self.Q_net2_target.load_state_dict(self.Q_net2_target_share.state_dict())
            self.actor1.load_state_dict(self.actor1_share.state_dict())
            self.actor1_target.load_state_dict(self.actor1_target_share.state_dict())
            self.actor2.load_state_dict(self.actor2_share.state_dict())
            self.actor2_target.load_state_dict(self.actor2_target_share.state_dict())
            if self.args.alpha == 'auto':
                self.log_alpha = self.log_alpha_share.detach().clone().requires_grad_(True)
                self.alpha = self.log_alpha.exp().to(self.device)

            index = np.random.randint(0, self.args.num_buffers)
            while self.experience_out_queue[index].empty() and not self.stop_sign.value:
                index = np.random.randint(0, self.args.num_buffers)
                time.sleep(0.1)
            if not self.experience_out_queue[index].empty():
                s, info, a, r, s_next, info_next, done = self.experience_out_queue[index].get()
                s, info, a, r, s_next, info_next, done = self.send_to_device(s, info, a, r, s_next, info_next, done, self.device)

            q_1, q_std_1, _ = self.Q_net1.evaluate(s, info, a,device=self.device, min=False)
            if self.args.double_Q:
                q_2, q_std_2, _ = self.Q_net2.evaluate(s, info, a,device=self.device, min=False)

            smoothing_trick = False
            if not self.args.stochastic_actor:
                if self.args.policy_smooth:
                    smoothing_trick = True

            a_new_1, log_prob_a_new_1, a_new_std_1 = self.actor1.evaluate(s, info, smooth_policy = False, device=self.device)
            a_next_1, log_prob_a_next_1, _ = self.actor1_target.evaluate(s_next, info_next, smooth_policy = smoothing_trick, device=self.device)
            if self.args.double_actor:
                a_new_2, log_prob_a_new_2, _ = self.actor2.evaluate(s, info, smooth_policy = False, device=self.device)
                a_next_2, log_prob_a_next_2, _ = self.actor2_target.evaluate(s_next, info_next, smooth_policy = smoothing_trick, device=self.device)

            if self.args.double_Q and self.args.double_actor:
                q_next_target_1, _, q_next_sample_1 = self.Q_net2_target.evaluate(s_next, info_next, a_next_1, device=self.device, min=False)
                q_next_target_2, _, _ = self.Q_net1_target.evaluate(s_next, info_next, a_next_2, device=self.device, min=False)
                target_q_1, target_q_1_bound = self.target_q(r, done, q_1.detach(), q_std_1.detach(), q_next_target_1.detach(), log_prob_a_next_1.detach())
                target_q_2, target_q_2_bound = self.target_q(r, done, q_2.detach(), q_std_2.detach(), q_next_target_2.detach(), log_prob_a_next_2.detach())
            else:
                q_next_1, _, q_next_sample_1 = self.Q_net1_target.evaluate(s_next, info_next, a_next_1, device=self.device, min=False)
                if self.args.double_Q:
                    q_next_2, _, _ = self.Q_net2_target.evaluate(s_next, info_next, a_next_1, device=self.device, min=False)
                    q_next_target_1 = torch.min(q_next_1,q_next_2)
                elif self.args.distributional_Q:
                    q_next_target_1 = q_next_sample_1
                else:
                    q_next_target_1 = q_next_1
                target_q_1, target_q_1_bound = self.target_q(r, done, q_1.detach(), q_std_1.detach(), q_next_target_1.detach(), log_prob_a_next_1.detach())

            if self.args.double_Q and self.args.double_actor:
                q_object_1, _, _ = self.Q_net1.evaluate(s, info, a_new_1, device=self.device, min=False)
                q_object_2, _, _ = self.Q_net2.evaluate(s, info, a_new_2, device=self.device, min=False)
            else:
                q_new_1, _, _ = self.Q_net1.evaluate(s, info, a_new_1,device=self.device, min=False)
                if self.args.double_Q:
                    q_new_2, _, _ = self.Q_net2.evaluate(s, info, a_new_1,device=self.device, min=False)
                    q_object_1 = torch.min(q_new_1,q_new_2)
                elif self.args.distributional_Q:
                    q_object_1 = q_new_1
                else:
                    q_object_1 = q_new_1

            if local_iteration % self.args.delay_update == 0:
                if self.args.alpha == 'auto':
                    alpha_loss = -(self.log_alpha * (log_prob_a_new_1.detach().cpu() + self.target_entropy)).mean()
                    self.update_net(alpha_loss, self.alpha_optimizer, self.log_alpha, self.log_alpha_share, self.scheduler_alpha)

            q_loss_1 = self.get_qloss(q_1, q_std_1, target_q_1, target_q_1_bound)
            self.update_net(q_loss_1, self.Q_net1_optimizer, self.Q_net1, self.Q_net1_share, self.scheduler_Q_net1)
            if self.args.double_Q:
                if self.args.double_actor:
                    q_loss_2 = self.get_qloss(q_2, q_std_2, target_q_2, target_q_2_bound)
                    self.update_net(q_loss_2, self.Q_net2_optimizer, self.Q_net2, self.Q_net2_share, self.scheduler_Q_net2)
                else:
                    q_loss_2 = self.get_qloss(q_2, q_std_2, target_q_1, target_q_1_bound)
                    self.update_net(q_loss_2, self.Q_net2_optimizer, self.Q_net2, self.Q_net2_share, self.scheduler_Q_net2)

            if self.args.code_model == "train":
                if local_iteration % self.args.delay_update == 0:
                    policy_loss_1 = self.get_policyloss(q_object_1, log_prob_a_new_1)
                    self.update_net(policy_loss_1, self.actor1_optimizer, self.actor1, self.actor1_share, self.scheduler_actor1)
                    slow_sync_param(self.actor1_share, self.actor1_target_share, self.args.tau, self.gpu)
                    if self.args.double_actor:
                        policy_loss_2 = self.get_policyloss(q_object_2, log_prob_a_new_2)
                        self.update_net(policy_loss_2, self.actor2_optimizer, self.actor2, self.actor2_share, self.scheduler_actor2)
                        slow_sync_param(self.actor2_share, self.actor2_target_share, self.args.tau, self.gpu)

            if local_iteration % self.args.delay_update == 0:
                slow_sync_param(self.Q_net1_share, self.Q_net1_target_share, self.args.tau, self.gpu)
                if self.args.double_Q:
                    slow_sync_param(self.Q_net2_share, self.Q_net2_target_share, self.args.tau, self.gpu)

            with self.lock:
                self.iteration_counter.value += 1
            local_iteration += 1

            if self.iteration % self.args.save_model_period == 0 or (self.iteration== 0 and self.agent_id==0):
                torch.save(self.actor1.state_dict(),'./'+self.args.env_name+'/method_' + str(self.args.method) + '/model/policy1_' + str(self.iteration) + '.pkl')
                torch.save(self.Q_net1.state_dict(),'./'+self.args.env_name+'/method_' + str(self.args.method) + '/model/Q1_' + str(self.iteration) + '.pkl')
                if self.args.alpha == 'auto':
                    np.save('./' + self.args.env_name + '/method_' + str(self.args.method) + '/model/log_alpha' + str(self.iteration), self.log_alpha.detach().cpu().numpy())
                if self.args.double_Q:
                    torch.save(self.Q_net2.state_dict(),'./'+self.args.env_name+'/method_' + str(self.args.method) + '/model/Q2_' + str(self.iteration) + '.pkl')
                if self.args.double_actor:
                    torch.save(self.actor2.state_dict(),'./' + self.args.env_name + '/method_' + str(self.args.method) + '/model/policy2_' + str(self.iteration) + '.pkl')

            if self.iteration % 500  == 0 or self.iteration== 0 and self.agent_id==0:
                print("agent",self.agent_id,"method",self.args.method,"iteration", self.iteration, "time",time.time() - self.init_time)
                print("loss_1", q_loss_1, "alpha",self.alpha,"lr",self.scheduler_Q_net1.get_lr(), self.scheduler_Q_net2.get_lr(),self.scheduler_actor1.get_lr(),
                      self.scheduler_actor2.get_lr(),self.scheduler_alpha.get_lr())
                print("q_std",q_std_1.t()[0][0:8])
                print("a_std", a_new_std_1.t()[0][0:8])
                #print("mean", q_1.t())


def test():
    for i in range(2,5,1):
        print(i)

    a = np.array([[1.,2.,3.],[2.,3.,1.]])
    b = np.array([1.,2.,3.])
    aa =torch.FloatTensor(a)
    bb = torch.FloatTensor(b)
    cc=torch.mul(aa,bb)
    dd = torch.min(aa,bb)

    method_name = {0: 'DSAC', 1: 'SAC', 2: 'SAC with Double-Q', 3: 'TD3', 4: 'DDPG'}



    print(method_name[0])



if __name__ == "__main__":
    test()
