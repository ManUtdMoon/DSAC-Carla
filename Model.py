from __future__ import print_function
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
from torch.distributions import Normal
import math

def conv_block(in_channel, out_channel, kernel_size, stride):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride),
        nn.BatchNorm2d(num_features=out_channel),
        nn.GELU(),)

def init_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            weight_shape = list(m.weight.data.size())
            fan_in = weight_shape[1]
            fan_out = weight_shape[0]
            w_bound = np.sqrt(6. / (fan_in + fan_out))
            m.weight.data.uniform_(-w_bound, w_bound)
            m.bias.data.fill_(0)
        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class QNet(nn.Module):
    def __init__(self, args, log_std_min=0., log_std_max=4):
        super(QNet, self).__init__()
        num_states = args.state_dim
        num_action = args.action_dim
        num_hidden_cell = args.num_hidden_cell
        num_info = args.info_dim
        self.NN_type = args.NN_type
        if self.NN_type == "CNN":
            self.conv_part = nn.Sequential(
                conv_block(in_channel=num_states[-1], out_channel=32, kernel_size=5, stride=2),  # in: n, 3, 160, 64
                conv_block(in_channel=32, out_channel=32, kernel_size=3, stride=1),  # in: n, 32, 78, 30
                conv_block(in_channel=32, out_channel=64, kernel_size=3, stride=2),  # in: n, 32, 76, 28
                conv_block(in_channel=64, out_channel=64, kernel_size=3, stride=1),  # in: n, 64, 37, 13
                conv_block(in_channel=64, out_channel=128, kernel_size=3, stride=2),  # in: n, 64, 35, 11
                conv_block(in_channel=128, out_channel=128, kernel_size=3, stride=1),  # in: n, 128, 17, 5
                conv_block(in_channel=128, out_channel=256, kernel_size=3, stride=2),)  # in: n, 128, 15, 3
            _conv_out_size = self._get_conv_out_size(num_states)
            # n, 256, 5, 1 -> 256
            self.linear_img = nn.Linear(256 * 7 * 1, num_hidden_cell, bias=True)
            self.linear_info = nn.Linear(num_info + num_action, 128, bias=True)
            self.linear1 = nn.Linear(256+128, num_hidden_cell, bias=True)
            self.linear2 = nn.Linear(num_hidden_cell, num_hidden_cell, bias=True)

        # the size of info tensor is (10,1)
        self.mean_layer = nn.Linear(num_hidden_cell, 1, bias=True)
        self.log_std_layer = nn.Linear(num_hidden_cell, 1, bias=True)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.denominator = max(abs(self.log_std_min), self.log_std_max)
        init_weights(self)

    def _get_conv_out_size(self, num_states):
        out = self.conv_part(torch.zeros(num_states).unsqueeze(0).permute(0,3,1,2))
        return int(np.prod(out.size()))

    def forward(self, state, info, action):
        if self.NN_type == "CNN":
            x1 = self.conv_part(state)
            x1 = x1.view(state.size(0),-1)
            x1 = F.gelu(self.linear_img(x1))

            x2 = F.gelu(self.linear_info(torch.cat([info, action], 1)))
            x = torch.cat([x1, x2], 1)

            x = F.gelu(self.linear1(x))
            x = F.gelu(self.linear2(x))

        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)

        log_std = torch.clamp_min(self.log_std_max*torch.tanh(log_std/self.denominator),0) + \
                  torch.clamp_max(-self.log_std_min * torch.tanh(log_std / self.denominator), 0)

        # log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def evaluate(self, state, info, action, device=torch.device("cpu"), min=False):
        mean, log_std = self.forward(state, info, action)
        std = log_std.exp()
        normal = Normal(torch.zeros(mean.shape), torch.ones(std.shape))

        if min == False:
            z = normal.sample().to(device)
            z = torch.clamp(z, -2, 2)
        elif min == True:
            z = -torch.abs(normal.sample()).to(device)

        q_value = mean + torch.mul(z, std)
        return mean, std, q_value



class PolicyNet(nn.Module):
    def __init__(self, args, log_std_min=-5, log_std_max=1):
        super(PolicyNet, self).__init__()
        num_states = args.state_dim
        num_hidden_cell = args.num_hidden_cell
        num_info = args.info_dim
        action_high = args.action_high
        action_low = args.action_low
        self.NN_type = args.NN_type
        self.args= args

        if self.NN_type == "CNN":
            self.conv_part = nn.Sequential(
                conv_block(in_channel=num_states[-1], out_channel=32, kernel_size=5, stride=2),  # in: n, 3, 160, 64
                conv_block(in_channel=32, out_channel=32, kernel_size=3, stride=1),  # in: n, 32, 78, 30
                conv_block(in_channel=32, out_channel=64, kernel_size=3, stride=2),  # in: n, 32, 76, 28
                conv_block(in_channel=64, out_channel=64, kernel_size=3, stride=1),  # in: n, 64, 37, 13
                conv_block(in_channel=64, out_channel=128, kernel_size=3, stride=2),  # in: n, 64, 35, 11
                conv_block(in_channel=128, out_channel=128, kernel_size=3, stride=1),  # in: n, 128, 17, 5
                conv_block(in_channel=128, out_channel=256, kernel_size=3, stride=2),)  # in: n, 128, 15, 3
            _conv_out_size = self._get_conv_out_size(num_states)

            # n, 32, 6, 6 -> 256
            self.linear_img = nn.Linear(256 * 7 * 1, num_hidden_cell, bias=True)
            self.linear_info = nn.Linear(num_info, 128, bias=True)
            self.linear1 = nn.Linear(256+128, num_hidden_cell, bias=True)
            self.linear2 = nn.Linear(num_hidden_cell, num_hidden_cell, bias=True)

        self.mean_layer = nn.Linear(num_hidden_cell, len(action_high), bias=True)
        self.log_std_layer = nn.Linear(num_hidden_cell, len(action_high), bias=True)
        init_weights(self)

        self.action_high = torch.tensor(action_high, dtype=torch.float32)
        self.action_low = torch.tensor(action_low, dtype=torch.float32)
        self.action_range = (self.action_high - self.action_low)/2
        self.action_bias =  (self.action_high + self.action_low)/2
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.denominator = max(abs(self.log_std_min), self.log_std_max)

    def _get_conv_out_size(self, num_states):
        out = self.conv_part(torch.zeros(num_states).unsqueeze(0).permute(0,3,1,2))
        return int(np.prod(out.size()))


    def forward(self, state, info):
        if self.NN_type == "CNN":
            x1 = self.conv_part(state)
            x1 = x1.view(state.size(0),-1)
            x1 = F.gelu(self.linear_img(x1))

            x2 = F.gelu(self.linear_info(info))
            x = torch.cat([x1, x2], 1)

            x = F.gelu(self.linear1(x))
            x = F.gelu(self.linear2(x))

        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp_min(self.log_std_max*torch.tanh(log_std/self.denominator),0) + \
                  torch.clamp_max(-self.log_std_min * torch.tanh(log_std / self.denominator), 0)
        # log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def evaluate(self, state, info, smooth_policy, device=torch.device("cpu") , epsilon=1e-4):

        mean, log_std = self.forward(state, info)
        normal = Normal(torch.zeros(mean.shape), torch.ones(log_std.shape))
        z = normal.sample().to(device)
        std = log_std.exp()
        if self.args.stochastic_actor:
            z = torch.clamp(z, -3, 3)
            action_0 = mean + torch.mul(z, std)
            action_1 = torch.tanh(action_0)
            action = torch.mul(self.action_range.to(device), action_1) + self.action_bias.to(device)
            log_prob = Normal(mean, std).log_prob(action_0)-torch.log(1. - action_1.pow(2) + epsilon) - torch.log(self.action_range.to(device))
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            return action, log_prob , std.detach(), std.detach()
        else:
            action_mean = torch.mul(self.action_range.to(device), torch.tanh(mean)) + self.action_bias.to(device)
            smooth_random = torch.clamp(0.2*z, -0.5, 0.5)
            action_random = action_mean + torch.mul(self.action_range.to(device),smooth_random)
            action_random = torch.min(action_random, self.action_high.to(device))
            action_random = torch.max(action_random, self.action_low.to(device))
            action = action_random if smooth_policy else action_mean
            return action, 0*log_std.sum(dim=-1, keepdim=True) , std.detach()


    def get_action(self, state, info, deterministic, epsilon=1e-4):
        mean, log_std = self.forward(state, info)
        normal = Normal(torch.zeros(mean.shape), torch.ones(log_std.shape))
        z = normal.sample()
        if self.args.stochastic_actor:
            std = log_std.exp()
            action_0 = mean + torch.mul(z, std)
            action_1 = torch.tanh(action_0)
            action = torch.mul(self.action_range, action_1) + self.action_bias
            log_prob = Normal(mean, std).log_prob(action_0)-torch.log(1. - action_1.pow(2) + epsilon) - torch.log(self.action_range)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            action_mean = torch.mul(self.action_range, torch.tanh(mean)) + self.action_bias
            action = action_mean.detach().cpu().numpy() if deterministic else action.detach().cpu().numpy()
            return action, log_prob.detach().item(), std.detach().cpu().numpy().squeeze()
        else:
            action_mean = torch.mul(self.action_range, torch.tanh(mean)) + self.action_bias
            action = action_mean + 0.1 * torch.mul(self.action_range,z)
            action = torch.min(action, self.action_high)
            action = torch.max(action, self.action_low)
            action = action_mean.detach().cpu().numpy() if deterministic else action.detach().cpu().numpy()
            return action, 0, 0*(log_std.detach().cpu().numpy().squeeze())




class ValueNet(nn.Module):
    def __init__(self, num_states, num_hidden_cell, NN_type):
        super(ValueNet, self).__init__()
        self.NN_type = NN_type
        num_info = 10

        if self.NN_type == "CNN":
            self.conv_part = nn.Sequential(
                conv_block(in_channel=num_states[-1], out_channel=32, kernel_size=5, stride=2),  # in: n, 3, 160, 64
                conv_block(in_channel=32, out_channel=32, kernel_size=3, stride=1),  # in: n, 32, 78, 30
                conv_block(in_channel=32, out_channel=64, kernel_size=3, stride=2),  # in: n, 32, 76, 28
                conv_block(in_channel=64, out_channel=64, kernel_size=3, stride=1),  # in: n, 64, 37, 13
                conv_block(in_channel=64, out_channel=128, kernel_size=3, stride=2),  # in: n, 64, 35, 11
                conv_block(in_channel=128, out_channel=128, kernel_size=3, stride=1),  # in: n, 128, 17, 5
                conv_block(in_channel=128, out_channel=256, kernel_size=3, stride=2),)  # in: n, 128, 15, 3
            _conv_out_size = self._get_conv_out_size(num_states)

            self.linear_img = nn.Linear(256 * 7 * 1, num_hidden_cell, bias=True)
            self.linear_info = nn.Linear(num_info, 128, bias=True)
            self.linear1 = nn.Linear(256+128, num_hidden_cell, bias=True)
            self.linear2 = nn.Linear(num_hidden_cell, num_hidden_cell, bias=True)
            self.linear3 = nn.Linear(num_hidden_cell, 1, bias=True)

        init_weights(self)

    def _get_conv_out_size(self, num_states):
        out = self.conv_part(torch.zeros(num_states).unsqueeze(0).permute(0,3,1,2))
        return int(np.prod(out.size()))

    def forward(self, state, info):
        if self.NN_type == "CNN":
            x1 = self.conv_part(state)
            x1 = x1.view(state.size(0),-1)
            x1 = F.gelu(self.linear_img(x1))

            x2 = F.gelu(self.linear_info(info))
            x = torch.cat([x1, x2], 1)

            x = F.gelu(self.linear1(x))
            x = F.gelu(self.linear2(x))
            x = self.linear3(x)

        return x



class Args(object):
    def __init__(self):
        self.state_dim = (64, 160, 3)
        self.action_dim = 2
        self.NN_type = 'CNN'
        self.num_hidden_cell = 256
        self.action_high = [1.0, 1.0]
        self.action_low = [-1.0, -1.0]
        self.stochastic_actor = True
        self.info_dim = 10


def test():
    # mean = torch.tensor([[0,0],[0.5,0.5]], dtype = torch.float32)
    # sig = torch.tensor([[1, 1],[2,2]], dtype=torch.float32)
    # print(mean.shape)
    # bbb = torch.zeros(mean.shape)
    # ccc = torch.ones(sig.shape)
    # dist = Normal(bbb, ccc).sample()

    # pro = Normal(bbb, ccc).log_prob(dist)
    # print(pro)

    # bb = dist.pow(2)
    # print(bb)
    # print(bb-1)
    # print(bb.sum(-1, keepdim=True))

    args = Args()
    img = torch.rand((1, 3, 160, 64))
    info = torch.rand((1, 10))
    action = torch.ones((1, 2))
    # q_net = QNet(args)
    # print(q_net.forward(img, info, action))
    # print(q_net.evaluate(img, info, action))
    # total_num = sum(p.numel() for p in q_net.parameters())
    # print(total_num)

    p_net = PolicyNet(args)
    total_num = sum(p.numel() for p in p_net.parameters())
    print(total_num)
    p_net.forward(img, info)
    print(info.requires_grad)
    print(p_net.get_action(img, info, True))
    p_net.evaluate(img, info, False)

    # v_net = ValueNet((160, 64, 3), 256, 'CNN')
    # v_net.forward(img, info)
    # total_num = sum(p.numel() for p in v_net.parameters())
    # print(total_num)


if __name__ == "__main__":
    test()
