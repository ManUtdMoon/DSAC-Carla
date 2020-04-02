import math

import numpy as np

import torch


def normal_entropy(std):
    var = std.pow(2)
    entropy = 0.5 + 0.5 * torch.log(2 * var * math.pi)
    return entropy.sum(1, keepdim=True)


def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (
        2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1, keepdim=True)


def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params


def get_flat_params_from_dict(state_dict):
    params = []
    for name, param in state_dict.items():
        params.append(param.data.view(-1))
    flat_params = torch.cat(params)
    return flat_params


def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


def get_flat_grad_from(net, grad_grad=False):
    grads = []
    for param in net.parameters():
        if grad_grad:
            grads.append(param.grad.grad.view(-1))
        else:
            grads.append(param.grad.view(-1))

    flat_grad = torch.cat(grads)
    return flat_grad

def ensure_shared_grads(model, shared_model,gpu):
    for param, shared_param in zip(model.parameters(),shared_model.parameters()):
        if shared_param.grad is not None:
            if torch.norm(shared_param.grad, p=1) != 0:
                return
        if not gpu:
            shared_param._grad = param.grad
        else:
            if param.grad is not None:
                shared_param._grad = param.grad.cpu()

def slow_sync_param(model, target_model,  tau, gpu):
    for param, target_param in zip(model.parameters(),target_model.parameters()):
        if not gpu:
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        else:
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.cpu().data * tau)


def info_dict_to_array(info):
    '''
    params: dict of ego state(velocity_t, accelearation_t, dist, command, delta_yaw_t, dyaw_dt_t)
    type: np.array
    return: array of size[12,], torch.Tensor (v, a, delta_yaw, dyaw, d, c)
    '''
    velocity_t = info['velocity_t']
    accel_t = info['acceleration_t']
    dist_t = info['dist_to_dest'].reshape((1,1))
    command_t = info['direction']
    delta_yaw_t = np.array(info['delta_yaw_t']).reshape((1,1))
    dyaw_dt_t = np.array(info['dyaw_dt_t']).reshape((1,1))

    info_vec = np.concatenate([velocity_t, accel_t, delta_yaw_t, dyaw_dt_t, dist_t, command_t], axis=0)
    info_vec = info_vec.squeeze()

    return  info_vec
