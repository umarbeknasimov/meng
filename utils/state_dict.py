import torch
import models

def flatten_state_dict(state_dict):
    vec = None
    for _, param in state_dict.items():
        if vec is None:
            vec = param.view(-1)
        else:
            vec = torch.cat((vec, param.view(-1)))
    return vec

def flatten_state_dict_w_o_num_batches(state_dict):
    vec = None
    for name, param in state_dict.items():
        if 'num_batches_tracked' in name:
            continue
        if vec is None:
            vec = param.view(-1)
        else:
            vec = torch.cat((vec, param.view(-1)))
    return vec

def flatten_model_params(model):
    vec = None
    for p in model.parameters():
        if vec is None:
            vec = p.view(-1)
        else:
            vec = torch.cat((vec, p.view(-1)))
    return vec

def get_state_dict_w_o_batch_stats(state_dict):
    model = models.frankleResnet20()
    new_state_dict = {}
    for name, param in model.state_dict().items():
        if 'num_batches_tracked' in name or 'mean' in name or 'var' in name:
            new_state_dict[name] = param
        else:
            new_state_dict[name] = state_dict[name]
    return new_state_dict

def get_state_dict_w_o_num_batches_tracked(state_dict):
    model = models.frankleResnet20()
    new_state_dict = {}
    for name, param in model.state_dict().items():
        if 'num_batches_tracked' in name:
            new_state_dict[name] = param
        else:
            new_state_dict[name] = state_dict[name]
    return new_state_dict

def create_state_dict_w_o_batch_stats_from_x(x):
    model = models.frankleResnet20()
    state_dict = {}
    x_start = 0
    for name, param in model.state_dict().items():
        if 'num_batches_tracked' in name or 'mean' in name or 'var' in name:
            state_dict[name] = param
            continue
        param_size = param.size()
        param_idx = 1
        for s in param_size:
            param_idx *= s
        x_part = x[x_start : x_start + param_idx]
        state_dict[name] = torch.Tensor(x_part.reshape(param_size))
        x_start += param_idx
    assert len(x) == x_start
    return state_dict

def create_state_dict_w_o_num_batches_tracked_from_x(x):
    model = models.frankleResnet20()
    state_dict = {}
    x_start = 0
    for name, param in model.state_dict().items():
        if 'num_batches_tracked' in name:
            state_dict[name] = param
            continue
    # for name, param in model.named_parameters():
        param_size = param.size()
        param_idx = 1
        for s in param_size:
            param_idx *= s
        x_part = x[x_start : x_start + param_idx]
        state_dict[name] = torch.Tensor(x_part.reshape(param_size))
        x_start += param_idx
    assert len(x) == x_start
    return state_dict