import torch

def flatten_state_dict_w_o_batch_stats(state_dict):
    vec = None
    for name, param in state_dict.items():
        if 'num_batches_tracked' in name or 'mean' in name or 'var' in name:
            continue
        if vec is None:
            vec = param.view(-1)
        else:
            vec = torch.cat((vec, param.view(-1)))
    return vec

def get_state_dict_wo_batch_stats(reference_model, state_dict):
    new_state_dict = {}
    for name, param in reference_model.state_dict().items():
        if 'num_batches_tracked' in name or 'mean' in name or 'var' in name:
            new_state_dict[name] = param
        else:
            new_state_dict[name] = state_dict[name]
    return new_state_dict

def create_state_dict_w_o_batch_stats_from_x(reference_model, x):
    state_dict = {}
    x_start = 0
    for name, param in reference_model.state_dict().items():
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
    assert len(x) == x_start, f'{len(x)} != {x_start}'
    return state_dict