# Load Model

import pickle

from foundations.hparams import DatasetHparams, ModelHparams
import datasets.registry
import models.registry
import torch
import torch.nn as nn


def del_attr(obj, names):
    if len(names) == 1:
      delattr(obj, names[0])
    else:
      del_attr(getattr(obj, names[0]), names[1:])

def set_attr(obj, names, val):
    if len(names) == 1:
      setattr(obj, names[0], val)
    else:
      set_attr(getattr(obj, names[0]), names[1:], val)

def make_functional(mod):
    orig_params = []
    # Remove all the parameters in the model
    names = []
    for name, p in list(mod.named_parameters()):
      orig_params.append(p)
      del_attr(mod, name.split("."))
      names.append(name)
    return tuple(orig_params), names

def load_weights(mod, names, params, as_params=False):
    for name, p in zip(names, params):
      set_attr(model, name.split("."), p)



# your forward function with update
def forward(new_params):
    criterion = nn.CrossEntropyLoss()

    load_weights(model, names, new_params)
    model.eval()
    out = model(x)
    loss = criterion(out, y)
    return loss




def vector_from_state_dict(state_dict):
    params_w = []
    for name in names:
        params_w.append(state_dict[name].to(device))
    vec = tuple(params_w)
    return vec

def state_dict_from_vector(flat_vec):
    result_of_mask_0 = tuple(flat_vec.split(tuple(mask_0)))#reshape_vector_to_tensor_tuple(flat_v1, mask_0)
    result_of_mask_1 = [reshape_vector_to_tensor_tuple(result_of_mask_0[i], mask_1[i]) for i in range(len(mask_1))]
    
    
    state_dict = dict()
    for name in names:
        state_dict[name] = result_of_mask_1[names.index(name)]
        
    return state_dict

def flatten_tensor_tuple(tensor_tuple):
    """
    Flattens a tuple of tensors into a 1-dimensional vector.
    
    Args:
    - tensor_tuple: A tuple of torch tensors with different sizes.
    
    Returns:
    - A 1-dimensional torch tensor.
    """
    flattened_tensor = torch.cat([t.flatten() for t in tensor_tuple])
    return flattened_tensor

def reshape_vector_to_tensor_tuple(flattened_tensor, tensor_shape):
    """
    Reshapes a 1-dimensional vector into a tuple of tensors.
    
    Args:
    - flattened_tensor: A 1-dimensional torch tensor.
    - tensor_sizes: A tuple of tensor sizes.
    
    Returns:
    - A tuple of torch tensors with the specified sizes.
    """
    tensor_tuple = flattened_tensor.reshape(tuple(tensor_shape))
    return tensor_tuple

def generate_noise():
    size = 272474
    epsilon = torch.randn(size).to('cuda')
    epsilon /= epsilon.norm()  # Normalize the vector to have unit norm

    # Optional: Scale the vector by a factor to control the magnitude of the norm
    scale_factor = 0.001
    epsilon *= scale_factor
    
    return epsilon

device = torch.device("cuda")

full_batch_dataset_hparams = DatasetHparams(batch_size=128)
full_batch_data_loader = datasets.registry.get(full_batch_dataset_hparams)


model = models.registry.get(ModelHparams(model_name='cifar_resnet_layernorm_20'))

model.to(device)

params, names = make_functional(model)
params = tuple(p.detach().requires_grad_() for p in params)




parent_file = 'model2048.pth'
parent_dict = torch.load(parent_file) 

parent_tuple = vector_from_state_dict(parent_dict)

tensor_sizes = []
for v in parent_tuple: tensor_sizes.append(torch.tensor(v.shape))
mask_1 = tensor_sizes
mask_0 = [torch.prod(size) for size in tensor_sizes]

parent_flat_vectors = flatten_tensor_tuple(parent_tuple)

pt = parent_flat_vectors
result_of_mask_0 = tuple(pt.split(tuple(mask_0)))#reshape_vector_to_tensor_tuple(flat_v1, mask_0)
result_of_mask_1 = [reshape_vector_to_tensor_tuple(result_of_mask_0[i], mask_1[i]) for i in range(len(mask_1))]
pt_tuple = result_of_mask_1


#compute parent_loss

val = 0
for i, train_batch in enumerate(full_batch_data_loader):
    print('Batch = ' + str(i), end='\r')
    x = train_batch[0].to(device)
    y = train_batch[1].to(device)
    val += forward(pt_tuple)/391
parent_loss = val
print(parent_loss)




#run brownian motion

last_pt = parent_flat_vectors
last_loss = parent_loss

pickle_filename = 'brownian_motion_trajectory.pkl'

losses = [last_loss]
pts = [last_pt]


num_iters = 8192

for k in range(num_iters):
    
    eps = generate_noise()
    
    new_pt = last_pt + eps

    result_of_mask_0 = tuple(pt.split(tuple(mask_0)))#reshape_vector_to_tensor_tuple(flat_v1, mask_0)
    result_of_mask_1 = [reshape_vector_to_tensor_tuple(result_of_mask_0[i], mask_1[i]) for i in range(len(mask_1))]
    pt_tuple = result_of_mask_1

    val = 0
    for i, train_batch in enumerate(full_batch_data_loader):
        print('Batch = ' + str(i), end='\r')
        x = train_batch[0].to(device)
        y = train_batch[1].to(device)
        val += forward(pt_tuple)/391
        
    if val < last_loss:
        last_loss = val
        last_pt = new_pt
    
    elif val > last_loss: 

        new_pt = last_pt - eps

        result_of_mask_0 = tuple(pt.split(tuple(mask_0)))#reshape_vector_to_tensor_tuple(flat_v1, mask_0)
        result_of_mask_1 = [reshape_vector_to_tensor_tuple(result_of_mask_0[i], mask_1[i]) for i in range(len(mask_1))]
        pt_tuple = result_of_mask_1

        val = 0
        for i, train_batch in enumerate(full_batch_data_loader):
            print('Batch = ' + str(i), end='\r')
            x = train_batch[0].to(device)
            y = train_batch[1].to(device)
            val += forward(pt_tuple)/391 
    
        if val < last_loss:
            last_loss = val
            last_pt = new_pt
            
    losses.append(last_loss)
    pts.append(last_pt)

    if k%50 == 0:
        print('pickle_dump')
        with open(pickle_filename, 'wb') as f:
            pickle.dump((losses, pts), f)
    
    print(last_loss)