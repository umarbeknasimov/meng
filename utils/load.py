import torch
import models

def load(filename, device):
    return torch.load(filename, map_location=device)

def model_with_state(state, device):
    model = models.frankleResnet20()
    model = model.to(device)
    model.load_state_dict(state)
    return model

def load_model_with_state(filename, device):
    weights = load(filename, device)
    return model_with_state(weights, device)