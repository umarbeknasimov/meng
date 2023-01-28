import os

def logger(root): return os.path.join(root, 'logger')


def state_dict(root, step): return os.path.join(root, 'ep{}_it{}.pth'.format(step.ep, step.it))