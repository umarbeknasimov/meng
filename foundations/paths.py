import os

def logger(root): return os.path.join(root, 'logger')


def state_dict(type, root, step): return os.path.join(root, '{}_ep{}_it{}.pth'.format(type, step.ep, step.it))