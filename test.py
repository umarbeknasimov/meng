import numpy as np
from models.cifar_resnet import Model
def test(net):
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))

m = Model.get_model_from_name('cifar_resnet_20_24')

print(test(m))