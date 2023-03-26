import torch
import torch.nn as nn
import torch.nn.functional as F
from foundations import hparams
from training.desc import TrainingDesc
from . import registry, cifar_resnet

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            return F.layer_norm(x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)

def permute_and_apply(x, norm):
    return norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

class Model(nn.Module):
    """A residual neural network as originally designed for CIFAR-10."""

    class Block(nn.Module):
        """A ResNet block."""

        def __init__(self, f_in: int, f_out: int, downsample=False):
            super(Model.Block, self).__init__()

            stride = 2 if downsample else 1
            self.conv1 = nn.Conv2d(f_in, f_out, kernel_size=3, stride=stride, padding=1, bias=False)
            self.n1 = nn.LayerNorm(f_out)
            self.conv2 = nn.Conv2d(f_out, f_out, kernel_size=3, stride=1, padding=1, bias=False)
            self.n2 = nn.LayerNorm(f_out)

            # No parameters for shortcut connections.
            if downsample or f_in != f_out:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(f_in, f_out, kernel_size=1, stride=2, bias=False),
                    nn.LayerNorm(f_out)
                    )
            else:
                self.shortcut = nn.Sequential(
                    nn.Sequential(),
                    nn.Sequential()
                )

        def forward(self, x):
            out = self.conv1(x)
            out = F.relu(permute_and_apply(out, self.n1))
            out = self.conv2(out)
            out = permute_and_apply(out, self.n2)
            shortcut = permute_and_apply(self.shortcut[0](x), self.shortcut[1])
            out += shortcut
            return F.relu(out)

    def __init__(self, plan, outputs=None):
        super(Model, self).__init__()
        outputs = outputs or 10

        # Initial convolution.
        current_filters = plan[0][0]
        self.conv = nn.Conv2d(3, current_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.n = nn.LayerNorm(current_filters)

        # The subsequent blocks of the ResNet.
        blocks = []
        for segment_index, (filters, num_blocks) in enumerate(plan):
            for block_index in range(num_blocks):
                downsample = segment_index > 0 and block_index == 0
                blocks.append(Model.Block(current_filters, filters, downsample))
                current_filters = filters

        self.blocks = nn.Sequential(*blocks)

        # Final fc layer. Size = number of filters in last segment.
        self.fc = nn.Linear(plan[-1][0], outputs)
        self.criterion = nn.CrossEntropyLoss()

        # Initialize.
        self.apply(registry.init_fn)

    def forward(self, x):
        out = F.relu(permute_and_apply(self.conv(x), self.n))
        out = self.blocks(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
    @staticmethod
    def is_valid_model_name(model_name):
        return "layernorm2" in model_name and cifar_resnet.Model.is_valid_model_name(model_name.replace('layernorm2_', ''))
    
    @staticmethod
    def get_model_from_name(model_name,  outputs=10):
        """The naming scheme for a ResNet is 'cifar_resnet_N[_W]'.
        """
        if not Model.is_valid_model_name(model_name):
            raise ValueError('Invalid model name: {}'.format(model_name))
        
        model_name = model_name.replace('layernorm2_', '')

        name = model_name.split('_')
        W = 16 if len(name) == 3 else int(name[3])
        D = int(name[2])
        if (D - 2) % 3 != 0:
            raise ValueError('Invalid ResNet depth: {}'.format(D))
        D = (D - 2) // 6
        plan = [(W, D), (2*W, D), (4*W, D)]

        return Model(plan, outputs)
    
    @staticmethod
    def default_hparams():
        model_hparams = hparams.ModelHparams(
            model_name='cifar_resnet_layernorm2_20'
        )

        dataset_hparams = hparams.DatasetHparams(
            batch_size=128
        )

        training_hparams = hparams.TrainingHparams(
            momentum=0.9,
            milestone_steps=None, #'80ep,120ep',
            lr=0.1,
            gamma=0.1,
            weight_decay=1e-4,
            training_steps='160ep'
        )

        return TrainingDesc(model_hparams, dataset_hparams, training_hparams)
    
    @property
    def loss_criterion(self):
        return nn.CrossEntropyLoss()