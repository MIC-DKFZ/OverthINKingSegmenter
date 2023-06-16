import torch
from torch import nn
import torch.nn.functional as F

from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks


class BottleneckClassifier3D(nn.Module):
    def __init__(self, 
                 input_channels: int,
                 conv_bias: bool = False,
                 dropout_op_kwargs: dict = None,
                 nonlin_first: bool = False,
                 num_classes: int = 1,
                 n_convs: int = 2,
                 ):
        super().__init__()
        conv_op = nn.Conv3d
        norm_op = nn.InstanceNorm3d
        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        nonlin = nn.LeakyReLU
        nonlin_kwargs = {'inplace': True}
        dropout_op = nn.Dropout3d if dropout_op_kwargs else None

        self.conv_pre = StackedConvBlocks(n_convs, conv_op, input_channels, input_channels, 1, 1, conv_bias, 
                                          norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, 
                                          nonlin, nonlin_kwargs, nonlin_first)
        self.class_conv = conv_op(input_channels, num_classes, 1, 1, 0, bias=True)

    def forward(self, x):
        x = self.conv_pre(x)
        x = F.adaptive_avg_pool3d(x, 1)
        x = self.class_conv(x).flatten(0)
        return x


class SqEx_Block(nn.Module):
    '''
    This class implements a Squeeze and Excitation block adapted to weight an axis of a 3D image.

    param input_size: The size of the input to the block
    param embed_size: The size of the embedding of the linear layers
    '''
    def __init__(self, input_size: int, embed_size: int, weighting_dim: int = 0):
        super().__init__()
        self.input_size = input_size
        self.embed_size = embed_size
        self.weighting_dim = weighting_dim # The dimension to weight, 0=x, 1=y, 2=z for 3D images

        self.mlp = nn.Sequential(
            nn.Linear(input_size, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''
        This function performs the forward pass of the SE block.

        param x: The input to the block of shape (b, c, x, y, z). In this special case 
        the x dimension represents the paper thickness which are then the "channels"
        of the typical SE block.
        '''
        dims_to_sum = tuple([i for i in range(2, len(x.shape)) if i != self.weighting_dim+2])
        y = torch.mean(x, dim=dims_to_sum) # shape (b, c, x)
        y = self.mlp(y)
        y = y.unsqueeze(dims_to_sum[0]).unsqueeze(dims_to_sum[1]) # shape (b, c, x, 1, 1)
        y = x * y
        x = x + y
        return x


def compute_skip_sizes(input_size, strides):
    """
    Given the patch size and thepooling operations computes the size of the skip connections
    """
    out = [torch.tensor(input_size)]
    for stride in strides:
        if not isinstance(stride, (list,tuple)):
            stride = [stride] * len(input_size)
        out.append(torch.ceil(out[-1] / torch.tensor(stride)).int())
    return out[1:]