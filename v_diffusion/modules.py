import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Iterable
from itertools import repeat

sys_version = tuple(sys.version_info)[:3]
if sys_version < (3, 8, 0):
    from functools import reduce
    import operator as op


    def math_prod(x):
        return reduce(op.mul, x, 1)

else:
    math_prod = math.prod


DEFAULT_DTYPE = torch.float32


def lecun_normal_(tensor: nn.Parameter, scale: float = 1.):
    """
    Lecun normal (in-place) initializer
    a specialization of variance scaling initialization
    reference: https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.initializers.lecun_normal.html#jax.nn.initializers.lecun_normal
    """
    assert tensor.ndim >= 2
    nn.init.trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.)  # truncated at 2 std
    fan_in = tensor.shape[1] * (math_prod(tensor.shape[2:]) if tensor.ndim > 2 else 1)
    tensor.data.mul_(math.sqrt(scale / fan_in))
    return tensor


DEFAULT_INITIALIZER = lecun_normal_


def ntuple(n, name="parse"):
    def parse(x):
        if isinstance(x, Iterable):
            return tuple(x)
        else:
            return tuple(repeat(x, n))

    parse.__name__ = name
    return parse


pair = ntuple(2, "pair")


class Linear(nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            init_scale=1.
    ):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), dtype=DEFAULT_DTYPE))
        if bias:
            self.bias = nn.Parameter(torch.empty((out_features,), dtype=DEFAULT_DTYPE))
        else:
            self.register_parameter('bias', None)
        self.init_scale = init_scale
        self.reset_parameters()

    def reset_parameters(self):
        DEFAULT_INITIALIZER(self.weight, scale=self.init_scale)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None)


class Conv2d(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",
            init_scale=1.
    ):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size = pair(kernel_size)
        self.weight = nn.Parameter(
            torch.empty((
                out_channels, in_channels // groups, kernel_size[0], kernel_size[1]
            ), dtype=DEFAULT_DTYPE))
        if bias:
            self.bias = nn.Parameter(torch.empty((out_channels,), dtype=DEFAULT_DTYPE))
        else:
            self.register_parameter("bias", None)
        self.stride = pair(stride)
        self.padding = padding if isinstance(padding, str) else pair(padding)
        self.dilation = pair(dilation)
        self.groups = groups
        self.padding_mode = padding_mode
        self.init_scale = init_scale
        self.reset_parameter()

    def reset_parameter(self):
        DEFAULT_INITIALIZER(self.weight, scale=self.init_scale)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def forward(self, x):
        return F.conv2d(
            x, self.weight, self.bias, stride=self.stride, padding=self.padding,
            dilation=self.dilation, groups=self.groups)


class ValidPad2d(nn.Module):
    def __init__(self, kernel_size, stride, mode="constant", value=0.0):
        super(ValidPad2d, self).__init__()
        self.kernel_size = pair(kernel_size)
        self.stride = pair(stride)
        self.mode = mode
        self.value = value

    def forward(self, x):
        _, _, h, w = x.shape
        (k1, k2), (s1, s2) = self.kernel_size, self.stride
        h_pad, w_pad = s1 * math.ceil((h - k1 + 1) / s1 - 1) + k1 - h, \
                       s2 * math.ceil((w - k2 + 1) / s2 - 1) + k2 - w
        top_pad, bottom_pad = (math.floor(h_pad / 2), math.ceil(h_pad / 2)) if h_pad else (0, 0)
        left_pad, right_pad = (math.floor(w_pad / 2), math.ceil(w_pad / 2)) if w_pad else (0, 0)
        x = F.pad(x, pad=(left_pad, right_pad, top_pad, bottom_pad), mode=self.mode, value=self.value)
        return x


class SamePad2d(nn.Module):
    def __init__(self, kernel_size, stride, mode="constant", value=0.0):
        super(SamePad2d, self).__init__()
        self.kernel_size = pair(kernel_size)
        self.stride = pair(stride)
        self.mode = mode
        self.value = value

    def forward(self, x):
        _, _, h, w = x.shape
        (k1, k2), (s1, s2) = self.kernel_size, self.stride
        h_pad, w_pad = s1 * math.ceil(h / s1 - 1) + k1 - h, s2 * math.ceil(w / s2 - 1) + k2 - w
        top_pad, bottom_pad = (math.floor(h_pad / 2), math.ceil(h_pad / 2)) if h_pad else (0, 0)
        left_pad, right_pad = (math.floor(w_pad / 2), math.ceil(w_pad / 2)) if w_pad else (0, 0)
        x = F.pad(x, pad=(left_pad, right_pad, top_pad, bottom_pad), mode=self.mode, value=self.value)
        return x


class OneHot(nn.Module):
    def __init__(self, num_classes=-1, exclude_zero=False):
        super().__init__()
        self.num_classes = num_classes
        self.exclude_zero = exclude_zero

    def forward(self, x):
        if x.dtype != torch.long:
            x = x.long()
        if self.exclude_zero:
            out = F.one_hot(
                x.sub(1).clamp(min=0),  # non-negative
                num_classes=self.num_classes)
            out[x == 0] = 0
        else:
            out = F.one_hot(x, num_classes=self.num_classes)

        return out.float()


class Sequential(nn.Sequential):
    def forward(self, input, **kwargs):
        for module in self:
            input = module(input, **kwargs)
        return input
