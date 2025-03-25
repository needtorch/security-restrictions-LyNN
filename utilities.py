from __future__ import division, print_function
# from asyncio.windows_events import NULL

import sys
import os
import importlib
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F

# import tensorflow._api.v2.compat.v1 as tf1
# import tensorflow as tf
# tf1.disable_v2_behavior()
from scipy import signal
from matplotlib.colors import ListedColormap

from safe_learning import config, DeterministicFunction, GridWorld
from safe_learning.utilities_torch import concatenate_inputs

if sys.version_info.major == 2:
    import imp
from scipy import stats

__all__ = ['import_from_directory', 'LyapunovNetwork']

NP_DTYPE = config.np_dtype
PY_DTYPE = config.dtype

def import_from_directory(library, path):
    """Import a library from a directory outside the path.

    Parameters
    ----------
    library: string
        The name of the library.
    path: string
        The path of the folder containing the library.

    """
    try:
        return importlib.import_module(library)
    except ImportError:
        module_path = os.path.abspath(path)
        version = sys.version_info

        if version.major == 2:
            f, filename, desc = imp.find_module(library, [module_path])
            return imp.load_module(library, f, filename, desc)
        else:
            sys.path.append(module_path)
            return importlib.import_module(library)




# class LyapunovNetwork(DeterministicFunction):
#     """A positive-definite neural network using PyTorch."""
#
#     def __init__(self, input_dim, layer_dims, activations, eps=1e-1,
#                  initializer=torch.nn.init.xavier_normal_, name='lyapunov_network'):
#         super(LyapunovNetwork, self).__init__()
#         self.input_dim = input_dim
#         self.num_layers = len(layer_dims)
#         self.activations = activations
#         self.eps = eps
#         self.name = name
#         self.initializer = initializer
#
#         if layer_dims[0] < input_dim:
#             raise ValueError('The first layer dimension must be at least the input dimension!')
#
#         if np.all(np.diff(layer_dims) >= 0):
#             self.output_dims = layer_dims
#         else:
#             raise ValueError('Each layer must maintain or increase the dimension of its input!')
#
#         self.layers = nn.ModuleList()
#         self.kernels = nn.ModuleList()
#         # for i, (in_dim, out_dim) in enumerate(zip([input_dim] + layer_dims[:-1], layer_dims)):
#         #     layer = nn.Linear(in_dim, out_dim)
#         #     self.initializer(layer.weight)
#         #     self.layers.append(layer)
#
#         self.hidden_dims = torch.zeros(self.num_layers, dtype=int)
#         for i in range(self.num_layers):
#             if i == 0:
#                 layer_input_dim = self.input_dim  # 在l_f_l实例中为8, 64, 64
#             else:
#                 layer_input_dim = self.output_dims[i - 1]
#             # hidden_dims为2, 33, 33
#             self.hidden_dims[i] = np.ceil((layer_input_dim + 1) / 2).astype(int)  # ceil向上取整
#             layer = nn.Linear(layer_input_dim, self.hidden_dims[i]).type(PY_DTYPE)
#             self.initializer(layer.weight)
#             self.layers.append(layer)
#
#             for i in range(self.num_layers):
#                 if i == 0:
#                     layer_input_dim = self.input_dim
#                 else:
#                     layer_input_dim = self.output_dims[i - 1]
#                 W = self.layers[i].weight
#                 kernel = torch.mm(W.t(), W) + self.eps * torch.eye(layer_input_dim, dtype=PY_DTYPE, requires_grad=False)
#                 dim_diff = self.output_dims[i] - layer_input_dim
#                 if dim_diff > 0:
#                     # if self.layers[i].out_features > self.layers[i].in_features:
#                     layer_ex = nn.Linear(layer_input_dim, dim_diff).type(PY_DTYPE)
#                     self.initializer(layer_ex.weight)
#                     W_extra = layer_ex.weight
#                     kernel = torch.cat([kernel, W_extra], dim=0)
#                 self.kernels.append(kernel)
#                 layer_output = torch.mm(net, kernel.t())
#                 net = self.activations[i](layer_output)
#             values = torch.sum(net ** 2, dim=1, keepdim=True)
#             return values
#
#
#
#             kernel = torch.matmul(self.layers[i].t(), self.layer[i]).type(PY_DTYPE) + self.eps * torch.eye(layer_input_dim, dtype=PY_DTYPE)
#             dim_diff = self.output_dims[i] - layer_input_dim
#             if dim_diff > 0:
#                 W = nn.Linear(layer_input_dim, dim_diff).type(PY_DTYPE)
#                 self.initializer(W.weight)
#                 kernel = torch.cat((kernel, W), dim=0).type(PY_DTYPE)
#             self.initializer(kernel)
#             self.kernels.append(kernel)
#         # self.output_layer = nn.Linear(layer_dims[-1], 1, bias=False)
#         # self.initializer(self.output_layer.weight)
#
#     def forward(self, x):
#         net = x.view(-1, self.input_dim)
#
#         if isinstance(net, np.array):
#             net = torch.constant(net)
#
#         for i, kernel in enumerate(self.kernels):
#             self.output_layer = nn.Linear(net, kernel[i].t(), bias=False)
#             net = self.activations[i](self.output_layer)
#         values = torch.sum(torch.square(net), dim=1, keepdims=True)
#         return values

# class LyapunovNetwork(DeterministicFunction):
#     """A positive-definite neural network using PyTorch."""
#
#     def __init__(self, input_dim, layer_dims, activations, eps=1e-1,
#                  initializer=torch.nn.init.xavier_normal_, name='lyapunov_network'):
#         super(LyapunovNetwork, self).__init__()
#         self.input_dim = input_dim
#         self.num_layers = len(layer_dims)
#         self.activations = activations
#         self.eps = eps
#         self.name = name
#         self.initializer = initializer
#
#         if layer_dims[0] < input_dim:
#             raise ValueError('The first layer dimension must be at least the input dimension!')
#
#         if np.all(np.diff(layer_dims) >= 0):
#             self.output_dims = layer_dims
#         else:
#             raise ValueError('Each layer must maintain or increase the dimension of its input!')
#
#         self.layers = nn.ModuleList()
#         self.kernels = nn.ModuleList()
#         # for i, (in_dim, out_dim) in enumerate(zip([input_dim] + layer_dims[:-1], layer_dims)):
#         #     layer = nn.Linear(in_dim, out_dim)
#         #     self.initializer(layer.weight)
#         #     self.layers.append(layer)
#
#         self.hidden_dims = torch.zeros(self.num_layers, dtype=int)
#         for i in range(self.num_layers):
#             if i == 0:
#                 layer_input_dim = self.input_dim  # 在l_f_l实例中为8, 64, 64
#             else:
#                 layer_input_dim = self.output_dims[i - 1]
#             # hidden_dims为2, 33, 33
#             self.hidden_dims[i] = np.ceil((layer_input_dim + 1) / 2).astype(int)  # ceil向上取整
#             layer = nn.Linear(layer_input_dim, self.hidden_dims[i]).type(PY_DTYPE)
#             self.initializer(layer.weight)
#             self.layers.append(layer)
#             # kernel = torch.matmul(self.layers[i].t(), self.layer[i]).type(PY_DTYPE) + self.eps * torch.eye(layer_input_dim, dtype=PY_DTYPE)
#             # dim_diff = self.output_dims[i] - layer_input_dim
#             # if dim_diff > 0:
#             #     W = nn.Linear(layer_input_dim, dim_diff).type(PY_DTYPE)
#             #     self.initializer(W.weight)
#             #     kernel = torch.cat((kernel, W), dim=0).type(PY_DTYPE)
#             # self.initializer(kernel)
#             # self.kernels.append(kernel)
#         # self.output_layer = nn.Linear(layer_dims[-1], 1, bias=False)
#         # self.initializer(self.output_layer.weight)
#
#     def forward(self, x):
#         net = x.reshape(-1, self.input_dim)
#         # net = torch.reshape(x, (-1, self.input_dim))
#
#         # if isinstance(net, np.ndarray):
#         #     net = torch.constant(net)
#         if isinstance(net, np.ndarray):
#             net = torch.from_numpy(net)
#         for i in range(self.num_layers):
#             if i == 0:
#                 layer_input_dim = self.input_dim
#             else:
#                 layer_input_dim = self.output_dims[i - 1]
#             W = self.layers[i].weight
#             kernel = torch.mm(W.t(), W) + self.eps * torch.eye(layer_input_dim, dtype=PY_DTYPE,requires_grad=False)
#             dim_diff = self.output_dims[i] - layer_input_dim
#             if dim_diff > 0:
#             # if self.layers[i].out_features > self.layers[i].in_features:
#                 layer_ex = nn.Linear(layer_input_dim,dim_diff).type(PY_DTYPE)
#                 self.initializer(layer_ex.weight)
#                 W_extra=layer_ex.weight
#                 kernel = torch.cat([kernel, W_extra], dim=0)
#             layer_output = torch.mm(net, kernel.t())
#             net = self.activations[i](layer_output)
#         values = torch.sum(net**2, dim=1, keepdim=True)
#         return values
#     def parameters(self):
#         return self.kernels

# class LyapunovNetwork(nn.Module):
#     """A positive-definite neural network using PyTorch."""
#
#     def __init__(self, input_dim, layer_dims, activations, eps=1e-1,
#                  initializer=torch.nn.init.xavier_normal_, name='lyapunov_network'):
#         super(LyapunovNetwork, self).__init__()
#         self.kernel_NN = None
#         self.input_dim = input_dim
#         self.num_layers = len(layer_dims)
#         self.activations = activations
#         self.eps = eps
#         self.name = name
#         self.initializer = initializer
#
#         if layer_dims[0] < input_dim:
#             raise ValueError('The first layer dimension must be at least the input dimension!')
#
#         if np.all(np.diff(layer_dims) >= 0):
#             self.output_dims = layer_dims
#         else:
#             raise ValueError('Each layer must maintain or increase the dimension of its input!')
#
#         self.layers = nn.ModuleList()
#         self.kernels = nn.ModuleList()
#         # for i, (in_dim, out_dim) in enumerate(zip([input_dim] + layer_dims[:-1], layer_dims)):
#         #     layer = nn.Linear(in_dim, out_dim)
#         #     self.initializer(layer.weight)
#         #     self.layers.append(layer)
#
#         self.hidden_dims = torch.zeros(self.num_layers, dtype=int)
#         for i in range(self.num_layers):
#             if i == 0:
#                 layer_input_dim = self.input_dim  # 在l_f_l实例中为8, 64, 64
#             else:
#                 layer_input_dim = self.output_dims[i - 1]
#             # hidden_dims为2, 33, 33
#             self.hidden_dims[i] = np.ceil((layer_input_dim + 1) / 2).astype(int)  # ceil向上取整
#             layer = nn.Linear(layer_input_dim, self.hidden_dims[i]).type(PY_DTYPE)
#             self.initializer(layer.weight)
#             self.layers.append(layer)
#             self.layers[i]
#             # kernel = torch.matmul(self.layers[i].t(), self.layer[i]).type(PY_DTYPE) + self.eps * torch.eye(layer_input_dim, dtype=PY_DTYPE)
#             # dim_diff = self.output_dims[i] - layer_input_dim
#             # if dim_diff > 0:
#             #     W = nn.Linear(layer_input_dim, dim_diff).type(PY_DTYPE)
#             #     self.initializer(W.weight)
#             #     kernel = torch.cat((kernel, W), dim=0).type(PY_DTYPE)
#             # self.initializer(kernel)
#             # self.kernels.append(kernel)
#         # self.output_layer = nn.Linear(layer_dims[-1], 1, bias=False)
#         # self.initializer(self.output_layer.weight)
#
#     def forward(self, x):
#         net = x.reshape(-1, self.input_dim)
#         # net = torch.reshape(x, (-1, self.input_dim))
#
#         # if isinstance(net, np.ndarray):
#         #     net = torch.constant(net)
#         if isinstance(net, np.ndarray):
#             net = torch.from_numpy(net)
#         for i in range(self.num_layers):
#             if i == 0:
#                 layer_input_dim = self.input_dim
#             else:
#                 layer_input_dim = self.output_dims[i - 1]
#             W = self.layers[i].weight
#             kernel_shape=torch.mm(W.t(), W).shape
#             # kernel = torch.tensor(torch.mm(W.t(), W) + self.eps * torch.eye(layer_input_dim, dtype=PY_DTYPE,requires_grad=True),requires_grad=True)
#             kernel = nn.Linear(kernel_shape[1],kernel_shape[0]).type(PY_DTYPE)
#             self.initializer(kernel.weight)
#             dim_diff = self.output_dims[i] - layer_input_dim
#             if dim_diff > 0:
#             # if self.layers[i].out_features > self.layers[i].in_features:
#                 layer_ex = nn.Linear(layer_input_dim,dim_diff).type(PY_DTYPE)
#                 self.initializer(layer_ex.weight)
#                 W_extra=layer_ex.weight
#                 kernel_shape=torch.cat([kernel.weight, W_extra], dim=0).shape
#                 kernel = nn.Linear(kernel_shape[1],kernel_shape[0]).type(PY_DTYPE)
#                 self.initializer(kernel.weight)
#             # self.kernel_NN=nn.parameter.Parameter(kernel)
#             # self.kernels.append(kernel)
#             layer_output = torch.mm(net, kernel.weight.t())
#             net = self.activations[i](layer_output)
#         values = torch.sum(net**2, dim=1, keepdim=True)
#         return values

# class LyapunovNetwork(nn.Module):
#     """A positive-definite neural network using PyTorch."""
#
#     def __init__(self, input_dim, layer_dims, activations, eps=1e-1,
#                  initializer=torch.nn.init.xavier_normal_, name='lyapunov_network'):
#         super(LyapunovNetwork, self).__init__()
#         self.kernel_NN = None
#         self.input_dim = input_dim
#         self.num_layers = len(layer_dims)
#         self.activations = activations
#         self.eps = eps
#         self.name = name
#         self.initializer = initializer
#
#         if layer_dims[0] < input_dim:
#             raise ValueError('The first layer dimension must be at least the input dimension!')
#
#         if np.all(np.diff(layer_dims) >= 0):
#             self.output_dims = layer_dims
#         else:
#             raise ValueError('Each layer must maintain or increase the dimension of its input!')
#
#         # self.layers = nn.ModuleList()
#         self.kernels = nn.ModuleList()
#         # for i, (in_dim, out_dim) in enumerate(zip([input_dim] + layer_dims[:-1], layer_dims)):
#         #     layer = nn.Linear(in_dim, out_dim)
#         #     self.initializer(layer.weight)
#         #     self.layers.append(layer)
#
#         self.hidden_dims = torch.zeros(self.num_layers, dtype=int,requires_grad=False)
#         for i in range(self.num_layers):
#             if i == 0:
#                 layer_input_dim = self.input_dim  # 在l_f_l实例中为8, 64, 64
#             else:
#                 layer_input_dim = self.output_dims[i - 1]
#             # hidden_dims为2, 33, 33
#             self.hidden_dims[i] = np.ceil((layer_input_dim + 1) / 2).astype(int)  # ceil向上取整
#             # layer = nn.Linear(layer_input_dim, self.hidden_dims[i]).type(PY_DTYPE)
#             # self.initializer(layer.weight)
#             W=torch.rand((self.hidden_dims[i],layer_input_dim),dtype=PY_DTYPE,requires_grad=False)
#             # self.initializer(W)
#             # W = layer.weight
#             kernel_shape = torch.mm(W.t(), W).shape
#             # kernel = torch.tensor(torch.mm(W.t(), W) + self.eps * torch.eye(layer_input_dim, dtype=PY_DTYPE,requires_grad=True),requires_grad=True)
#             kernel = nn.Linear(kernel_shape[1], kernel_shape[0]).type(PY_DTYPE)
#             dim_diff = self.output_dims[i] - layer_input_dim
#             if dim_diff > 0:
#                 # if self.layers[i].out_features > self.layers[i].in_features:
#                 # layer_ex = nn.Linear(layer_input_dim, dim_diff).type(PY_DTYPE)
#                 # self.initializer(layer_ex.weight)
#                 W_extra = torch.rand((dim_diff,layer_input_dim),dtype=PY_DTYPE,requires_grad=False)
#                 # self.initializer(W_extra)
#                 kernel_shape = torch.cat([kernel.weight, W_extra], dim=0).shape
#                 kernel = nn.Linear(kernel_shape[1], kernel_shape[0]).type(PY_DTYPE)
#             # kernel.weight.data=kernel.weight.data.double()
#             # kernel.bias.data.fill_(self.eps)
#             self.initializer(kernel.weight)
#             if kernel.weight.shape[1] == 4:
#                 kernel.weight.data.add_(self.eps * torch.eye(256, 4, dtype=torch.float64))
#             else:
#                 kernel.weight.data.add_(self.eps * torch.eye(256, dtype=torch.float64))
#             self.kernels.append(kernel)
#             # kernel = torch.matmul(self.layers[i].t(), self.layer[i]).type(PY_DTYPE) + self.eps * torch.eye(layer_input_dim, dtype=PY_DTYPE)
#             # dim_diff = self.output_dims[i] - layer_input_dim
#             # if dim_diff > 0:
#             #     W = nn.Linear(layer_input_dim, dim_diff).type(PY_DTYPE)
#             #     self.initializer(W.weight)
#             #     kernel = torch.cat((kernel, W), dim=0).type(PY_DTYPE)
#             # self.initializer(kernel)
#             # self.kernels.append(kernel)
#         # self.output_layer = nn.Linear(layer_dims[-1], 1, bias=False)
#         # self.initializer(self.output_layer.weight)
#
#     def forward(self, x):
#         # net = x.reshape(-1, self.input_dim)
#         # net = torch.reshape(x, (-1, self.input_dim))
#
#         # if isinstance(net, np.ndarray):
#         #     net = torch.constant(net)
#         # if isinstance(x, np.ndarray):
#         #     x = torch.from_numpy(x)
#         for i in range(self.num_layers):
#             layer_output = self.kernels[i](x)
#             x = torch.tanh(layer_output)
#         x_sq = torch.square(x)
#         values = torch.sum(x_sq, dim=1, keepdim=True)
#         return values

# class LyapunovNetwork(nn.Module):
#     """A positive-definite neural network using PyTorch."""
#
#     def __init__(self, input_dim, layer_dims, activations, eps=1e-1,
#                  initializer=torch.nn.init.xavier_normal_, name='lyapunov_network'):
#         super(LyapunovNetwork, self).__init__()
#         self.input_dim = input_dim
#         self.num_layers = len(layer_dims)
#         self.activations = activations
#         self.eps = eps
#         self.name = name
#         self.initializer = initializer
#
#         if layer_dims[0] < input_dim:
#             raise ValueError('The first layer dimension must be at least the input dimension!')
#
#         if np.all(np.diff(layer_dims) >= 0):
#             self.output_dims = layer_dims
#         else:
#             raise ValueError('Each layer must maintain or increase the dimension of its input!')
#
#         self.kernel_0 = nn.Linear(4, 256).type(PY_DTYPE)
#         self.kernel_1 = nn.Linear(256, 256).type(PY_DTYPE)
#         self.kernel_2 = nn.Linear(256, 256).type(PY_DTYPE)
#
#         self.initializer(self.kernel_0.weight)
#         self.initializer(self.kernel_1.weight)
#         self.initializer(self.kernel_2.weight)
#
#         self.kernel_0.weight.data.add_(self.eps * torch.eye(256, 4, dtype=torch.float64))
#         self.kernel_1.weight.data.add_(self.eps * torch.eye(256, dtype=torch.float64))
#         self.kernel_2.weight.data.add_(self.eps * torch.eye(256, dtype=torch.float64))
#
#         # self.layers = nn.ModuleList()
#         # self.kernels = nn.ModuleList()
#         # # for i, (in_dim, out_dim) in enumerate(zip([input_dim] + layer_dims[:-1], layer_dims)):
#         # #     layer = nn.Linear(in_dim, out_dim)
#         # #     self.initializer(layer.weight)
#         # #     self.layers.append(layer)
#         #
#         # self.hidden_dims = torch.zeros(self.num_layers, dtype=int,requires_grad=False)
#         # for i in range(self.num_layers):
#         #     if i == 0:
#         #         layer_input_dim = self.input_dim  # 在l_f_l实例中为8, 64, 64
#         #     else:
#         #         layer_input_dim = self.output_dims[i - 1]
#         #     # hidden_dims为2, 33, 33
#         #     self.hidden_dims[i] = np.ceil((layer_input_dim + 1) / 2).astype(int)  # ceil向上取整
#         #     # layer = nn.Linear(layer_input_dim, self.hidden_dims[i]).type(PY_DTYPE)
#         #     # self.initializer(layer.weight)
#         #     W=torch.rand((self.hidden_dims[i],layer_input_dim),dtype=PY_DTYPE,requires_grad=False)
#         #     # self.initializer(W)
#         #     # W = layer.weight
#         #     kernel_shape = torch.mm(W.t(), W).shape
#         #     # kernel = torch.tensor(torch.mm(W.t(), W) + self.eps * torch.eye(layer_input_dim, dtype=PY_DTYPE,requires_grad=True),requires_grad=True)
#         #     kernel = nn.Linear(kernel_shape[1], kernel_shape[0]).type(PY_DTYPE)
#         #     dim_diff = self.output_dims[i] - layer_input_dim
#         #     if dim_diff > 0:
#         #         # if self.layers[i].out_features > self.layers[i].in_features:
#         #         # layer_ex = nn.Linear(layer_input_dim, dim_diff).type(PY_DTYPE)
#         #         # self.initializer(layer_ex.weight)
#         #         W_extra = torch.rand((dim_diff,layer_input_dim),dtype=PY_DTYPE,requires_grad=False)
#         #         # self.initializer(W_extra)
#         #         kernel_shape = torch.cat([kernel.weight, W_extra], dim=0).shape
#         #         kernel = nn.Linear(kernel_shape[1], kernel_shape[0]).type(PY_DTYPE)
#         #     self.initializer(kernel.weight)
#         #     self.kernels.append(kernel)
#         #     # kernel = torch.matmul(self.layers[i].t(), self.layer[i]).type(PY_DTYPE) + self.eps * torch.eye(layer_input_dim, dtype=PY_DTYPE)
#         #     # dim_diff = self.output_dims[i] - layer_input_dim
#         #     # if dim_diff > 0:
#         #     #     W = nn.Linear(layer_input_dim, dim_diff).type(PY_DTYPE)
#         #     #     self.initializer(W.weight)
#         #     #     kernel = torch.cat((kernel, W), dim=0).type(PY_DTYPE)
#         #     # self.initializer(kernel)
#         #     # self.kernels.append(kernel)
#         # # self.output_layer = nn.Linear(layer_dims[-1], 1, bias=False)
#         # # self.initializer(self.output_layer.weight)
#
#     def forward(self, x):
#         # net = x.reshape(-1, self.input_dim)
#         # net = torch.reshape(x, (-1, self.input_dim))
#
#         # if isinstance(net, np.ndarray):
#         #     net = torch.constant(net)
#         # if isinstance(net, np.ndarray):
#         #     net = torch.from_numpy(net)
#         output = self.kernel_0(x)
#         x = torch.tanh(output)
#         output = self.kernel_1(x)
#         x = torch.tanh(output)
#         output = self.kernel_2(x)
#         x = torch.tanh(output)
#         net_sq = torch.square(x)
#         values = torch.sum(net_sq, dim=1, keepdim=True)
#
#
#         # for i in range(self.num_layers):
#         #     layer_output = self.kernels[i](net)
#         #     net = torch.tanh(layer_output)
#         # values = torch.sum(net**2, dim=1, keepdim=True)
#         return values

class LyapunovNetwork(nn.Module):
    """A positive-definite neural network using PyTorch."""

    def __init__(self, input_dim, layer_dims, activations, eps=1e-1,
                 initializer=torch.nn.init.xavier_normal_, name='lyapunov_network'):
        super(LyapunovNetwork, self).__init__()
        self.input_dim = int(input_dim)
        self.num_layers = len(layer_dims)
        self.activations = activations
        self.eps = eps
        self.name = name
        self.initializer = initializer

        if layer_dims[0] < input_dim:
            raise ValueError('The first layer dimension must be at least the input dimension!')

        if np.all(np.diff(layer_dims) >= 0):
            self.output_dims = layer_dims
        else:
            raise ValueError('Each layer must maintain or increase the dimension of its input!')

        self.kernel_0 = nn.Linear(self.input_dim, 256).type(PY_DTYPE)
        self.kernel_1 = nn.Linear(256, 256).type(PY_DTYPE)
        self.kernel_2 = nn.Linear(256, 256).type(PY_DTYPE)

        self.initializer(self.kernel_0.weight)
        self.initializer(self.kernel_1.weight)
        self.initializer(self.kernel_2.weight)

        self.kernel_0.weight.data.add_(self.eps * torch.eye(256, self.input_dim, dtype=torch.float64))
        self.kernel_1.weight.data.add_(self.eps * torch.eye(256, dtype=torch.float64))
        self.kernel_2.weight.data.add_(self.eps * torch.eye(256, dtype=torch.float64))

        # self.layers = nn.ModuleList()
        # self.kernels = nn.ModuleList()
        # # for i, (in_dim, out_dim) in enumerate(zip([input_dim] + layer_dims[:-1], layer_dims)):
        # #     layer = nn.Linear(in_dim, out_dim)
        # #     self.initializer(layer.weight)
        # #     self.layers.append(layer)
        #
        # self.hidden_dims = torch.zeros(self.num_layers, dtype=int,requires_grad=False)
        # for i in range(self.num_layers):
        #     if i == 0:
        #         layer_input_dim = self.input_dim  # 在l_f_l实例中为8, 64, 64
        #     else:
        #         layer_input_dim = self.output_dims[i - 1]
        #     # hidden_dims为2, 33, 33
        #     self.hidden_dims[i] = np.ceil((layer_input_dim + 1) / 2).astype(int)  # ceil向上取整
        #     # layer = nn.Linear(layer_input_dim, self.hidden_dims[i]).type(PY_DTYPE)
        #     # self.initializer(layer.weight)
        #     W=torch.rand((self.hidden_dims[i],layer_input_dim),dtype=PY_DTYPE,requires_grad=False)
        #     # self.initializer(W)
        #     # W = layer.weight
        #     kernel_shape = torch.mm(W.t(), W).shape
        #     # kernel = torch.tensor(torch.mm(W.t(), W) + self.eps * torch.eye(layer_input_dim, dtype=PY_DTYPE,requires_grad=True),requires_grad=True)
        #     kernel = nn.Linear(kernel_shape[1], kernel_shape[0]).type(PY_DTYPE)
        #     dim_diff = self.output_dims[i] - layer_input_dim
        #     if dim_diff > 0:
        #         # if self.layers[i].out_features > self.layers[i].in_features:
        #         # layer_ex = nn.Linear(layer_input_dim, dim_diff).type(PY_DTYPE)
        #         # self.initializer(layer_ex.weight)
        #         W_extra = torch.rand((dim_diff,layer_input_dim),dtype=PY_DTYPE,requires_grad=False)
        #         # self.initializer(W_extra)
        #         kernel_shape = torch.cat([kernel.weight, W_extra], dim=0).shape
        #         kernel = nn.Linear(kernel_shape[1], kernel_shape[0]).type(PY_DTYPE)
        #     self.initializer(kernel.weight)
        #     self.kernels.append(kernel)
        #     # kernel = torch.matmul(self.layers[i].t(), self.layer[i]).type(PY_DTYPE) + self.eps * torch.eye(layer_input_dim, dtype=PY_DTYPE)
        #     # dim_diff = self.output_dims[i] - layer_input_dim
        #     # if dim_diff > 0:
        #     #     W = nn.Linear(layer_input_dim, dim_diff).type(PY_DTYPE)
        #     #     self.initializer(W.weight)
        #     #     kernel = torch.cat((kernel, W), dim=0).type(PY_DTYPE)
        #     # self.initializer(kernel)
        #     # self.kernels.append(kernel)
        # # self.output_layer = nn.Linear(layer_dims[-1], 1, bias=False)
        # # self.initializer(self.output_layer.weight)

    def forward(self, x):
        # net = x.reshape(-1, self.input_dim)
        # net = torch.reshape(x, (-1, self.input_dim))

        # if isinstance(net, np.ndarray):
        #     net = torch.constant(net)
        # if isinstance(net, np.ndarray):
        #     net = torch.from_numpy(net)
        output = self.kernel_0(x)
        x = torch.tanh(output)
        output = self.kernel_1(x)
        x = torch.tanh(output)
        output = self.kernel_2(x)
        x = torch.tanh(output)
        net_sq = torch.square(x)
        values = torch.sum(net_sq, dim=1, keepdim=True)


        # for i in range(self.num_layers):
        #     layer_output = self.kernels[i](net)
        #     net = torch.tanh(layer_output)
        # values = torch.sum(net**2, dim=1, keepdim=True)
        return values

def balanced_class_weights(y_true, scale_by_total=True):
    """Compute class weights from class label counts.
        从类标签计数计算类权重
    """
    y = y_true.astype(np.bool_)
    nP = y.sum()
    nN = y.size - y.sum()
    class_counts = np.array([nN, nP])

    weights = np.ones_like(y, dtype=float)
    weights[y] /= nP
    weights[~y] /= nN
    if scale_by_total:
        weights *= y.size

    return weights, class_counts


# class Dynamics_NN(object):
#     def __init__(self, states_before, states_after):
#         super(Dynamics_NN, self).__init__()
#         # states_before[:, :2] = states_before[:, :2]/100
#         # states_after[:, :2] = states_after[:, :2]/100
#         self.states_before = states_before
#         self.states_after = states_after
#
#     def build_evaluation(self, states):
#         index = []
#         dim = self.states_after.shape[1]
#         states = states.reshape(-1, dim)
#         for i in range(states.shape[0]):
#             if np.where(self.states_before[:, :dim]==states[i])[0].shape[0] == 0:
#                 index.append(stats.mode(np.where(self.states_after[:, :dim]==states[i])[0])[0][0])
#             else:
#                 index.append(stats.mode(np.where(self.states_before[:, :dim]==states[i])[0])[0][0])
#         index = np.array(index)
#         return self.states_after[index]

class Dynamics_NN(object):
    def __init__(self, states_before, states_after):
        super(Dynamics_NN, self).__init__()
        # states_before[:, :2] = states_before[:, :2]/100
        # states_after[:, :2] = states_after[:, :2]/100
        self.states_before = states_before
        self.states_after = states_after

    def build_evaluation(self, states):
        index = []
        dim = self.states_after.shape[1]
        states = states.reshape(-1, dim)
        # for i in range(states.shape[0]):
        #     if np.where(self.states_before[:, :dim]==states[i])[0].shape[0] == 0:
        #         index.append(stats.mode(np.where(self.states_after[:, :dim]==states[i])[0])[0][0])
        #     else:
        #         index.append(stats.mode(np.where(self.states_before[:, :dim]==states[i])[0])[0][0])
        for i in range(states.shape[0]):
            if states[i][-1] == 1000.0:
                index.append(999.0)
            elif (states[i][-1] == self.states_after[-1][-1]):
                index.append(states[i][-1] - 1.0)
            else:
                index.append(states[i][-1])
        index = np.array(index).astype(int)
        return self.states_after[index]


class Loss(nn.Module):
    def __init__(self,lagrange_multiplier,eps):
        super(Loss, self).__init__()
        self.eps = eps
        self.lagrange_multiplier = lagrange_multiplier


    def forward(self, value, nn_train_value,train_level,train_roa_labels,class_weights):
        class_labels = 2 * train_roa_labels - 1
        # decision_distance = torch.tensor(train_level - value, dtype=torch.float64, requires_grad=True)
        class_labels_torch = torch.tensor(class_labels, dtype=torch.float64)
        class_weights_torch = torch.tensor(class_weights, dtype=torch.float64)
        # 创建一个300行1列的全0矩阵，数据类型为float64
        zero_matrix = torch.zeros((512, 1), dtype=torch.float64, requires_grad=False)

        classifier_loss = class_weights_torch * torch.maximum(- class_labels_torch * (train_level - value), zero_matrix)
        # classifier_loss_ = torch.tensor(classifier_loss, dtype=torch.float64, requires_grad=True)
        tf_dv_nn_train = nn_train_value - value
        # tf_dv_nn_train_ = torch.tensor(tf_dv_nn_train, dtype=torch.float64, requires_grad=True)
        stop_gra = (value+self.eps).detach()
        train_roa_labels_torch = torch.tensor(train_roa_labels, dtype=torch.float64)
        decrease_loss = train_roa_labels_torch * torch.maximum(tf_dv_nn_train, torch.zeros([512, 1])) / (stop_gra)
        # decrease_loss_ = torch.tensor(decrease_loss, dtype=torch.float64, requires_grad=True)
        # res = torch.mean((class_weights_torch * torch.maximum(- class_labels_torch * (train_level - value), zero_matrix)) + self.lagrange_multiplier *
        #        (train_roa_labels_torch * torch.maximum(nn_train_value - value, torch.zeros([300, 1])) / (stop_gra + self.eps)))
        res = torch.mean(classifier_loss + self.lagrange_multiplier * decrease_loss)
        return res
