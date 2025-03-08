from __future__ import absolute_import, print_function, division

from types import ModuleType
from itertools import product as cartesian
from functools import partial

import torch
from future.builtins import zip, range
from scipy import spatial, sparse, linalg
# import tensorflow._api.v2.compat.v1 as tf1
# tf1.disable_v2_behavior()
import numpy as np
# import gpflow0_4_0 as gpflow04
from .utilities_torch import (concatenate_inputs, batchify, unique_rows)
from safe_learning import config
# import configuration as config
_EPS = np.finfo(config.np_dtype()).eps
class DimensionError(Exception):
    pass

class GridWorld(object):
    """Base class for function approximators on a regular grid.

    Parameters
    ----------
    limits: 2d array-like
        A list of limits. For example, [(x_min, x_max), (y_min, y_max)]
    num_points: 1d array-like
        The number of points with which to grid each dimension.

    """

    def __init__(self, limits, num_points, states):
        """Initialization, see `GridWorld`."""
        super(GridWorld, self).__init__()

        self.limits = np.atleast_2d(limits).astype(config.np_dtype)
        num_points = np.broadcast_to(num_points, len(self.limits))
        self.num_points = num_points.astype(int, copy=False)

        if np.any(self.num_points < 2):
            raise DimensionError('There must be at least 2 points in each '
                                 'dimension.')

        # Compute offset and unit hyperrectangle
        self.offset = self.limits[:, 0]
        # 把区间进行分割
        self.unit_maxes = ((self.limits[:, 1] - self.offset)
                           / (self.num_points - 1)).astype(config.np_dtype)
        self.offset_limits = np.stack((np.zeros_like(self.limits[:, 0]),
                                       self.limits[:, 1] - self.offset),
                                      axis=1)

        # Statistics about the grid
        # limit区间平均分割成n份
        # 第一组low,up分别为limits数组的[0, 0]和[0, 1], n为num_points[0]，第二组[1,0], [1,1], num_points[1]
        self.discrete_points = [np.linspace(low, up, n, dtype=config.np_dtype)
                                for (low, up), n in zip(self.limits,
                                                        self.num_points)]

        self.nrectangles = np.prod(self.num_points - 1)
        self.nindex = np.prod(self.num_points)  # 数组元素相乘

        self.ndim = len(self.limits)
        self.origin_states = states
        self._all_points = None

    @property
    def all_points(self):
        """Return all the discrete points of the discretization.
        在网格中的每一个位置，根据limit和num_states

        Returns
        -------
        points : ndarray
            An array with all the discrete points with size
            (self.nindex, self.ndim).

        """
        if self._all_points is None:
            # [0]矩阵以discrete_points为列向量，[1]矩阵以discrete_points为行向量
            mesh = np.meshgrid(*self.discrete_points, indexing='ij')  # mesh[0].shape: (num_states, num_states)
            # ravel进行降维，column_stack再把两列组合起来
            points = np.column_stack(col.ravel() for col in mesh)  # col.ravel():(nindex,), points.shape:(nindex,2)
            self._all_points = points.astype(config.np_dtype)

        return self._all_points

    def __len__(self):
        """Return the number of points in the discretization."""
        return self.nindex

    def sample_continuous(self, num_samples):
        """Sample uniformly at random from the continuous domain.

        Parameters
        ----------
        num_samples : int

        Returns
        -------
        points : ndarray
            Random points on the continuous rectangle.

        """
        limits = self.limits
        rand = np.random.uniform(0, 1, size=(num_samples, self.ndim))
        return rand * np.diff(limits, axis=1).T + self.offset

    def sample_discrete(self, num_samples, replace=False):
        """Sample uniformly at random from the discrete domain.

        Parameters
        ----------
        num_samples : int
        replace : bool, optional
            Whether to sample with replacement.

        Returns
        -------
        points : ndarray
            Random points on the continuous rectangle.

        """
        idx = np.random.choice(self.nindex, size=num_samples, replace=replace)
        return self.index_to_state(idx)

    def _check_dimensions(self, states):
        """Raise an error if the states have the wrong dimension.

        Parameters
        ----------
        states : ndarray

        """
        if not states.shape[1] == self.ndim:
            raise DimensionError('the input argument has the wrong '
                                 'dimensions.')

    def _center_states(self, states, clip=True):
        """Center the states to the interval [0, x].

        Parameters
        ----------
        states : np.array
        clip : bool, optinal
            If False the data is not clipped to lie within the limits.

        Returns
        -------
        offset_states : ndarray

        """
        states = np.atleast_2d(states).astype(config.np_dtype)
        states = states - self.offset[None, :]
        if clip:
            np.clip(states,
                    self.offset_limits[:, 0] + 2 * _EPS,
                    self.offset_limits[:, 1] - 2 * _EPS,
                    out=states)
        return states

    def index_to_state(self, indices):
        """Convert indices to physical states.

        Parameters
        ----------
        indices : ndarray (int)
            The indices of points on the discretization.

        Returns
        -------
        states : ndarray
            The states with physical units that correspond to the indices.

        """
        indices = np.atleast_1d(indices)
        # np.unravel_index把前面一维索引转为在后面shape中的多维索引位置
        # np.vstack是按照垂直方向（行顺序）堆叠构成新的数组。
        ijk_index = np.vstack(np.unravel_index(indices, self.num_points)).T
        ijk_index = ijk_index.astype(config.np_dtype)
        return ijk_index * self.unit_maxes + self.offset

    def state_to_index(self, states):
        """Convert physical states to indices.

        Parameters
        ----------
        states: ndarray
            Physical states on the discretization.

        Returns
        -------
        indices: ndarray (int)
            The indices that correspond to the physical states.

        """
        states = np.atleast_2d(states)
        self._check_dimensions(states)
        states = np.clip(states, self.limits[:, 0], self.limits[:, 1])
        states = (states - self.offset) * (1. / self.unit_maxes)
        ijk_index = np.rint(states).astype(np.int32)
        return np.ravel_multi_index(ijk_index.T, self.num_points)

    def state_to_rectangle(self, states):
        """Convert physical states to its closest rectangle index.

        Parameters
        ----------
        states : ndarray
            Physical states on the discretization.

        Returns
        -------
        rectangles : ndarray (int)
            The indices that correspond to rectangles of the physical states.

        """
        ind = []
        for i, (discrete, num_points) in enumerate(zip(self.discrete_points,
                                                       self.num_points)):
            idx = np.digitize(states[:, i], discrete)
            idx -= 1
            np.clip(idx, 0, num_points - 2, out=idx)

            ind.append(idx)
        return np.ravel_multi_index(ind, self.num_points - 1)

    def rectangle_to_state(self, rectangles):
        """
        Convert rectangle indices to the states of the bottem-left corners.

        Parameters
        ----------
        rectangles : ndarray (int)
            The indices of the rectangles

        Returns
        -------
        states : ndarray
            The states that correspond to the bottom-left corners of the
            corresponding rectangles.

        """
        rectangles = np.atleast_1d(rectangles)
        ijk_index = np.vstack(np.unravel_index(rectangles,
                                               self.num_points - 1))
        ijk_index = ijk_index.astype(config.np_dtype)
        return (ijk_index.T * self.unit_maxes) + self.offset

    def rectangle_corner_index(self, rectangles):
        """Return the index of the bottom-left corner of the rectangle.

        Parameters
        ----------
        rectangles: ndarray
            The indices of the rectangles.

        Returns
        -------
        corners : ndarray (int)
            The indices of the bottom-left corners of the rectangles.

        """
        ijk_index = np.vstack(np.unravel_index(rectangles,
                                               self.num_points - 1))
        return np.ravel_multi_index(np.atleast_2d(ijk_index),
                                    self.num_points)



class Function(torch.nn.Module):
    """TensorFlow function baseclass.

    It makes sure that variables are reused if the function is called
    multiple times with a TensorFlow template.
    """

    def __init__(self, name='function'):
        super(Function, self).__init__()

        # # Reserve the TensorFlow scope immediately to avoid problems with
        # # Function instances with the same `name`
        # with tf1.variable_scope(name) as scope:
        #     self._scope = scope
        #
        # # Use `original_name_scope` explicitly in case `scope_name` method is
        # # overridden in a child class
        # self._template = tf1.make_template(self._scope.original_name_scope,
        #                                   self.build_evaluation,
        #                                   create_scope_now_=True)

    # @property
    # def scope_name(self):
    #     return self._scope.original_name_scope

    @property
    # def parameters(self):
    #     """Return the variables within the current scope."""
    #     return tf1.get_collection(tf1.GraphKeys.TRAINABLE_VARIABLES,
    #                              scope=self.scope_name)


    def __call__(self, *args, **kwargs):
        """Evaluate the function using the template to ensure variable sharing.

        Parameters
        ----------
        args : list
            The input arguments to the function.
        kwargs : dict, optional
            The keyword arguments to the function.

        Returns
        -------
        outputs : list
            The output arguments of the function as given by evaluate.

        # """
        # with tf1.name_scope('evaluate'):
        #     outputs = self._template(*args, **kwargs)
        # return outputs

    def build_evaluation(self, *args, **kwargs):
        """Build the function evaluation tree.

        Parameters
        ----------
        args : list
        kwargs : dict, optional

        Returns
        -------
        outputs : list

        """
        raise NotImplementedError('This function has to be implemented by the'
                                  'child class.')

    def copy_parameters(self, other_instance):
        """Copy over the parameters from another instance in PyTorch."""
        # 遍历当前实例和other_instance中的所有参数
        for param, other_param in zip(self.parameters(), other_instance.parameters()):
            # 直接使用.data进行参数值的复制
            param.data.copy_(other_param.data)
    def forward(self, *args):
        """Return the constant value. The inputs are ignored."""
        # We use self.constant to ensure that it works correctly with PyTorch's device management.
        # If self.constant is a tensor, it will be on the same device as the module.
        return self.constant

    def __add__(self, other):
        """Add this function to another."""
        return AddedFunction(self, other)

    def __mul__(self, other):
        """Multiply this function with another."""
        return MultipliedFunction(self, other)

    def __neg__(self):
        """Negate the function."""
        return MultipliedFunction(self, -1)

class AddedFunction(Function):
        """A class for adding two individual functions.

        Parameters
        ----------
        fun1 : instance of Function or scalar
        fun2 : instance of Function or scalar

        """

        def __init__(self, fun1, fun2):
            """Initialization, see `AddedFunction`."""
            super(AddedFunction, self).__init__()

            if not isinstance(fun1, Function):
                fun1 = ConstantFunction(fun1)
            if not isinstance(fun2, Function):
                fun2 = ConstantFunction(fun2)

            self.fun1 = fun1
            self.fun2 = fun2

        @property
        def parameters(self):
            """Return the parameters."""
            return self.fun1.parameters + self.fun2.parameters

        def copy_parameters(self, other_instance):
            """Return a copy of the function (new tf variables with same values."""
            return AddedFunction(self.fun1.copy_parameters(other_instance.fun1),
                                 self.fun2.copy_parameters(other_instance.fun1))

        @concatenate_inputs(start=1)
        def build_evaluation(self, points):
            """Evaluate the function."""
            return self.fun1(points) + self.fun2(points)


        def forward(self, *args):
            """Return the constant value. The inputs are ignored."""
            # We use self.constant to ensure that it works correctly with PyTorch's device management.
            # If self.constant is a tensor, it will be on the same device as the module.
            return self.constant


class MultipliedFunction(Function):
    """A class for pointwise multiplying two individual functions.

    Parameters
    ----------
    fun1 : instance of Function or scalar
    fun2 : instance of Function or scalar

    """

    def __init__(self, fun1, fun2):
        """Initialization, see `AddedFunction`."""
        super(MultipliedFunction, self).__init__()

        if not isinstance(fun1, Function):
            fun1 = ConstantFunction(fun1)
        if not isinstance(fun2, Function):
            fun2 = ConstantFunction(fun2)

        self.fun1 = fun1
        self.fun2 = fun2

    @property
    def parameters(self):
        """Return the parameters."""
        return self.fun1.parameters() + self.fun2.parameters()

    def copy_parameters(self, other_instance):
        """Return a copy of the function (copies parameters)."""
        copied_fun1 = self.fun1.copy_parameters(other_instance.fun1)
        copied_fun2 = self.fun2.copy_parameters(other_instance.fun2)
        return MultipliedFunction(copied_fun1, copied_fun2)

    @concatenate_inputs(start=1)
    def build_evaluation(self, points):
        """Evaluate the function."""
        return self.fun1(points) * self.fun2(points)


    def forward(self, *args):
        """Return the constant value. The inputs are ignored."""
        # We use self.constant to ensure that it works correctly with PyTorch's device management.
        # If self.constant is a tensor, it will be on the same device as the module.
        return self.constant


class DeterministicFunction(Function):
    """Base class for function approximators."""

    def __init__(self, **kwargs):
        """Initialization, see `Function` for details."""
        super(DeterministicFunction, self).__init__(**kwargs)

    def forward(self, *args):
        """Return the constant value. The inputs are ignored."""
        # We use self.constant to ensure that it works correctly with PyTorch's device management.
        # If self.constant is a tensor, it will be on the same device as the module.
        return self.constant



class ConstantFunction(DeterministicFunction):
    """A function with a constant value."""

    def __init__(self, constant, name='constant_function'):
        """Initialize, see `ConstantFunction`."""
        super(ConstantFunction, self).__init__(name=name)
        self.constant = constant

    @concatenate_inputs(start=1)
    def build_evaluation(self, points):
        return self.constant

    def forward(self, *args):
        """Return the constant value. The inputs are ignored."""
        # We use self.constant to ensure that it works correctly with PyTorch's device management.
        # If self.constant is a tensor, it will be on the same device as the module.
        return self.constant

class LinearSystem(object):
    """A linear system.

    y = A_1 * x + A_2 * x_2 ...

    Parameters
    ----------
    *matrices : list
        Can specify an arbitrary amount of matrices for the linear system. Each
        is multiplied by the corresponding state that is passed to evaluate.

    """

    def __init__(self, matrices, name='linear_system'):
        """Initialize."""
        super(LinearSystem, self).__init__()
        fun = lambda x: np.atleast_2d(x).astype(np.float64)  # 变成至少二维的
        # python3中map返回迭代器，根据fun运算matrices，np.hstack是在水平方向堆叠数组
        self.matrix = torch.from_numpy(np.hstack(map(fun, matrices)))
        # self.matrix=tensor.type
        self.matrix=torch.cat((self.matrix,self.matrix),dim=1)

        self.output_dim, self.input_dim = self.matrix.shape

    @concatenate_inputs(start=1)
    def forward(self, points):
        """Return the function values.

        Parameters
        ----------
        points : ndarray
            The points at which to evaluate the function. One row for each
            data points.

        Returns
        -------
        values : tf.Tensor
            A 2D array with the function values at the points.

        """
        return torch.matmul(points, self.matrix.t())
