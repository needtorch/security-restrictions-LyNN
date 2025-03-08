from __future__ import absolute_import, division, print_function
import warnings

import torch
import numpy as np
import itertools
from heapq import heappop, heappush
from typing import Union, Tuple
import torch.nn as nn
from future.builtins import zip, range
from collections import Sequence
from .utilities_torch import (batchify, unique_rows, combinations)
# import configuration as config
from safe_learning import config


# def smallest_boundary_value(fun, discretization):
#     """Determine the smallest value of a function on its boundary using PyTorch.
#
#     Parameters
#     ----------
#     fun : callable
#         A PyTorch function that we want to evaluate.
#     discretization : instance of `GridWorld`
#         The discretization. If None, then the function is assumed to be
#         defined on a discretization already.
#
#     Returns
#     -------
#     min_value : float
#         The smallest value on the boundary.
#
#     """
#     min_value = float('inf')
#
#     # Check boundaries for each axis
#     for i in range(discretization.ndim):
#         # Use boundary values only for the ith element
#         tmp = list(discretization.discrete_points)
#         tmp[i] = discretization.discrete_points[i][[0, -1]]
#
#         # Generate all points
#         mesh = np.meshgrid(*tmp, indexing='ij')  # Create mesh grid
#         all_points = np.column_stack([col.ravel() for col in mesh])  # Stack columns
#
#         # Convert all_points to PyTorch tensor
#         all_points_tensor = torch.tensor(all_points, dtype=torch.float32)
#
#         # Evaluate the function
#         values = fun(all_points_tensor)
#
#         # Update the minimum value
#         smallest = torch.min(values)
#         min_value = min(min_value, smallest.item())  # Convert to Python float and update min_value
#
#     return min_value

def smallest_boundary_value(fun, discretization):
    """Determine the smallest value of a function on its boundary.

    Parameters
    ----------
    fun : callable
        A tensorflow function that we want to evaluate.
    discretization : instance of `GridWorld`
        The discretization. If None, then the function is assumed to be
        defined on a discretization already.

    Returns
    -------
    min_value : float
        The smallest value on the boundary.

    """
    min_value = np.inf

    # Check boundaries for each axis
    for i in range(discretization.ndim):
        # Use boundary values only for the ith element
        tmp = list(discretization.discrete_points)
        tmp[i] = discretization.discrete_points[i][[0, -1]]

        # Generate all points
        columns = (x.ravel() for x in np.meshgrid(*tmp, indexing='ij'))
        all_points = np.column_stack(columns)
        all_points_tensor = torch.tensor(all_points, dtype=config.dtype)

        # Update the minimum value
        smallest = torch.min(fun(all_points_tensor))
        min_value = min(min_value, smallest.item())

    return min_value

def get_lyapunov_region(lyapunov, discretization, init_node):
    """Get the region within which a function is a Lyapunov function using PyTorch.

    Parameters
    ----------
    lyapunov : callable
        A PyTorch function.
    discretization : instance of `GridWorld`
        The discretization on which to check the increasing property.
    init_node : tuple
        The node at which to start the verification.

    Returns
    -------
    region : ndarray
        A boolean array that contains all the states for which lyapunov is a
        Lyapunov function that can be used for stability verification.

    """
    # Convert all points to a PyTorch tensor
    all_points_tensor = torch.tensor(discretization.all_points, dtype=torch.float32)

    # Evaluate the Lyapunov function
    values = lyapunov(all_points_tensor)
    lyapunov_values = values.reshape(discretization.num_points).detach().numpy()

    # Starting point for the verification
    init_value = lyapunov_values[init_node]

    ndim = discretization.ndim
    num_points = discretization.num_points

    # Indices for generating neighbors
    index_generator = itertools.product(*[(0, -1, 1) for _ in range(ndim)])
    neighbor_indices = np.array(list(index_generator)[1:])

    # Array keeping track of visited nodes
    visited = np.zeros(discretization.num_points, dtype=bool)
    visited[init_node] = True

    # Create priority queue
    tiebreaker = itertools.count()
    last_value = init_value
    priority_queue = [(init_value, next(tiebreaker), init_node)]

    while priority_queue:
        value, _, next_node = heappop(priority_queue)

        # Check if we reached the boundary of the discretization
        if np.any(next_node == 0) or np.any(next_node == num_points - 1):
            visited[tuple(next_node)] = False
            break

        # Ensure we are in the positive definite part of the function
        if value < last_value:
            break

        last_value = value

        # Get all neighbors
        neighbors = next_node + neighbor_indices

        # Filter out neighbors that are already visited
        is_new = ~visited[tuple(neighbors.T)]
        neighbors = neighbors[is_new]

        if neighbors.size > 0:
            indices = tuple(neighbors.T)
            # Mark as visited
            visited[indices] = True
            # Retrieve values
            neighbor_values = lyapunov_values[indices]

            # Add new neighbors to the priority queue
            for neighbor_value, neighbor in zip(neighbor_values, neighbors):
                heappush(priority_queue, (neighbor_value, next(tiebreaker), neighbor))

    # Prune nodes that were neighbors but haven't been visited
    for _, _, node in priority_queue:
        visited[tuple(node)] = False

    return visited

class Lyapunov(object):
    """A class for general Lyapunov functions.

    Parameters
    ----------
    discretization : ndarray
        A discrete grid on which to evaluate the Lyapunov function.
    lyapunov_function : callable or instance of `DeterministicFunction`
        The lyapunov function. Can be called with states and returns the
        corresponding values of the Lyapunov function.
    dynamics : a callable or an instance of `Function`
        The dynamics model. Can be either a deterministic function or something
        uncertain that includes error bounds.
    lipschitz_dynamics : ndarray or float
        The Lipschitz constant of the dynamics. Either globally, or locally
        for each point in the discretization (within a radius given by the
        discretization constant. This is the closed-loop Lipschitz constant
        including the policy!
    lipschitz_lyapunov : ndarray or float
        The Lipschitz constant of the lyapunov function. Either globally, or
        locally for each point in the discretization (within a radius given by
        the discretization constant.
    tau : float
        The discretization constant.
    policy : ndarray, optional
        The control policy used at each state (Same number of rows as the
        discretization).
    initial_set : ndarray, optional
        A boolean array of states that are known to be safe a priori.
    adaptive : bool, optional
        A boolean determining whether an adaptive discretization is used for
        stability verification.

    """

    def __init__(self, states, states_all, lyapunov_function, dynamics,
                     lipschitz_dynamics, lipschitz_lyapunov,
                     tau, policy, initial_set=None, adaptive=False):
        """Initialization, see `Lyapunov` for details."""
        super(Lyapunov, self).__init__()
        self.states_all = states_all
        self.states = states #torch.tensor(states, dtype=torch.float32)
        self.policy = policy  # Assume policy is a callable function or PyTorch module
        self.exist_unsafe = True
        self.safe_nodes = None
        self.nodes_withff = None
        # Keep track of the safe sets
        self.safe_set = torch.zeros(states.shape[0], dtype=torch.bool)

        self.initial_safe_set = initial_set
        if initial_set is not None:
            self.safe_set[initial_set] = True

        # Discretization constant
        self.tau = tau

        # Storage for graph
        self.storage = None

        # Lyapunov values
        self.values = torch.empty(states.shape[0], dtype=config.dtype)

        self.c_max = 1.5
        self.max = 1
        '''err???   self.feed_dict[self.c_max] = 1'''

         # Dynamics and Lyapunov function as PyTorch modules or functions
        self.dynamics = dynamics
        if hasattr(self.dynamics, 'property'):
            var = self.dynamics.property
            print("var:",var)
        self.lyapunov_function = lyapunov_function

        # Lipschitz constants
        self._lipschitz_dynamics = lipschitz_dynamics
        self._lipschitz_lyapunov = lipschitz_lyapunov

        # Adaptive discretization
        self.adaptive = adaptive
        # Keep track of the refinement `N(x)` used around each state `x`
        self._refinement = torch.zeros(states.shape[0], dtype=torch.int32)
        if initial_set is not None:
            self._refinement[initial_set] = 1

        # Additional methods would be here

    def lipschitz_dynamics(self, states):
        """Return the Lipschitz constant for given states and actions.

        Parameters
        ----------
        states : ndarray or Tensor

        Returns
        -------
        lipschitz : float, ndarray or Tensor
            If lipschitz_dynamics is a callable then returns local Lipschitz
            constants. Otherwise returns the Lipschitz constant as a scalar.

        """
        if hasattr(self._lipschitz_dynamics, '__call__'):
            return self._lipschitz_dynamics(states)
        else:
            return self._lipschitz_dynamics

    def lipschitz_lyapunov(self, states):
        """Return the local Lipschitz constant at a given state.

        Parameters
        ----------
        states : ndarray or Tensor

        Returns
        -------
        lipschitz : float, ndarray or Tensor
            If lipschitz_lyapunov is a callable then returns local Lipschitz
            constants. Otherwise returns the Lipschitz constant as a scalar.

        """
        if hasattr(self._lipschitz_lyapunov, '__call__'):
            return self._lipschitz_lyapunov(states)
        else:
            return self._lipschitz_lyapunov

    def threshold(self, states, tau=None):
        """Return the safety threshold for the Lyapunov condition.
        返回安全阈值

        Parameters
        ----------
        states : ndarray or Tensor

        tau : float or Tensor, optional
            Discretization constant to consider.

        Returns
        -------
        lipschitz : float, ndarray or Tensor
            Either the scalar threshold or local thresholds, depending on
            whether lipschitz_lyapunov and lipschitz_dynamics are local or not.

        """
        if tau is None:
            tau = self.tau
        lv = self.lipschitz_lyapunov(states)
        # 对 lv 进行维度检查和调整
        if hasattr(self._lipschitz_lyapunov, '__call__') and lv.dim() > 1:
            lv = torch.norm(lv, p=1, dim=1, keepdim=True)
        lf = self.lipschitz_dynamics(states)
        return -lv * (1. + lf) * tau

    def is_safe(self, state):
        """Return a boolean array that indicates whether the state is safe.

        Parameters
        ----------
        state : ndarray

        Returns
        -------
        safe : boolean
            Is true if the corresponding state is inside the safe set.

        !!!!! change function !!!!!!!
        """
        return self.safe_set[self.discretization.state_to_index(state)]

    # def update_values(self):
    #     """Update the discretized values when the Lyapunov function changes.
    #         把原本的值用lyapunov函数计算得到新的值
    #     """
    #     # Use a placeholder to avoid loading a large discretization into the
    #     # TensorFlow graph
    #     storage = get_storage(self._storage)
    #     if storage is None:
    #         tf_points = tf1.placeholder(tf1.float64,
    #                                    shape=[None, self.states.shape[1]],
    #                                    name='discretization_points')
    #         tf_values = self.lyapunov_function(tf_points)
    #         storage = [('points', tf_points), ('values', tf_values)]
    #         set_storage(self._storage, storage)
    #     else:
    #         tf_points, tf_values = storage.values()
    #
    #     feed_dict = self.feed_dict
    #     feed_dict[tf_points] = self.states
    #     # 用类中定义的lyapunov函数计算
    #     self.values = tf_values.eval(feed_dict).squeeze()
    #     # print(self.values[:10])
    #     # self.values = self.lyapunov_function(feed_dict[tf_points]).eval()

    def update_values(self):
        """Update the discretized values when the Lyapunov function changes."""
        # 直接使用PyTorch张量调用Lyapunov函数计算新的值
        self.values = self.lyapunov_function(self.states[:, :-1]).squeeze()
        # 如果lyapunov_function返回的是多维张量，squeeze()方法可以移除单维条目

    # 示例用法

    def v_decrease_confidence(self, states, next_states):
        """Compute confidence intervals for the decrease along Lyapunov function.

        Parameters
        ----------
        states : np.array
            The states at which to start (could be equal to discretization).
        next_states : np.array
            The dynamics evaluated at each point on the discretization. If
            the dynamics are uncertain then next_states is a tuple with mean
            and error bounds.

        Returns
        -------
        mean : np.array
            The expected decrease in values at each grid point.
        error_bounds : np.array
            The error bounds for the decrease at each grid point

        """
        if isinstance(next_states, Sequence):
            next_states, error_bounds = next_states
            lv = self.lipschitz_lyapunov(next_states)
            bound = torch.sum(lv * error_bounds, dim=1, keepdims=True)
        else:
            bound = torch.zeros((), dtype=config.dtype)

        v_decrease = (self.lyapunov_function(next_states[:, :-1])
                      - self.lyapunov_function(states[:, :-1]))

        return v_decrease, bound
    # def v_decrease_confidence(self, states, next_states):
    #
    #     if isinstance(next_states, Sequence):
    #         next_states, error_bounds = next_states
    #         lv = self.lipschitz_lyapunov(next_states)
    #         bound = tf1.reduce_sum(lv * error_bounds, axis=1, keepdims=True)
    #     else:
    #         bound = tf1.constant(0., dtype=config.dtype)
    #
    #     v_decrease = (self.lyapunov_function(next_states)
    #                   - self.lyapunov_function(states))

        return v_decrease, bound
    # def v_decrease_confidence(self, states: torch.Tensor,
    #                           next_states: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]):
    #     """Compute confidence intervals for the decrease along Lyapunov function in PyTorch.
    #     """
    #     if isinstance(next_states, tuple):
    #         next_states_mean, error_bounds = next_states
    #         lv = self.lyapunov_function(next_states_mean)
    #         bound = torch.sum(lv * error_bounds, dim=1, keepdim=True)
    #     else:
    #         next_states_mean = next_states
    #         bound = torch.tensor(0., dtype=states.dtype, device=states.device)
    #
    #     v_decrease = self.lyapunov_function(next_states_mean) - self.lyapunov_function(states)
    #
    #     return v_decrease, bound

    # def v_decrease_bound(self, states, next_states):
    #     """Compute confidence intervals for the decrease along Lyapunov function.
    #     计算沿着lyapunov函数减少的置信区间
    #
    #     Parameters
    #     ----------
    #     states : np.array
    #         The states at which to start (could be equal to discretization).
    #     next_states : np.array or tuple
    #         The dynamics evaluated at each point on the discretization. If
    #         the dynamics are uncertain then next_states is a tuple with mean
    #         and error bounds.
    #
    #     Returns
    #     -------
    #     upper_bound : np.array
    #         The upper bound on the change in values at each grid point.
    #
    #     """
    #     v_dot, v_dot_error = self.v_decrease_confidence(states, next_states)
    #
    #     return v_dot + v_dot_error

    def v_decrease_bound(self, states: torch.Tensor,
                         next_states: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]):
        """Compute the upper bound on the change in values at each grid point in PyTorch.
        """
        v_dot, v_dot_error = self.v_decrease_confidence(states, next_states)

        return v_dot + v_dot_error

    # def safety_constraint(self, policy, include_initial=True):
    #     """Return the safe set for a given policy.
    #
    #     Parameters
    #     ----------
    #     policy : ndarray
    #         The policy used at each discretization point.
    #     include_initial : bool, optional
    #         Whether to include the initial safe set.
    #
    #     Returns
    #     -------
    #     constraint : ndarray
    #         A boolean array indicating where the safety constraint is
    #         fulfilled.
    #
    #     """
    #     prediction = self.dynamics(self.discretization, policy)
    #     v_dot_bound = self.v_decrease_bound(self.discretization, prediction)
    #
    #     # Update the safe set
    #     v_dot_negative = v_dot_bound < self.threshold
    #
    #     # Make sure initial safe set is included
    #     if include_initial and self.initial_safe_set is not None:
    #         v_dot_negative[self.initial_safe_set] = True
    #
    #     return v_dot_negative

    def safety_constraint(self, policy, include_initial=True):
        """
        Return the safe set for a given policy, adapted for PyTorch 2.

        Parameters
        ----------
        policy : torch.Tensor
            The policy used at each discretization point.
        include_initial : bool, optional
            Whether to include the initial safe set.

        Returns
        -------
        constraint : torch.Tensor
            A boolean tensor indicating where the safety constraint is fulfilled.
        """
        # `self.dynamics` is adapted to accept and return PyTorch tensors
        prediction = self.dynamics.build_evaluation(self.discretization, policy)

        # `self.v_decrease_bound` also needs to be adapted for PyTorch tensors
        v_dot_bound = self.v_decrease_bound(self.discretization, prediction)

        # Threshold comparison in PyTorch
        v_dot_negative = v_dot_bound < self.threshold

        # Including the initial safe set if specified
        if include_initial and self.initial_safe_set is not None:
            # Here, `initial_safe_set` should be a boolean tensor or a tensor of indices
            v_dot_negative[self.initial_safe_set] = True

        return v_dot_negative


    # @with_scope('update_C')
    # def update_c(self):
    #     tf_dv_nn = self.lyapunov_function(self.dynamics(self.states)) - self.values
    #     decrease_region = np.squeeze(tf_dv_nn < 0)
    #     self.safe_set |= decrease_region
    #     value_order = np.argsort(self.values.numpy().reshape(self.values.shape[0]))
    #     safe_set = self.safe_set
    #     safe_set_order = safe_set[value_order]
    #     if self.exist_unsafe:
    #         FF = 30
    #         for i in range(safe_set.shape[0]):
    #             if safe_set_order[i]==False and FF!=0:
    #                 FF = FF - 1
    #             if FF==0 and safe_set_order[i]==False:
    #                 index = i - 1
    #                 if index < 0:
    #                     index = 0
    #                 break
    #     else:
    #         index = np.argmin(safe_set_order) - 1
    #         if index < 0:
    #             index = 0
    #         # print("index:",index)
    #     # safe_nodes = value_order[safe_set]
    #     # index = self.safe_set.shape[0] - np.argmin(re_safe_set_order) - 1
    #     self.c_max = self.values[value_order[index]]
    #     # self.c_max = self.values[value_order[150]]
    #     safe_set_order[index:] = False
    #     safe_nodes = value_order[safe_set_order]
    #     self.safe_set[:] = False
    #     self.safe_set[safe_nodes] = True
    #     self.safe_nodes = safe_nodes
    #     self.nodes_withff = value_order[:index]
    #     if self.initial_safe_set is not None:
    #         self.safe_set[self.initial_safe_set] = True
    #     self.max = self.values[value_order[-1]]

    def update_c(self):
        tf_dv_nn = self.lyapunov_function(torch.tensor(self.dynamics.states_after[:, :-1],
                                                       dtype=torch.float64)) - self.values
        # tf_dv_nn = self.lyapunov_function((torch.tensor(self.dynamics.states_after,
        #                                                 dtype=torch.float64).detach() - x_ly_mean) / x_ly_std) - self.values
        decrease_region = (tf_dv_nn < 0).squeeze()
        print('decrease_region: {}'.format(decrease_region.sum()))
        self.safe_set |= decrease_region
        value_order = np.argsort(self.values.detach().numpy().reshape(self.values.shape[0]))
        safe_set = self.safe_set.numpy()
        safe_set_order = safe_set[value_order]
        # index=0
        if self.exist_unsafe:
            FF = 25
            for i in range(safe_set.shape[0]):
                if safe_set_order[i] == False and FF != 0:
                    FF = FF - 1
                if FF == 0 and safe_set_order[i] == False:
                    index = i - 1
                    if index < 0:
                        index = 0
                    break
        else:
            index = np.argmin(safe_set_order) - 1
            if index < 0:
                index = 0

        # if index < 0 or index == 0:
        #     index = 0
        print("index:",index)
        self.c_max = self.values[value_order[index]]
        safe_set_order[index:] = False
        safe_nodes = value_order[safe_set_order]
        self.safe_set.fill_(False)
        self.safe_set[safe_nodes] = True
        self.safe_nodes = safe_nodes
        self.nodes_withff = value_order[:index]
        if self.initial_safe_set is not None:
            self.safe_set[self.initial_safe_set] = True
        self.max = self.values[value_order[-1]]


    # @with_scope('update_safe_set')
    # def update_safe_set(self, can_shrink=True, max_refinement=1,
    #                     safety_factor=1., parallel_iterations=1):
    #     """Compute and update the safe set.
    #
    #     Parameters
    #     ----------
    #     can_shrink : bool, optional
    #         A boolean determining whether previously safe states other than the
    #         initial safe set must be verified again (i.e., can the safe set
    #         shrink in volume?)
    #         确定是否必须再次验证除初始安全集以外的先前安全状态的布尔值（即，安全集的容量是否可以缩小？）
    #     max_refinement : int, optional
    #         The maximum integer divisor used for adaptive discretization.
    #     safety_factor : float, optional
    #         A multiplicative factor greater than 1 used to conservatively
    #         estimate the required adaptive discretization.
    #     parallel_iterations : int, optional
    #         The number of parallel iterations to use for safety verification in
    #         the adaptive case. Passed to `tf.map_fn`.
    #
    #     """
    #     safety_factor = np.maximum(safety_factor, 1.)
    #     storage = get_storage(self._storage)
    #     # print('lyapunov_nn.storage: {}'.format(storage))
    #     if storage is None:
    #         # Placeholder for states to evaluate for safety
    #         tf_states = tf1.placeholder(config.dtype,
    #                                    shape=[None, self.states.shape[1]],
    #                                    name='verification_states')
    #         actions = self.policy(tf_states)
    #         next_states = self.dynamics(tf_states, actions)
    #
    #         decrease = self.v_decrease_bound(tf_states, next_states)  # 计算沿着lyapunov函数减少的置信区间
    #         threshold = self.threshold(tf_states, self.tau)  # tau为0，安全阈值，结果为0
    #         tf_negative = tf1.squeeze(tf1.less(decrease, threshold), axis=1)  # less：如果小于，返回True，▲v<0
    #
    #         storage = [('states', tf_states), ('negative', tf_negative)]
    #
    #         if self.adaptive:  # False
    #             # Compute an integer n such that dv < threshold for tau / n
    #             ratio = safety_factor * threshold / decrease
    #             # If dv = 0, check for nan values, and clip to n = 0
    #             tf_n_req = tf1.where(tf1.is_nan(ratio),
    #                                 tf1.zeros_like(ratio), ratio)
    #             # Edge case: ratio = 1 should correspond to n = 2
    #             # TODO
    #             # If dv < 0, also clip to n = 0
    #             tf_n_req = tf1.ceil(tf1.maximum(tf_n_req, 0))
    #
    #             dim = int(self.states.shape[1])
    #             lengths = self.discretization.unit_maxes.reshape((-1, 1))
    #
    #             def refined_safety_check(data):
    #                 """Verify decrease condition in a locally refined grid."""
    #                 center = tf1.reshape(data[:-1], [1, dim])
    #                 n_req = tf1.cast(data[-1], tf1.int32)
    #
    #                 start = tf1.constant(-1., dtype=config.dtype)
    #                 spacing = tf1.reshape(tf1.linspace(start, 1., n_req),
    #                                      [1, -1])
    #                 border = (0.5 * (1 - 1 / n_req) * lengths *
    #                           tf1.tile(spacing, [dim, 1]))
    #                 mesh = tf1.meshgrid(*tf1.unstack(border), indexing='ij')
    #                 points = tf1.stack([tf1.reshape(col, [-1]) for col in mesh],
    #                                   axis=1)
    #                 points += center
    #
    #                 refined_threshold = self.threshold(center,
    #                                                    self.tau / n_req)
    #                 negative = tf1.less(decrease, refined_threshold)
    #                 refined_negative = tf1.reduce_all(negative)
    #                 return refined_negative
    #
    #             tf_refinement = tf1.placeholder(tf1.int32, [None, 1],
    #                                            'refinement')
    #             data = tf1.concat([tf_states, tf1.cast(tf_refinement,
    #                                                  config.dtype)], axis=1)
    #             tf_refined_negative = tf1.map_fn(refined_safety_check, data,
    #                                             tf1.bool, parallel_iterations)
    #             storage += [('n_req', tf_n_req), ('refinement', tf_refinement),
    #                         ('refined_negative', tf_refined_negative)]
    #
    #         set_storage(self._storage, storage)
    #     else:
    #         if self.adaptive:
    #             (tf_states, tf_negative, tf_n_req, tf_refinement,
    #              tf_refined_negative) = storage.values()
    #         else:
    #             tf_states, tf_negative = storage.values()
    #
    #     # Get relevant properties
    #     feed_dict = self.feed_dict
    #
    #     # print('self.initial_safe_set: {}'.format(self.initial_safe_set))
    #     if can_shrink:
    #         # Reset the safe set and adaptive discretization
    #         safe_set = np.zeros_like(self.safe_set, dtype=bool)
    #         refinement = np.zeros_like(self._refinement, dtype=int)  # _refinement 安全点为1，其余为0
    #         # 如果存在初始安全集，在safe_set和refinement数组中做好标记
    #         if self.initial_safe_set is not None:
    #             safe_set[self.initial_safe_set] = True
    #             refinement[self.initial_safe_set] = 1
    #     else:
    #         # Assume safe set cannot shrink
    #         # 设置为函数内属性的值
    #         safe_set = self.safe_set
    #         refinement = self._refinement
    #
    #     value_order = np.argsort(self.values) # 把value进行排序，并返回索引
    #     # print('value_order.shape : {}'.format(value_order.shape))
    #     # 按照已排序的value的索引，重新排列safe_set和refinement，所得到的结果就是数组前面的部分小于lyapunov函数的值
    #     safe_set = safe_set[value_order]
    #     refinement = refinement[value_order]
    #
    #     # print('safe_set_after_order: {}'.format(safe_set))
    #     # Verify safety in batches
    #     batch_size = config.gp_batch_size   # 1000,values的数量
    #     # 分批生成数组
    #     batch_generator = batchify((value_order, safe_set, refinement),
    #                                batch_size)
    #     # index_to_state = self.discretization.index_to_state
    #     #######################################################################
    #
    #     for i, (indices, safe_batch, refine_batch) in batch_generator:
    #         # print(indices)
    #         states = self.states[indices]
    #         feed_dict[tf_states] = states
    #         # print(feed_dict[tf_states])
    #         # Update the safety with the safe_batch result
    #         negative = tf_negative.eval(feed_dict)  # ▲v < 0的区域
    #         # ▲v < 0的区域加入到safe_set中
    #         # print('safe_batch: {}'.format(safe_batch))
    #         safe_batch |= negative
    #         refine_batch[negative] = 1
    #
    #         # Boolean array: argmin returns first element that is False
    #         # If all are safe then it returns 0
    #         # np.argmin 返回第一个为false的值，否则返回0
    #         bound = np.argmin(safe_batch)
    #         refine_bound = 0  # why 0？
    #
    #         # Check if there are unsafe elements in the batch
    #         # 查看是否有不安全因素
    #         if bound > 0 or not safe_batch[0]:
    #             if self.adaptive and max_refinement > 1:
    #                 # Compute required adaptive refinement
    #                 feed_dict[tf_states] = states[bound:]
    #                 refine_batch[bound:] = tf_n_req.eval(feed_dict).ravel()
    #
    #                 # We do not need to refine cells that correspond to known
    #                 # safe states
    #                 idx_safe = np.logical_or(negative,
    #                                          self.initial_safe_set[indices])
    #                 refine_batch[idx_safe] = 1
    #
    #                 # Identify cells to refine
    #                 states_to_check = np.logical_and(refine_batch >= 1,
    #                                                  refine_batch <=
    #                                                  max_refinement)
    #                 states_to_check = states_to_check[bound:]
    #
    #                 if np.all(states_to_check):
    #                     stop = len(states_to_check)
    #                 else:
    #                     stop = np.argmin(states_to_check)
    #
    #                 if stop > 0:
    #                     feed_dict[tf_states] = states[bound:bound + stop]
    #                     feed_dict[tf_refinement] = refine_batch[bound:
    #                                                             bound + stop,
    #                                                             None]
    #                     refined_safe = tf_refined_negative.eval(feed_dict)
    #
    #                     # Determine which states are safe under the refined
    #                     # discretization
    #                     if np.all(refined_safe):
    #                         refine_bound = len(refined_safe)
    #                     else:
    #                         refine_bound = np.argmin(refined_safe)
    #                     safe_batch[bound:bound + refine_bound] = True
    #
    #                 # Break if the refined discretization does not work for all
    #                 # states after `bound`
    #                 if stop < len(states_to_check) or refine_bound < stop:
    #                     safe_batch[bound + refine_bound:] = False
    #                     refine_batch[bound + refine_bound:] = 0
    #                     break
    #             else:
    #                 # Make sure all following points are labeled as unsafe
    #                 safe_batch[bound:] = False
    #                 refine_batch[bound:] = 0
    #                 break
    #
    #
    #     max_index = i + bound + refine_bound - 1
    #
    #     #######################################################################
    #
    #     # Set placeholder for c_max to the corresponding value
    #     feed_dict[self.c_max] = self.values[value_order[max_index]]
    #
    #     # Restore the order of the safe set and adaptive refinement
    #     safe_nodes = value_order[safe_set]
    #     self.safe_set[:] = False
    #     self.safe_set[safe_nodes] = True
    #     self._refinement[value_order] = refinement
    #
    #     # Ensure the initial safe set is kept
    #     if self.initial_safe_set is not None:
    #         self.safe_set[self.initial_safe_set] = True
    #         self._refinement[self.initial_safe_set] = 1
    #     # print('safe_set: {}'.format(self.safe_set.sum()))
    #
    #
    #
    #
    # def update_safe_set(self, can_shrink=True, max_refinement=1,
    #                     safety_factor=1., parallel_iterations=1):
    #     safety_factor = np.maximum(safety_factor, 1.)
    #
    #     if can_shrink:
    #         # Reset the safe set and adaptive discretization
    #         safe_set = np.zeros_like(self.safe_set, dtype=bool)
    #         refinement = np.zeros_like(self._refinement, dtype=int)  # _refinement 安全点为1，其余为0
    #         # 如果存在初始安全集，在safe_set和refinement数组中做好标记
    #         if self.initial_safe_set is not None:
    #             safe_set[self.initial_safe_set] = True
    #             refinement[self.initial_safe_set] = 1
    #     else:
    #         # Assume safe set cannot shrink
    #         # 设置为函数内属性的值
    #         safe_set = self.safe_set
    #         refinement = self._refinement
    #
    #     value_order = np.argsort(self.values)  # 把value进行排序，并返回索引
    #     # print('value_order.shape : {}'.format(value_order.shape))
    #     # 按照已排序的value的索引，重新排列safe_set和refinement，所得到的结果就是数组前面的部分小于lyapunov函数的值
    #     safe_set = safe_set[value_order]
    #     refinement = refinement[value_order]
    #
    #     # print('safe_set_after_order: {}'.format(safe_set))
    #     # Verify safety in batches
    #     batch_size = config.gp_batch_size  # 1000,values的数量
    #     # 分批生成数组
    #     batch_generator = batchify((value_order, safe_set, refinement),
    #                                batch_size)
    #     for i, (indices, safe_batch, refine_batch) in batch_generator:
    #         # print(indices)
    #         states = self.states[indices]
    #         feed_dict[tf_states] = states
    #         # print(feed_dict[tf_states])
    #         # Update the safety with the safe_batch result
    #         negative = tf_negative.eval(feed_dict)  # ▲v < 0的区域
    #         # ▲v < 0的区域加入到safe_set中
    #         # print('safe_batch: {}'.format(safe_batch))
    #         safe_batch |= negative
    #         refine_batch[negative] = 1
    #
    #         # Boolean array: argmin returns first element that is False
    #         # If all are safe then it returns 0
    #         # np.argmin 返回第一个为false的值，否则返回0
    #         bound = np.argmin(safe_batch)
    #         refine_bound = 0  # why 0？
    #
    #         # Check if there are unsafe elements in the batch
    #         # 查看是否有不安全因素
    #         if bound > 0 or not safe_batch[0]:
    #             if self.adaptive and max_refinement > 1:
    #                 # Compute required adaptive refinement
    #                 feed_dict[tf_states] = states[bound:]
    #                 refine_batch[bound:] = tf_n_req.eval(feed_dict).ravel()
    #
    #                 # We do not need to refine cells that correspond to known
    #                 # safe states
    #                 idx_safe = np.logical_or(negative,
    #                                          self.initial_safe_set[indices])
    #                 refine_batch[idx_safe] = 1
    #
    #                 # Identify cells to refine
    #                 states_to_check = np.logical_and(refine_batch >= 1,
    #                                                  refine_batch <=
    #                                                  max_refinement)
    #                 states_to_check = states_to_check[bound:]
    #
    #                 if np.all(states_to_check):
    #                     stop = len(states_to_check)
    #                 else:
    #                     stop = np.argmin(states_to_check)
    #
    #                 if stop > 0:
    #                     feed_dict[tf_states] = states[bound:bound + stop]
    #                     feed_dict[tf_refinement] = refine_batch[bound:
    #                                                             bound + stop,
    #                                                None]
    #                     refined_safe = tf_refined_negative.eval(feed_dict)
    #
    #                     # Determine which states are safe under the refined
    #                     # discretization
    #                     if np.all(refined_safe):
    #                         refine_bound = len(refined_safe)
    #                     else:
    #                         refine_bound = np.argmin(refined_safe)
    #                     safe_batch[bound:bound + refine_bound] = True
    #
    #                 # Break if the refined discretization does not work for all
    #                 # states after `bound`
    #                 if stop < len(states_to_check) or refine_bound < stop:
    #                     safe_batch[bound + refine_bound:] = False
    #                     refine_batch[bound + refine_bound:] = 0
    #                     break
    #             else:
    #                 # Make sure all following points are labeled as unsafe
    #                 safe_batch[bound:] = False
    #                 refine_batch[bound:] = 0
    #                 break
    #
    #     max_index = i + bound + refine_bound - 1
    #
    #     #######################################################################
    #
    #     # Set placeholder for c_max to the corresponding value
    #     feed_dict[self.c_max] = self.values[value_order[max_index]]
    #
    #     # Restore the order of the safe set and adaptive refinement
    #     safe_nodes = value_order[safe_set]
    #     self.safe_set[:] = False
    #     self.safe_set[safe_nodes] = True
    #     self._refinement[value_order] = refinement
    #
    #     # Ensure the initial safe set is kept
    #     if self.initial_safe_set is not None:
    #         self.safe_set[self.initial_safe_set] = True
    #         self._refinement[self.initial_safe_set] = 1
    #     # Assuming states is a tensor of shape [num_states, state_dim]
    #     # and actions is a tensor of shape [num_states, action_dim]
    #     actions = self.policy(self.states)
    #     next_states = dynamics(states, actions)
    #     v_decrease = lyapunov_nn.lyapunov_function(next_states) - lyapunov_nn.lyapunov_function(states)
    #
    #     # Calculate the safety threshold
    #     if threshold is None:
    #         threshold = -lyapunov_nn.lipschitz_lyapunov(states) * (1. + lyapunov_nn.lipschitz_dynamics(states)) * tau
    #
    #     # Mark states as safe based on the decrease condition
    #     safe_batch = v_decrease < threshold
    #
    #     # Refinement for adaptive discretization
    #     if lyapunov_nn.adaptive and max_refinement > 1:
    #         # Example for handling refinement; specifics depend on your model and application
    #         # Placeholder code for adaptive refinement logic
    #         pass
    #
    #     # Update the safe set and refinement arrays based on batch evaluations
    #     safe_set |= safe_batch
    #     refinement[safe_batch] = 1
    #
    #     # Update the LyapunovNetwork instance properties
    #     lyapunov_nn.safe_set = safe_set
    #     lyapunov_nn.refinement = refinement

    def update_safe_set(self, can_shrink=True, max_refinement=1,
                        safety_factor=1., parallel_iterations=1):
        safety_factor = max(safety_factor, 1.)
        if self.storage is None:
            py_states = torch.empty((1, self.states.shape[1]), dtype=config.dtype)
            py_states = self.states
            # py_states = torch.tensor([self.states.shape[1]], dtype=config.dtype)
            # actions = self.policy(py_states)
            next_states = self.dynamics.build_evaluation(py_states)

            decrease = self.v_decrease_bound(py_states, next_states)
            threshold = self.threshold(py_states, self.tau)
            # negative = torch.squeeze(torch.lt(decrease,threshold),dim=0)
            negative = torch.squeeze(torch.lt(decrease, threshold), dim=1)

            self.storage = {'states': py_states, 'negative': negative}

            if self.adaptive:
                ratio = safety_factor * threshold / decrease
                n_req = torch.where(torch.isnan(ratio),
                                torch.zeros_like(ratio), ratio)
                n_req = torch.ceil(torch.maximum(n_req, 0))

                dim = self.states.shape[1]
                lengths = self.discretization.unit_maxes.reshape((-1, 1))

                def refined_safety_check(data):
                    center = data[:-1].reshape(1, dim)
                    n_req = data[-1].type(torch.int32)

                    term_rad_roll = term_rad_pitch = np.deg2rad(30)

                    # 判断roll是否小于np.deg2rad(30)
                    roll = center[0, 40]
                    pitch = center[0,41]
                    if abs(roll) < term_rad_roll and (pitch) < term_rad_pitch:
                        # 如果roll小于np.deg2rad(30)，则认为状态是安全的
                        return True

                    # 如果roll不小于np.deg2rad(30)，则进行原有的安全检查
                    refined_threshold = self.threshold(center, self.tau / n_req)
                    decrease = self.v_decrease_bound(center, self.tau / n_req)
                    negative = torch.lt(decrease, refined_threshold)
                    refined_negative = negative.all()
                    return refined_negative

                # def refined_safety_check(data):
                #     center = data[:-1].reshape(1, dim)
                #     n_req = data[-1].type(torch.int32)
                #
                #     start = torch.tensor(-1., dtype=config.dtype)
                #     spacing = torch.linspace(start, 1., n_req).reshape(1, -1)
                #     border = 0.5 * (1 - 1 / n_req) * lengths * spacing.repeat(dim, 1)
                #     mesh = torch.meshgrid(*border.unbind())
                #     points = torch.stack([col.reshape(-1) for col in mesh], dim=1)
                #     points += center
                #
                #     refined_threshold = self.threshold(center, self.tau / n_req)
                #     negative = torch.lt(decrease, refined_threshold)
                #     refined_negative = negative.all()
                #
                #     return refined_negative

                refinement = torch.zeros((py_states.shape[0], py_states.shape[1]), dtype=torch.int32)
                data = torch.cat([py_states, refinement.type(config.dtype)], dim=1)
                refined_negative = torch.stack([refined_safety_check(d) for d in data])

                self.storage.update({'n_req': n_req, 'refinement': refinement,
                                     'refined_negative': refined_negative})

        else:
            if self.adaptive:
                py_states, negative, n_req, refinement, refined_negative = self.storage.values()
            else:
                py_states, negative = self.storage.values()

        if can_shrink:
            safe_set = torch.zeros_like(self.safe_set, dtype=torch.bool)
            refinement = torch.zeros_like(self._refinement, dtype=torch.int32)
            if self.initial_safe_set is not None:
                safe_set[self.initial_safe_set] = True
                refinement[self.initial_safe_set] = 1
        else:
            safe_set = self.safe_set
            refinement = self._refinement

        value_order = torch.argsort(self.values)
        safe_set = safe_set[value_order]
        refinement = refinement[value_order]

        batch_size = config.gp_batch_size
        batch_generator = batchify((value_order, safe_set, refinement), batch_size)

        for i, (indices, safe_batch, refine_batch) in enumerate(batch_generator):
            states = self.states[indices]
            negative = torch.squeeze(torch.lt(decrease,threshold),dim=1)
            safe_batch |= negative
            refine_batch[negative] = 1

            bound = torch.argmin(safe_batch)
            refine_bound = 0

            if bound > 0 or not safe_batch[0]:
                if self.adaptive and max_refinement > 1:
                    refine_batch[bound:] = n_req
                    idx_safe = negative | self.initial_safe_set[indices]
                    refine_batch[idx_safe] = 1

                    states_to_check = (refine_batch >= 1) & (refine_batch <= max_refinement)
                    states_to_check = states_to_check[bound:]

                    if states_to_check.all():
                        stop = len(states_to_check)
                    else:
                        stop = torch.argmin(states_to_check)

                    if stop > 0:
                        refined_safe = torch.stack([refined_safety_check(d) for d in data[bound:bound + stop]])

                        if refined_safe.all():
                            refine_bound = len(refined_safe)
                        else:
                            refine_bound = torch.argmin(refined_safe)
                        safe_batch[bound:bound + refine_bound] = True

                    if stop < len(states_to_check) or refine_bound < stop:
                        safe_batch[bound + refine_bound:] = False
                        refine_batch[bound + refine_bound:] = 0
                        break
                else:
                    safe_batch[bound:] = False
                    refine_batch[bound:] = 0
                    break

        max_index = i + bound + refine_bound - 1

        self.c_max = self.values[value_order[max_index]]

        safe_nodes = value_order[safe_set]
        self.safe_set[:] = False
        self.safe_set[safe_nodes] = True
        self._refinement[value_order] = refinement

        if self.initial_safe_set is not None:
            self.safe_set[self.initial_safe_set] = True
            self._refinement[self.initial_safe_set] = 1


def perturb_actions(states, actions, perturbations, limits=None):
    """Create state-action pairs by perturbing the actions.

    Parameters
    ----------
    states : ndarray
        An (N x n) array of states at which we want to generate state-action
        pairs.
    actions : ndarray
        An (N x m) array of baseline actions at these states. These
        corresponds to the actions taken by the current policy.
    perturbations : ndarray
        An (X x m) array of policy perturbations that are to be applied to
        each state-action pair.
    limits : list
        List of action-limit tuples.

    Returns
    -------
    state-actions : ndarray
        An (N*X x n+m) array of state-actions pairs, where for each state
        the corresponding action is perturbed by the perturbations.

    """
    num_states, state_dim = states.shape

    # repeat states
    states_new = np.repeat(states, len(perturbations), axis=0)

    # generate perturbations from perturbations around baseline policy
    actions_new = (np.repeat(actions, len(perturbations), axis=0)
                   + np.tile(perturbations, (num_states, 1)))

    state_actions = np.column_stack((states_new, actions_new))

    if limits is not None:
        # Clip the actions
        perturbations = state_actions[:, state_dim:]
        np.clip(perturbations, limits[:, 0], limits[:, 1], out=perturbations)
        # Remove rows that are not unique
        state_actions = unique_rows(state_actions)

    return state_actions


# # @with_scope('get_safe_sample')
# def get_safe_sample(lyapunov, perturbations=None, limits=None, positive=False,
#                     num_samples=None, actions=None):
#     """Compute a safe state-action pair for sampling.
#
#     This function returns the most uncertain state-action pair close to the
#     current policy (as a result of the perturbations) that is safe (maps
#     back into the region of attraction).
#
#     Parameters
#     ----------
#     lyapunov : instance of `Lyapunov'
#         A Lyapunov instance with an up-to-date safe set.
#     perturbations : ndarray
#         An array that, on each row, has a perturbation that is added to the
#         baseline policy in `lyapunov.policy`.
#     limits : ndarray, optional
#         The actuator limits. Of the form [(u_1_min, u_1_max), (u_2_min,..)...].
#         If provided, state-action pairs are clipped to ensure the limits.
#     positive : bool
#         Whether the Lyapunov function is positive-definite (radially
#         increasing). If not, additional checks are carried out to ensure
#         safety of samples.
#     num_samples : int, optional
#         Number of samples to select (uniformly at random) from the safe
#         states within lyapunov.discretization as testing points.
#     actions : ndarray
#         A list of actions to evaluate for each state. Ignored if perturbations
#         is not None.
#
#     Returns
#     -------
#     state-action : ndarray
#         A row-vector that contains a safe state-action pair that is
#         promising for obtaining future observations.
#     var : float
#         The uncertainty remaining at this state.
#
#     """
#     state_dim = lyapunov.discretization.ndim
#     if perturbations is None:
#         action_dim = actions.shape[1]
#     else:
#         action_dim = perturbations.shape[1]
#     action_limits = limits
#
#     storage = get_storage(_STORAGE, index=lyapunov)
#
#     if storage is None:
#         tf_states = tf1.placeholder(config.dtype, shape=[None, state_dim])
#         tf_actions = lyapunov.policy(tf_states)
#
#         # Placeholder for state-actions to evaluate
#         tf_state_actions = tf1.placeholder(config.dtype,
#                                           shape=[None, state_dim + action_dim])
#
#         # Account for deviations of the next value due to uncertainty
#         tf_mean, tf_std = lyapunov.dynamics(tf_state_actions)
#         tf_bound = tf1.reduce_sum(tf_std, axis=1, keepdims=True)
#         tf_lv = lyapunov.lipschitz_lyapunov(tf_mean)
#         tf_error = tf1.reduce_sum(tf_lv * tf_std, axis=1, keepdims=True)
#         tf_mean_future_values = lyapunov.lyapunov_function(tf_mean)
#
#         # Check whether the value is below c_max
#         tf_future_values = tf_mean_future_values + tf_error
#         tf_maps_inside = tf1.less(tf_future_values, lyapunov.c_max,
#                                  name='maps_inside_levelset')
#
#         # Put everything into storage
#         storage = [('tf_states', tf_states),
#                    ('tf_actions', tf_actions),
#                    ('tf_state_actions', tf_state_actions),
#                    ('tf_mean', tf_mean),
#                    ('tf_bound', tf_bound),
#                    ('tf_maps_inside', tf_maps_inside)]
#         set_storage(_STORAGE, storage, index=lyapunov)
#     else:
#         (tf_states, tf_actions, tf_state_actions, tf_mean, tf_bound,
#          tf_maps_inside) = storage.values()
#
#     # Subsample from all safe states within the discretization
#     safe_idx = np.where(lyapunov.safe_set)
#     safe_states = lyapunov.discretization.index_to_state(safe_idx)
#     if num_samples is not None and len(safe_states) > num_samples:
#         idx = np.random.choice(len(safe_states), num_samples, replace=True)
#         safe_states = safe_states[idx]
#
#     # Update the feed_dict accordingly
#     feed_dict = lyapunov.feed_dict
#     feed_dict[tf_states] = safe_states
#
#     if perturbations is None:
#         # Generate all state-action pairs
#         arrays = [arr.ravel() for arr in np.meshgrid(safe_states,
#                                                      actions,
#                                                      indexing='ij')]
#         state_actions = np.column_stack(arrays)
#     else:
#         # Generate state-action pairs around the current policy
#         safe_actions = tf_actions.eval(feed_dict=feed_dict)
#         state_actions = perturb_actions(safe_states,
#                                         safe_actions,
#                                         perturbations=perturbations,
#                                         limits=action_limits)
#
#     # Update feed value
#     lyapunov.feed_dict[tf_state_actions] = state_actions
#
#     # Evaluate the safety of the proposed state-action pairs
#     session = tf1.get_default_session()
#     (maps_inside, mean, bound) = session.run([tf_maps_inside, tf_mean,
#                                               tf_bound],
#                                              feed_dict=lyapunov.feed_dict)
#     maps_inside = maps_inside.squeeze(axis=1)
#
#     # Check whether states map back to the safe set in expectation
#     if not positive:
#         next_state_index = lyapunov.discretization.state_to_index(mean)
#         safe_in_expectation = lyapunov.safe_set[next_state_index]
#         maps_inside &= safe_in_expectation
#
#     # Return only state-actions pairs that are safe
#     bound_safe = bound[maps_inside]
#     if len(bound_safe) == 0:
#         # Nothing is safe, so revert to backup policy
#         msg = "No safe state-action pairs found! Using backup policy ..."
#         warnings.warn(msg, RuntimeWarning)
#         zero_perturbation = np.array([[0.]], dtype=config.np_dtype)
#         state_actions = perturb_actions(safe_states,
#                                         safe_actions,
#                                         perturbations=zero_perturbation,
#                                         limits=action_limits)
#         lyapunov.feed_dict[tf_state_actions] = state_actions
#         bound = session.run(tf_bound, feed_dict=lyapunov.feed_dict)
#         max_id = np.argmax(bound)
#         max_bound = bound[max_id].squeeze()
#         return state_actions[[max_id]], max_bound
#     else:
#         max_id = np.argmax(bound_safe)
#         max_bound = bound_safe[max_id].squeeze()
#         return state_actions[maps_inside, :][[max_id]], max_bound

def get_safe_sample(lyapunov, perturbations=None, limits=None, positive=False,
                    num_samples=None, actions=None):
    state_dim = lyapunov.discretization.ndim
    if perturbations is None:
        action_dim = actions.shape[1]
    else:
        action_dim = perturbations.shape[1]
    action_limits = limits

    safe_idx = torch.where(lyapunov.safe_set)
    safe_states = lyapunov.discretization.index_to_state(safe_idx)
    if num_samples is not None and len(safe_states) > num_samples:
        idx = torch.randint(len(safe_states), (num_samples,))
        safe_states = safe_states[idx]

    if perturbations is None:
        arrays = [arr.flatten() for arr in torch.meshgrid(safe_states, actions)]
        state_actions = torch.stack(arrays, dim=-1)
    else:
        safe_actions = lyapunov.policy(safe_states)
        state_actions = perturb_actions(safe_states, safe_actions, perturbations=perturbations, limits=action_limits)

    mean, std = lyapunov.dynamics.build_evaluation(state_actions)
    bound = torch.sum(std, dim=1, keepdim=True)
    lv = lyapunov.lipschitz_lyapunov(mean)
    error = torch.sum(lv * std, dim=1, keepdim=True)
    mean_future_values = lyapunov.lyapunov_function(mean)

    future_values = mean_future_values + error
    maps_inside = future_values < lyapunov.c_max

    if not positive:
        next_state_index = lyapunov.discretization.state_to_index(mean)
        safe_in_expectation = lyapunov.safe_set[next_state_index]
        maps_inside &= safe_in_expectation

    bound_safe = bound[maps_inside]
    if len(bound_safe) == 0:
        msg = "No safe state-action pairs found! Using backup policy ..."
        warnings.warn(msg, RuntimeWarning)
        zero_perturbation = torch.tensor([[0.]], dtype=torch.float64)
        state_actions = perturb_actions(safe_states, safe_actions, perturbations=zero_perturbation, limits=action_limits)
        bound = torch.sum(lyapunov.dynamics.build_evaluation(state_actions)[1], dim=1, keepdim=True)
        max_id = torch.argmax(bound)
        max_bound = bound[max_id].item()
        return state_actions[max_id].unsqueeze(0), max_bound
    else:
        max_id = torch.argmax(bound_safe)
        max_bound = bound_safe[max_id].item()
        return state_actions[maps_inside][max_id].unsqueeze(0), max_bound