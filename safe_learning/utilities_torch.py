from __future__ import absolute_import, division, print_function

import itertools
import inspect
from functools import wraps, partial

import numpy as np
import scipy.interpolate
import scipy.linalg

import torch
# tf1.disable_v2_behavior()

from future.builtins import zip, range
from future.backports import OrderedDict

from safe_learning import config
# import configuration as config

def batchify(arrays, batch_size):
    """Yield the arrays in batches and in order.

    The last batch might be smaller than batch_size.

    Parameters
    ----------
    arrays : list of ndarray
        The arrays that we want to convert to batches.
    batch_size : int
        The size of each individual batch.
    """
    if not isinstance(arrays, (list, tuple)):
        arrays = (arrays,)

    # Iterate over array in batches
    for i, i_next in zip(itertools.count(start=0, step=batch_size),
                         itertools.count(start=batch_size, step=batch_size)):

        batches = [array[i:i_next] for array in arrays]

        # Break if there are no points left
        if batches[0].size:
            yield i, batches
        else:
            break

def combinations(arrays):
    """Return a single array with combinations of parameters.

    Parameters
    ----------
    arrays : list of np.array

    Returns
    -------
    array : np.array
        An array that contains all combinations of the input arrays
    """
    return np.array(np.meshgrid(*arrays)).T.reshape(-1, len(arrays))


# def concatenate_inputs(start=0):
#     """Concatenate the torch.tensor or numpy array inputs to the functions.
#
#     Parameters
#     ----------
#     start : int, optional
#         The attribute number at which to start concatenating.
#     """
#
#     def wrap(function):
#         @wraps(function)
#         def wrapped_function(*args, **kwargs):
#             """Concatenate the input arguments."""
#             nargs = len(args) - start
#             # Check for torch.tensor objects or numpy arrays
#             if any(isinstance(arg, (torch.Tensor, np.ndarray)) for arg in args[start:]):
#                 # Convert numpy arrays to torch tensors if any
#                 args = tuple(torch.from_numpy(arg) if isinstance(arg, np.ndarray) else arg for arg in args)
#
#                 # concatenate extra arguments
#                 if nargs > 1:
#                     concatenated_args = torch.cat(args[start:], dim=1)
#                     args = args[:start] + (concatenated_args,)
#                 return function(*args, **kwargs)
#             else:
#                 # Assuming all inputs are of the same type (either all torch Tensors or numpy arrays)
#                 # If it's only one argument, there is no need to concatenate
#                 if nargs == 1:
#                     return function(*args, **kwargs)
#                 else:
#                     # If the inputs are numpy arrays, use np.hstack to concatenate
#                     if isinstance(args[start], np.ndarray):
#                         concatenated = (np.hstack(args[start:]),)
#                         args = args[:start] + concatenated
#                     else:
#                         # Otherwise, if inputs are torch Tensors, use torch.cat
#                         concatenated_args = torch.cat([arg.unsqueeze(0) for arg in args[start:]], dim=1)
#                         args = args[:start] + (concatenated_args,)
#                     return function(*args, **kwargs)
#
#         return wrapped_function
#
#     return wrap

# def concatenate_inputs(start=0):
#     """Concatenate the numpy array inputs to the functions.
#
#     Parameters
#     ----------
#     start : int, optional
#         The attribute number at which to start concatenating.
#     """
#     def wrap(function):
#         @wraps(function)
#         def wrapped_function(*args, **kwargs):
#             """Concatenate the input arguments."""
#             nargs = len(args) - start
#             # Check for tensorflow objects
#             torch_objects = (torch.Tensor)
#             if any(isinstance(arg, torch_objects) for arg in args[start:]):
#                 # reduce number of function calls in graph
#                 if nargs == 1:
#                     return function(*args, **kwargs)
#                 # concatenate extra arguments
#                 concatenated_args = torch.cat(args[start:], dim=1)
#                 args = args[:start] + (concatenated_args,)
#                 return function(*args, **kwargs)
#             else:
#                 # Map to 2D objects
#                 to_concatenate = map(np.atleast_2d, args[start:])
#
#                 if nargs == 1:
#                     concatenated = tuple(to_concatenate)
#                 else:
#                     concatenated = (np.hstack(to_concatenate),)
#
#                 args = args[:start] + concatenated
#                 return function(*args, **kwargs)
#
#         return wrapped_function
#
#     return wrap

def concatenate_inputs(start=0):
    """Concatenate the numpy array inputs to the functions.

    Parameters
    ----------
    start : int, optional
        The attribute number at which to start concatenating.
    """
    def wrap(function):
        @wraps(function)
        def wrapped_function(*args, **kwargs):
            """Concatenate the input arguments."""
            nargs = len(args) - start
            torch_objects = (torch.Tensor)
            if any(isinstance(arg, torch_objects) for arg in args[start:]):
                # reduce number of function calls in graph
                if nargs == 1:
                    return function(*args, **kwargs)
                # concatenate extra arguments
                if all(arg.dim() == 1 for arg in args[start:]):
                    concatenated_args = torch.cat(args[start:], dim=0)
                else:
                    concatenated_args = torch.cat(args[start:], dim=1)
                args = args[:start] + (concatenated_args,)
                return function(*args, **kwargs)
            else:
                # Map to 2D objects
                to_concatenate = map(np.atleast_2d, args[start:])

                if nargs == 1:
                    concatenated = tuple(to_concatenate)
                else:
                    concatenated = (np.hstack(to_concatenate),)

                args = args[:start] + concatenated
                return function(*args, **kwargs)

        return wrapped_function

    return wrap

def unique_rows(array):
    """Return the unique rows of the array.

    Parameters
    ----------
    array : ndarray
        A 2D numpy array.

    Returns
    -------
    unique_array : ndarray
        A 2D numpy array that contains all the unique rows of array.
    """
    array = np.ascontiguousarray(array)
    # Combine all the rows into a single element of the flexible void datatype
    dtype = np.dtype((np.void, array.dtype.itemsize * array.shape[1]))
    combined_array = array.view(dtype=dtype)
    # Get all the unique rows of the combined array
    _, idx = np.unique(combined_array, return_index=True)

    return array[idx]
