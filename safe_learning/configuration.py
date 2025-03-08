from __future__ import absolute_import, print_function, division
import torch
import numpy as np
# import tensorflow._api.v2.compat.v1 as tf1
# tf1.disable_v2_behavior()
"""General configuration class for dtypes."""

class Configuration(object):
    """Configuration class."""

    def __init__(self):
        """Initialization."""
        super(Configuration, self).__init__()

        # Dtype for computations
        self.dtype = torch.float64

        # Batch size for stability verification
        self.gp_batch_size = 1000

    @property
    def np_dtype(self):
        """Return the numpy dtype."""
        dtype_mapping = {
            torch.float32: np.float32,
            torch.float64: np.float64,
            torch.int32: np.int32,
            torch.int64: np.int64,
        }
        return dtype_mapping[self.dtype]

    def __repr__(self):
        """Print the parameters."""
        params = ['Configuration parameters:', '']
        for param, value in self.__dict__.items():
            params.append('{}: {}'.format(param, value.__repr__()))

        return '\n'.join(params)
