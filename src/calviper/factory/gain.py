from abc import ABC

import numpy as np
import xarray as xr

import calviper.math.tools as tools
import toolviper.utils.logger as logger

from calviper.factory.base import JonesMatrix
from typing import TypeVar, Type, Union

T = TypeVar('T', bound='Parent')


@xr.register_dataset_accessor("gain")
class GainMatrix(JonesMatrix, ABC):
    def __init__(self, dataset: xr.Dataset):
        super(GainMatrix, self).__init__()
        self._object = dataset

        self.type: str = "G"
        self.dtype: Type = np.complex64
        self.channel_dependent_parameters: bool = False

        self._matrix: Union[None, np.ndarray] = None

    @property
    def data(self):
        return self._object

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix

    @matrix.setter
    def matrix(self, array: np.ndarray) -> None:
        # (self.n_times, self.n_baselines, self.n_channels, _, _) = array.shape

        # There should be a check on the shape here. I don't think we want to allow, for instance,
        # an axis to be averaged while also having the dimensions stored in the object not change.
        self._matrix = array

    def example(self):
        logger.info("This is a gain matrix specific function that does something with the data.")