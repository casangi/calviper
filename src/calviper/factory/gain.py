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
    def matrix(self, data: np.ndarray) -> None:
        # (self.n_times, self.n_baselines, self.n_channels, _, _) = array.shape

        # There should be a check on the shape here. I don't think we want to allow, for instance,
        # an axis to be averaged while also having the dimensions stored in the object not change.
        self._matrix = data

    @property
    def parameter(self, data: Union[xr.DataArray, np.ndarray]) -> None:
        if type(data) == np.ndarray:
            assert len(data.shape) < 4, logger.error("Parameter dimensions can't be larger then four.")
            dims = None

    def example(self):
        logger.info("This is a gain matrix specific function that does something with the data.")

    def initialize(self):
        identity = np.identity(self._object.sizes["polarization"], dtype=self.dtype)

        matrix = xr.DataArray(
            np.tile(identity, reps=[self._object.sizes["time"], self._object.sizes["baseline_id"], self._object.sizes["frequency"], 1, 1]),
            dims=("time", "baseline_id", "frequency", "p",  "q"),
            coords={"time": self._object.time, "baseline_id": self._object.baseline_id, "frequency": self._object.frequency},
        )

        # Add the new variable to the dataset
        self._object["MATRIX"]=matrix

    def transform(self)->Union[None, np.ndarray]:

        if "baseline" in self._object.PARAMETER.dims:
            logger.info("Baseline transformation applied already.")
            return None

        _, _, n_parameter = self._object.PARAMETER.shape

        identity = np.identity(2, dtype=np.complex64)

        array = tools.to_baseline(self._object.PARAMETER.value)

        n_baseline, n_frequency, _ = array.shape

        identity_tensor = np.tile(identity, reps=[n_baseline, n_frequency, 1, 1])

        array_ = array.reshape(n_baseline, n_frequency, 2, 2)

        return identity_tensor * array_
