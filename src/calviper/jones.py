import numpy as np
import xarray as xr

from calviper.base import JonesMatrix
from typing import TypeVar, Type, Union

T = TypeVar('T', bound='Parent')

class GainJones(JonesMatrix):
    def __init__(self):
        super(GainJones, self).__init__()

        # Public parent variable
        self.n_times = None
        self.type: Union[str, None] = "G"
        self.dtype = np.complex64
        self.n_polarizations: Union[int, None] = 2
        self.n_parameters: Union[int, None] = 1
        self.channel_dependent_parameters: bool = False

        # Private variables
        self._parameters = None
        self._matrix = None

        self.name: str = "GainJonesMatrix"

    # This is just an example of how this would be done. There should certainly be checks and customization
    # but for now just set the values simply as the original code doesn't do anything more complicated for now.
    @property
    def parameters(self) -> np.ndarray:
        return self._parameters

    @parameters.setter
    def parameters(self, array: np.ndarray) -> None:
        self._parameters = array

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix

    @matrix.setter
    def matrix(self, array: np.ndarray) -> None:
        self._matrix = array

    def calculate(self) -> None:
        self.initialize_jones()

        self.matrix = np.identity(2, dtype=np.complex64)
        self.matrix = np.tile(self.matrix, [self.n_times, self.n_antennas, self.n_channel_matrices, 1, 1])

    @classmethod
    def from_visibility(cls: Type[T], dataset: xr.Dataset, time_dependence: bool = False) -> T:
        shape = dataset.VISIBILITY.shape

        # There should be a gain value for each independent antenna. Here we choose antenna_1 names but either
        # would work fine.
        n_parameters = np.unique(dataset.baseline_antenna1_name).shape[0]

        identity = np.identity(n_parameters, dtype=np.complex64)

        instance = cls()

        instance.n_times, instance.n_antennas, instance.n_channels, instance.n_polarizations = shape

        instance.n_parameters = n_parameters

        instance.matrix = np.tile(identity, reps=[*shape, 1, 1])

        return instance