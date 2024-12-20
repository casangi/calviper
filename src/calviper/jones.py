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

        #self.dtype = np.complex64
        self.n_polarizations: Union[int, None] = 2
        self.n_parameters: Union[int, None] = 1
        self.channel_dependent_parameters: bool = False

        # Private variables
        self._parameters = None
        self._matrix = None
        self._antenna_map = None

        self.name: str = "GainJonesMatrix"

    # This is just an example of how this would be done. There should certainly be checks and customization
    # but for now just set the values simply as the original code doesn't do anything more complicated for now.
    @property
    def parameters(self) -> np.ndarray:
        if self._matrix:
            return self._matrix

        # Assuming we have an N-dimensional tensor of values, calculate the last two axes.
        n_axes = len(self._matrix)

        # Using diagonal() we get the diagonal values of the last two axes, so we calculate that from
        # the total number of axes.
        axis1 = n_axes - 2
        axis2 = n_axes - 1

        return self._matrix.diagonal(axis1=axis1, axis2=axis2)

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
        """
        Build Jones matrix from visibility data.
        :param dataset:
        :param time_dependence:
        :return:
        """

        # For the prototype we will pretend there is no polarization and one channel.
        import calviper.math.tools as tools

        shape = dataset.VISIBILITY.shape

        # This will encode the antenna values into an integer list.
        index, antennas = tools.encode(dataset.baseline_antenna1_name.to_numpy())

        instance = cls()

        # There should be a gain value for each independent antenna. Here we choose antenna_1 names but either
        # would work fine.
        instance.n_antennas = np.unique(dataset.baseline_antenna1_name).shape[0]


        # With no polarization and one channel, n_parameters = n_antennas
        # instance.n_parameters = n_parameters
        instance.n_parameters = instance.n_antennas

        identity = np.identity(instance.n_parameters, dtype=np.complex64)

        instance._antenna_map = {i: str(antenna) for i, antenna in enumerate(antennas)}

        instance.n_times, instance.n_baselines, instance.n_channels, instance.n_polarizations = shape

        # Build on the above idea ... wrong as they may be. Simplicity first.
        # instance.matrix = np.tile(identity, reps=[*shape, 1, 1])
        instance.matrix = np.tile(identity, reps=[instance.n_times, 1, 1])

        return instance