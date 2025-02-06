from abc import ABC

import numpy as np
import xarray as xr

import calviper.math.tools as tools
import toolviper.utils.logger as logger

from calviper.base import JonesMatrix
from typing import TypeVar, Type, Union

T = TypeVar('T', bound='Parent')


class GainJones(JonesMatrix, ABC):
    def __init__(self):
        super(GainJones, self).__init__()

        # Public parent variable
        self.n_times = None
        self.type: Union[str, None] = "G"

        #self.dtype = np.complex64
        self.n_polarizations: Union[int, None] = 4
        self.n_parameters: Union[int, None] = None
        self.n_baselines: Union[int, None] = None
        self.n_channels: Union[int, None] = None
        self.channel_dependent_parameters: bool = False

        # Private variables
        self._parameters = None
        self._matrix = None
        self._antenna_map = None

        self.name: str = "GainJonesMatrix"
    '''
    # This is just an example of how this would be done. There should certainly be checks and customization
    # but for now just set the values simply as the original code doesn't do anything more complicated for now.
    @property
    def parameters(self) -> np.ndarray:
        return self._parameters

    @parameters.setter
    def parameters(self, array: np.ndarray) -> None:
        self._parameters = array
    '''
    @property
    def matrix(self) -> np.ndarray:
        return self._matrix

    @matrix.setter
    def matrix(self, array: np.ndarray) -> None:
        #(self.n_times, self.n_baselines, self.n_channels, _, _) = array.shape

        # There should be a check on the shape here. I don't think we want to allow, for instance,
        # an axis to be averaged while also having the dimensions stored in the object not change.
        self._matrix = array
    '''
    def calculate(self) -> None:
        #self.initialize_jones()

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

        shape = dataset.VISIBILITY.shape

        # This will encode the antenna values into an integer list.
        index, antennas = tools.encode(dataset.baseline_antenna1_name.to_numpy())

        instance = cls()

        # There should be a gain value for each independent antenna. Here we choose antenna_1 names but either
        # would work fine.
        instance.n_antennas = np.unique(dataset.baseline_antenna1_name).shape[0]

        # With no polarization and one channel, n_parameters = n_antennas
        # instance.n_parameters = n_parameters
        instance.n_parameters = instance.n_antennas * instance.n_polarizations

        polarization_axis_ = int(instance.n_polarizations // 2)

        identity = np.identity(polarization_axis_, dtype=np.complex64)

        instance._antenna_map = {i: str(antenna) for i, antenna in enumerate(antennas)}

        instance.n_times, instance.n_baselines, instance.n_channels, instance.n_polarizations = shape

        # Initialize default parameters
        instance.parameters = np.empty((instance.n_times, instance.n_channels, instance.n_parameters),
                                       dtype=np.complex64)

        # Build on the above idea ... wrong as they may be. Simplicity first.
        # instance.matrix = np.tile(identity, reps=[*shape, 1, 1])
        instance.matrix = np.tile(identity, reps=[instance.n_times, instance.n_baselines, instance.n_channels, 1, 1])

        return instance
    '''