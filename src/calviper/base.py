import xarray as xr

import numpy as np

from toolviper.utils import logger
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import TypeVar, Type, Union

T = TypeVar('T', bound='Parent')


@dataclass(init=False)
class PolarizationBasis:
    name: str
    type: str
    length: int


class JonesMatrix(ABC):

    def __init__(self):
        # private parent variables
        self._parameters: Union[np.array, None] = None
        self._matrix: Union[np.array, None] = None

        # public parent variable
        self.type: Union[dict, None] = {"name":None, "value":None}
        self.dtype: Union[type, None] = None
        self.n_times: Union[int, None] = None
        self.n_antennas: Union[int, None] = None
        self.n_channels: Union[int, None] = None
        self.n_polarizations: Union[int, None] = None
        self.n_channel_matrices: Union[int, None] = None
        self.n_parameters: Union[int, None] = None
        self.caltable_name: Union[str, None] = None
        self.channel_dependent_parameters: bool = False

        self.polarization_basis: PolarizationBasis = PolarizationBasis()
        self.name: str = "BaseJonesMatrix"

    # Inherited member properties
    @property
    def shape(self) -> tuple:
        return self.n_times, self.n_antennas, self.n_channels, self.n_polarizations, self.n_parameters

    @shape.setter
    def shape(self, shape: tuple):
        # Unpack shape values
        self.n_times, self.n_antennas, self.n_channels, self.n_polarizations, self.n_parameters = shape

        # Reset parameters and matrices
        self._parameters = np.empty(shape, dtype=self.dtype)
        self._matrix = np.empty((self.n_times, self.n_antennas, self.n_channels), dtype=complex)

    @property
    @abstractmethod
    def parameters(self) -> np.ndarray:
        return self._parameters

    @parameters.setter
    @abstractmethod
    def parameters(self, array: np.array) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def matrix(self) -> np.ndarray:
        return self._matrix

    @matrix.setter
    @abstractmethod
    def matrix(self, array: np.array) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def calculate(self) -> None:
        raise NotImplementedError

    # Inherited method properties
    @classmethod
    def from_parameters(cls: Type[T], parameters: dict) -> T:
        import inspect

        obj = cls()
        updated_params = {}

        # This is a bit of a complicated way to do this BUT it should allow for a generic form of
        # from_parameters() for all child classes. I think ...
        for key, value in parameters.items():
            if key in inspect.getmembers(cls.__bases__[0], predicate=inspect.isfunction):
                updated_params[f"_{key}"] = value

            elif key in inspect.getmembers(cls, predicate=inspect.isfunction):
                updated_params[f"_{key}"] = value

            else:
                if key in cls().__dict__.keys():
                    updated_params[key] = value

                elif key in cls.__bases__[0]().__dict__.keys():
                    updated_params[key] = value

                else:
                    pass

        vars(obj).update(updated_params)

        return obj

    @classmethod
    def from_visibility(cls: Type[T], data: xr.Dataset, time_dependence: bool = False) -> T:
        # from the xarray set the n_times n_anttenas and n_channels properties
        # shape is (time, baseline_id, channel, polarization)
        # in this case expecting the data.VISIBILITY (?)
        # if a property has no shape then it is just 1 i.e. 1 time, polarization, or channel (can I even assert this universally?)
        obj = cls()

        if data['time'].shape:
            obj.n_times = len(data['time'])
        else:
            obj.n_times = 1

        if data['frequency'].shape:
            obj.n_channels = len(data['frequency'])
        else:
            obj.n_channels = 1

        if data['polarization'].shape:
            obj.n_polarizations = len(data['polarization'])
        else:
            obj.n_polarizations = 1
        # calculate n_antennas from baselines
        if data['baseline_id'].shape:
            n_baselines = len(data['baseline_id'])
        else:
            n_baselines = 1

        obj.n_antennas = int(0.5 * (np.sqrt(8 * n_baselines + 1) + 1))

        # matrix is computed VISIBILITES?
        obj.matrix = data.compute()
        
        return obj

    def initialize_parameters(self, dtype: np.dtype, shape: tuple = None):
        # Set data type
        self.type = dtype

        # Update shape is needed
        if shape is not None:
            self.shape = shape

        assert self.shape is not None, logger.error("Matrix shape is not set.")

        # Initialize the parameters to default
        self.parameters = np.ones(self.shape, dtype=dtype)

        # Reset Jones
        self.matrix = np.empty([])

    def initialize_jones(self, shape: tuple = None):
        if shape is not None:
            self.shape = shape

        self.matrix = np.identity(2, dtype=np.complex64)
        self.matrix = np.tile(self.matrix, [self.n_times, self.n_antennas, self.n_channel_matrices, 1, 1])

    def invert(self) -> Union[np.array, None]:
        if np.any(np.abs(np.linalg.det(self.matrix)) == 0.):
            logger.error(f"Jones matrix is singular: {np.linalg.det(self.matrix)}")
            return None

        return np.linalg.inv(self.matrix)

    def accumulate(self, other: Type[T]) -> T:
        # I think this could just be an overload of __mul__()
        return np.matmul(other.matrix, self.matrix, out=self.matrix)

    def apply_left(self):
        # Need to inspect use case
        pass

    def apply_right(self):
        # Need to inspect use case
        pass
