import numpy as np
import xarray as xr

import toolviper.utils.logger as logger
import toolviper.utils.parameter

from abc import ABC
from abc import abstractmethod

from calviper.factory import accessor

from typing import Union

from xarray import Dataset


class BaseJonesMatrix(ABC):

    # Base calibration matrix abstract class
    @abstractmethod
    def generate(self, coords: dict) -> Union[xr.Dataset, None]:
        pass


class JonesFactory(ABC):
    # Base factory class for matrix factory
    @abstractmethod
    def create_jones(self, factory: Union[None, str]):
        pass

@accessor.register_subclass
class GainMatrixDataset(BaseJonesMatrix):

    # This is intended to be an implementation of a gain jones simulator. It is
    # currently very rough and filled with random numbers. Generally based on the
    # original cal.py
    def generate(self, coords: dict) -> None:
        '''
        shape = tuple(value.shape[0] for value in coords.values())

        dims = {}
        for key, value in coords.items():
            dims[key] = value.shape[0]

        parameter = np.random.uniform(-np.pi, np.pi, shape)
        amplitude = np.random.normal(1.0, 0.1, shape)
        parameter = np.vectorize(complex)(
            np.cos(parameter),
            np.sin(parameter)
        )

        xds = xr.Dataset()

        xds["PARAMETER"] = xr.DataArray(amplitude * parameter, dims=dims)
        xds = xds.assign_coords(coords)

        return GainMatrix(xds)
        '''
        logger.info("This function is not implemented yet. Look forward to it in the future.")

    @staticmethod
    def empty_like(dataset: xr.Dataset) -> Dataset:

        antenna = np.union1d(dataset.baseline_antenna1_name.values, dataset.baseline_antenna2_name.values)
        polarizations = np.unique([p for value in dataset.polarization.values for p in list(value)])

        dims = dict(
            time=dataset.sizes["time"],
            antenna=antenna.shape[0],
            frequency=dataset.sizes["frequency"],
            polarization=polarizations.shape[0],
        )

        coords = dict(
            time=(["time"], dataset.time.values),
            antenna=(["antenna"], antenna),
            frequency=(["frequency"], dataset.frequency.values),
            polarization=(["polarization"], polarizations),
            scan_id=(["scan_id"], dataset.scan_number.values),
            baseline_id=(["baseline_id"], dataset.baseline_id.values),
        )

        xds = xr.Dataset()

        xds["PARAMETER"] = xr.DataArray(
            np.empty(list(dims.values())),
            dims=dims
        )

        xds["WEIGHT"] = xr.DataArray(
            np.empty(list(dims.values())),
            dims=dims
        )

        xds["FLAG"] = xr.DataArray(
            np.empty(list(dims.values())),
            dims=dims
        )

        xds.attrs["calibration_type"] = "gain"
        xds.attrs["observation_info"] = dataset.attrs["observation_info"]

        xds = xds.assign_coords(coords)

        return xds


class CalibrationMatrix(JonesFactory, ABC):

    def __init__(self):
        self.factory_list = {
            "gain": GainMatrixDataset,
        }

    @toolviper.utils.parameter.validate()
    def create_jones(self, factory: str) -> Union[BaseJonesMatrix, None]:
        try:
            return self.factory_list[factory]()

        except KeyError:
            logger.error(f"Factory method, {factory} not implemented.")
            return None
