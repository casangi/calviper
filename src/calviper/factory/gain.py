import xarray as xr


@xr.register_dataset_accessor("gain")
class GainMatrix:
    def __init__(self, dataset: xr.Dataset):
        self._object = dataset

    @property
    def data(self):
        return self._object