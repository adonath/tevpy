import numpy as np
from astropy import units as u
from .base import Map
from .geom import pix_tuple_to_idx
from .utils import INVALID_INDEX
from astropy.visualization import quantity_support


class RegionNDMap(Map):
    """"""

    def __init__(self, geom, data=None, dtype="float32", meta=None, unit=""):
        if data is None:
            data = np.zeros(geom.data_shape, dtype=dtype)

        self.geom = geom
        self.data = data
        self.meta = meta
        self.unit = u.Unit(unit)

    def plot(self, ax=None):
        """Plot map.
        """
        import matplotlib.pyplot as plt

        ax = ax or plt.gca()

        if len(self.geom.axes) > 1:
            raise TypeError("Use `.plot_interactive()` if more the one extra axis is present.")

        axis = self.geom.axes[0]
        with quantity_support():
            ax.plot(axis.center, self.quantity.squeeze())

        if axis.interp == "log":
            ax.set_xscale("log")

    @classmethod
    def create(cls, region, **kwargs):
        """
        """
        if isinstance(region, str):
            region = None

        return cls(region, **kwargs)

    def downsample(self, factor, axis=None):
        pass

    def fill_by_idx(self, idx, weights=None):
        idx = pix_tuple_to_idx(idx)

        msk = np.all(np.stack([t != INVALID_INDEX.int for t in idx]), axis=0)
        idx = [t[msk] for t in idx]

        if weights is not None:
            if isinstance(weights, u.Quantity):
                weights = weights.to_value(self.unit)
            weights = weights[msk]

        idx = np.ravel_multi_index(idx, self.data.T.shape)
        idx, idx_inv = np.unique(idx, return_inverse=True)
        weights = np.bincount(idx_inv, weights=weights).astype(self.data.dtype)
        self.data.T.flat[idx] += weights

    def get_by_idx(self, idxs):
        return self.data[idxs[::-1]]

    def interp_by_coord(self):
        raise NotImplementedError

    def interp_by_pix(self):
        raise NotImplementedError

    def set_by_idx(self, idx, value):
        self.data[idx[::-1]] = value

    def upsample(self, factor, axis=None):
        pass

    @staticmethod
    def read(cls, filename):
        pass

    def write(self, filename):
        pass

    def to_hdulist(self):
        pass

    @classmethod
    def from_hdulist(cls):
        pass

    def crop(self):
        raise NotImplementedError

    def pad(self):
        raise NotImplementedError

    def sum_over_axes(self):
        raise NotImplementedError

    def get_image_by_coord(self):
        raise NotImplementedError

    def get_image_by_idx(self):
        raise NotImplementedError

    def get_image_by_pix(self):
        raise NotImplementedError

    def stack(self, other):
        self.data += other.data