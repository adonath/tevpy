# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from astropy import units as u
from .utils import get_random_state, sample_powerlaw

__all__ = ["InverseCDFSampler", "distribute_coords_log"]


class InverseCDFSampler:
    """Inverse CDF sampler.

    It determines a set of random numbers and calculate the cumulative
    distribution function.

    Parameters
    ----------
    pdf : `~gammapy.maps.Map`
        Map of the predicted source counts.
    axis : int
        Axis along which sampling the indexes.
    random_state : {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}
        Defines random number generator initialisation.
        Passed to `~gammapy.utils.random.get_random_state`.
    distribute : bool
        Whether to distribute the sampled coords uniformly in the bin.
    """

    def __init__(self, pdf, axis=None, random_state=0, distribute=True):
        self.random_state = get_random_state(random_state)
        self.axis = axis
        self.distribute = distribute

        if axis is not None:
            self.cdf = np.cumsum(pdf, axis=self.axis)
            self.cdf /= self.cdf[:, [-1]]
        else:
            self.pdf_shape = pdf.shape

            pdf = pdf.ravel() / pdf.sum()
            self.sortindex = np.argsort(pdf, axis=None)

            self.pdf = pdf[self.sortindex]
            self.cdf = np.cumsum(self.pdf)

    def sample_axis(self):
        """Sample along a given axis.

        Returns
        -------
        index : tuple of `~numpy.ndarray`
            Coordinates of the drawn sample.
        """
        choice = self.random_state.uniform(high=1, size=len(self.cdf))

        # find the indices corresponding to this point on the CDF
        index = np.argmin(np.abs(choice.reshape(-1, 1) - self.cdf), axis=self.axis)

        if self.distribute:
            index = index + self.random_state.uniform(low=-0.5, high=0.5, size=len(self.cdf))

        return index

    def sample(self, size):
        """Draw sample from the given PDF.

        Parameters
        ----------
        size : int
            Number of samples to draw.

        Returns
        -------
        index : tuple of `~numpy.ndarray`
            Coordinates of the drawn sample.
        """
        # pick numbers which are uniformly random over the cumulative distribution function
        choice = self.random_state.uniform(high=1, size=size)

        # find the indices corresponding to this point on the CDF
        index = np.searchsorted(self.cdf, choice)
        index = self.sortindex[index]

        # map back to multi-dimensional indexing
        index = np.unravel_index(index, self.pdf_shape)
        index = np.vstack(index)

        if self.distribute:
            index = index + self.random_state.uniform(low=-0.5, high=0.5, size=index.shape)
        return index


def distribute_coords_log(coords, pdf, axis):
    """Distribute coordinates using a log-log interpolation

    Parameters
    ----------
    coords : `MapCoord`
        Event coordinates
    pdf : `Map`
        PDF map.
    axis : `MapAxis`
        MapAxis along which to interpolate.
    """
    axis_idx = pdf.geom.get_axis_index_by_name(axis.name)
    data = -np.gradient(np.log(pdf.data), np.log(axis.center.value), axis=axis_idx)
    index_pwl = pdf.copy(data=data)

    gamma = index_pwl.get_by_coord(coords)

    idx = axis.coord_to_idx(coords["energy"])
    edges = axis.edges.value
    x_min, x_max = edges[idx], edges[idx + 1]
    values = sample_powerlaw(
        x_min=x_min,
        x_max=x_max,
        gamma=gamma,
        size=gamma.size
    )
    return u.Quantity(values, unit=axis.unit)
