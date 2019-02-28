# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from astropy.utils import lazyproperty
import astropy.units as u
from ..utils.fitting import Fit, Parameters
from ..stats import cash
from ..maps import Map, MapAxis
from .models import SkyModels, SkyModel

__all__ = ["MapEvaluator", "MapDataset"]


class MapDataset:
    """Perform sky model likelihood fit on maps.

    Parameters
    ----------
    model : `~gammapy.cube.models.SkyModel`
        Fit model
    counts : `~gammapy.maps.WcsNDMap`
        Counts cube
    exposure : `~gammapy.maps.WcsNDMap`
        Exposure cube
    mask : `~gammapy.maps.WcsNDMap`
        Mask to apply to the likelihood.
    psf : `~gammapy.cube.PSFKernel`
        PSF kernel
    edisp : `~gammapy.irf.EnergyDispersion`
        Energy dispersion
    background_model: `~gammapy.cube.models.BackgroundModel`
        Background model to use for the fit.
    likelihood : {"cash"}
	    Likelihood function to use for the fit.
    """

    def __init__(
        self,
        model,
        counts=None,
        exposure=None,
        mask=None,
        psf=None,
        edisp=None,
        background_model=None,
    ):
        if mask is not None and mask.data.dtype != np.dtype("bool"):
            raise ValueError("mask data must have dtype bool")

        if isinstance(model, SkyModel):
            model = SkyModels([model])

        self.model = model
        self.counts = counts
        self.exposure = exposure
        self.mask = mask
        self.psf = psf
        self.edisp = edisp
        self.background_model = background_model
        if background_model:
            self.parameters = Parameters(
                self.model.parameters.parameters +
                self.background_model.parameters.parameters
            )
        else:
            self.parameters = Parameters(self.model.parameters.parameters)

        self.setup()

    @property
    def data_shape(self):
        """Shape of the counts data"""
        return self.counts.data.shape

    def setup(self):
        """Setup `MapDataset`"""
        evaluators = []

        for component in self.model.skymodels:
            evaluator = MapEvaluator(component)
            evaluators.append(evaluator)

        self._evaluators = evaluators


    def npred(self):
        """Returns npred map (model + background)"""
        if self.background_model:
            npred_total = self.background_model.evaluate()
        else:
            npred_total = Map.from_geom(self.counts.geom)

        for evaluator in self._evaluators:
            if evaluator.needs_update:
                evaluator.update(self.exposure, self.psf, self.edisp)

            npred = evaluator.compute_npred()
            npred_total.fill_by_coord(evaluator.coords, npred.data)

        return npred_total

    def likelihood_per_bin(self):
        """Likelihood per bin given the current model parameters"""
        return cash(n_on=self.counts.data, mu_on=self.npred().data)

    def likelihood(self, parameters, mask=None):
        """Total likelihood given the current model parameters.

        Parameters
        ----------
        mask : `~numpy.ndarray`
            Mask to be combined with the dataset mask.
        """
        if self.mask is None and mask is None:
            stat = self.likelihood_per_bin()
        elif self.mask is None:
            stat = self.likelihood_per_bin()[mask]
        elif mask is None:
            stat = self.likelihood_per_bin()[self.mask.data]
        else:
            stat = self.likelihood_per_bin()[mask & self.mask.data]
        return np.sum(stat, dtype=np.float64)


class MapEvaluator:
    """Sky model evaluation on maps.

    This evaluates a sky model on a 3D map and convolves with the IRFs,
    and returns a map of the predicted counts.
    Note that background counts are not added.

    For now, we only make it work for 3D WCS maps with an energy axis.
    No HPX, no other axes, those can be added later here or via new
    separate model evaluator classes.


    Parameters
    ----------
    model : `~gammapy.cube.models.SkyModel`
        Sky model
    exposure : `~gammapy.maps.Map`
        Exposure map
    background : `~gammapy.maps.Map`
        Background map
    psf : `~gammapy.cube.PSFKernel`
        PSF kernel
    edisp : `~gammapy.irf.EnergyDispersion`
        Energy dispersion
    """

    def __init__(self, model=None, exposure=None, psf=None, edisp=None):
        self.model = model
        self.exposure = exposure
        self.psf = psf
        self.edisp = edisp

    @property
    def geom(self):
        """This will give the energy axes in e_true"""
        return self.exposure.geom

    @lazyproperty
    def geom_image(self):
        return self.geom.to_image()

    @lazyproperty
    def energy_center(self):
        """True energy axis bin centers (`~astropy.units.Quantity`)"""
        energy_axis = self.geom.get_axis_by_name("energy")
        energy = energy_axis.center * energy_axis.unit
        return energy[:, np.newaxis, np.newaxis]

    @lazyproperty
    def energy_edges(self):
        """Energy axis bin edges (`~astropy.units.Quantity`)"""
        energy_axis = self.geom.get_axis_by_name("energy")
        energy = energy_axis.edges * energy_axis.unit
        return energy[:, np.newaxis, np.newaxis]

    @lazyproperty
    def energy_bin_width(self):
        """Energy axis bin widths (`astropy.units.Quantity`)"""
        return np.diff(self.energy_edges, axis=0)

    @lazyproperty
    def lon_lat(self):
        """Spatial coordinate pixel centers.

        Returns ``lon, lat`` tuple of `~astropy.units.Quantity`.
        """
        lon, lat = self.geom_image.get_coord()
        return (u.Quantity(lon, "deg", copy=False), u.Quantity(lat, "deg", copy=False))

    @property
    def coords(self):
        """Return evaluator coords"""
        lon, lat = self.lon_lat
        if self.edisp:
            energy = self.edisp.e_reco.nodes[:, np.newaxis, np.newaxis]
        else:
            energy = self.energy_center
        return {"lon": lon.value, "lat": lat.value, "energy": energy}

    @lazyproperty
    def solid_angle(self):
        """Solid angle per pixel"""
        return self.geom.solid_angle()

    @lazyproperty
    def bin_volume(self):
        """Map pixel bin volume (solid angle times energy bin width)."""
        omega = self.solid_angle
        de = self.energy_bin_width
        return omega * de

    def update(self, exposure, psf, edisp):
        # TODO: lookup correct PSF for this component
        width = np.max(psf.psf_kernel_map.geom.width) * u.deg + self.model.evaluation_radius

        self.exposure = exposure.cutout(position=self.model.position, width=width)

        # TODO: lookup correct Edisp for this component
        self.edisp = edisp
        self.psf = psf

        # Reset cached quantities
        for cached_property in ["lon_lat", "solid_angle", "bin_volume"]:
            self.__dict__.pop(cached_property, None)

    def compute_dnde(self):
        """Compute model differential flux at map pixel centers.

        Returns
        -------
        model_map : `~gammapy.maps.Map`
            Sky cube with data filled with evaluated model values.
            Units: ``cm-2 s-1 TeV-1 deg-2``
        """
        lon, lat = self.lon_lat
        dnde = self.model.evaluate(lon, lat, self.energy_center)
        return dnde

    def compute_flux(self):
        """Compute model integral flux over map pixel volumes.

        For now, we simply multiply dnde with bin volume.
        """
        dnde = self.compute_dnde()
        volume = self.bin_volume
        flux = dnde * volume
        return flux

    def apply_exposure(self, flux):
        """Compute npred cube

        For now just divide flux cube by exposure
        """
        npred = (flux * self.exposure.quantity).to_value("")
        return self.exposure.copy(data=npred)

    def apply_psf(self, npred):
        """Convolve npred cube with PSF"""
        return npred.convolve(self.psf)

    def apply_edisp(self, npred):
        """Convolve map data with energy dispersion.

        Parameters
        ----------
        npred : `~gammapy.maps.Map`
            Predicted counts in true energy bins

        Returns
        ---------
        npred_reco : `~gammapy.maps.Map`
            Predicted counts in reco energy bins
        """
        loc = npred.geom.get_axis_index_by_name("energy")
        data = np.rollaxis(npred.data, loc, len(npred.data.shape))
        data = np.dot(data, self.edisp.pdf_matrix)
        data = np.rollaxis(data, -1, loc)
        e_reco_axis = MapAxis.from_edges(
            self.edisp.e_reco.bins, unit=self.edisp.e_reco.unit, name="energy"
        )
        geom_ereco = self.exposure.geom.to_image().to_cube(axes=[e_reco_axis])
        npred = Map.from_geom(geom_ereco, unit="")
        npred.data = data
        return npred

    def compute_npred(self):
        """
        Evaluate model predicted counts.

        Returns
        -------
        npred : `~gammapy.maps.Map`
            Predicted counts on the map (in reco energy bins)
        """
        flux = self.compute_flux()
        npred = self.apply_exposure(flux)
        if self.psf is not None:
            npred = self.apply_psf(npred)
        if self.edisp is not None:
            npred = self.apply_edisp(npred)
        return npred

    @property
    def needs_update(self):
        """"""
        if self.exposure is None:
            update = True
        else:
            position = self.model.position
            separation = self.exposure.geom.center_skydir.separation(position)
            update = separation > 0.5 * u.deg
        return update
