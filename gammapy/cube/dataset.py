from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.utils import lazyproperty
from astropy import units as u
from ..utils.fitting import Parameters, Parameter, Model
from ..maps import Map, MapAxis
from ..stats import cash, cstat


class IRFSkyModel(object):
    """Model bundling a `SkyModel` wit its corresponding IRFs.

    Parameters
    ----------
    model : `~gammapy.cube.models.SkyModel`
        Sky model
    exposure : `~gammapy.maps.Map`
        Exposure map
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

    @lazyproperty
    def geom(self):
        """This will give the energy axes in e_true"""
        return self.exposure.geom

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
    def lon_lat(self):
        """Spatial coordinate pixel centers.

        Returns ``lon, lat`` tuple of `~astropy.units.Quantity`.
        """
        lon, lat = self.geom.to_image().get_coord()
        return (u.Quantity(lon, "deg", copy=False), u.Quantity(lat, "deg", copy=False))

    @lazyproperty
    def solid_angle(self):
        """Solid angle per pixel"""
        return self.geom.solid_angle()

    @lazyproperty
    def bin_volume(self):
        """Map pixel bin volume (solid angle times energy bin width)."""
        omega = self.solid_angle
        return omega * np.diff(self.energy_edges, axis=0)

    @property
    def coords(self):
        energy = self.edisp.e_reco.nodes[:, np.newaxis, np.newaxis] * self.energy_edges.unit
        return {"lon": self.lon_lat[0].to_value("deg"), "lat": self.lon_lat[1].to_value("deg"), "energy": energy}

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
            self.edisp.e_reco.bins, unit=self.edisp.e_reco.unit, name="energy",
        )
        geom_ereco = self.exposure.geom.to_image().to_cube(axes=[e_reco_axis])
        npred = Map.from_geom(geom_ereco, unit="")
        npred.data = data
        return npred

    def evaluate(self):
        """
        Evaluate model predicted counts.

        Returns
        -------
        npred.data : ~numpy.ndarray
            array of the predicted counts in each bin (in reco energy)
        """
        coord = (self.lon_lat[0], self.lon_lat[1], self.energy_center)
        dnde = self.model.evaluate(*coord)
        flux = dnde * self.bin_volume
        npred = (flux * self.exposure.quantity).to_value("")
        npred = self.exposure.copy(data=npred)

        if self.psf is not None:
            npred = self.apply_psf(npred)
        if self.edisp is not None:
            npred = self.apply_edisp(npred)
        return npred


class BackgroundModel(Model):
    """Background model

    Parameters
    ----------
    background : `Map`
        Background model map
    norm : float
        Norm parameter.
    tilt : float
        Tilt parameter.
    reference : `Quantity`
        Reference energy of the tilt.
    """
    def __init__(self, background, norm=1, tilt=0, reference="1 TeV"):
        self.map = background
        self.parameters = Parameters([
            Parameter("norm", norm, unit=""),
            Parameter("tilt", tilt, unit="", frozen=True),
            Parameter("reference", reference, frozen=True),
        ])

    @lazyproperty
    def energy_center(self):
        """True energy axis bin centers (`~astropy.units.Quantity`)"""
        energy_axis = self.map.geom.get_axis_by_name("energy")
        energy = energy_axis.center
        return energy[:, np.newaxis, np.newaxis]

    def evaluate(self):
        """Evaluate background model"""
        norm = self.parameters["norm"].value
        tilt = self.parameters["tilt"].value
        reference = self.parameters["reference"].value
        tilt_factor = np.power(self.energy_center / reference, -tilt)
        data = norm * self.map.data * tilt_factor
        return self.map.copy(data=data)


class MapDataset(object):
    """ Map dataset

    Parameters
    ----------
    maps : `OrderedDict`
        Dict of maps
    model : `SourceModels`
        Source model.
    background : `BackgroundModel`
        Background model.
    mask : `Map`
        Mask to exclude regions from the likelihood computation.
    likelihood : {"cash", "cstat"}
        Choice of likelihood.
    name : str
        Name of the dataset.

    """
    def __init__(self, maps, model, background=None, mask=None, likelihood="cash", name=None):
        self.name = name
        self.maps = maps
        self.model = model
        self.background = background

        if likelihood == "cash":
            self._likelihood = cash
        elif likelihood == "cstat":
            self._likelihood = cstat
        else:
            raise ValueError()

        self.mask = mask
        self.parameters = Parameters(model.parameters.parameters +
                                     background.parameters.parameters)

    def setup(self):
        """Setup the dataset for given model and geometry.

        Parameters
        ----------
        geom : `MapGeom`
            Map geometry.
        model : `SkyModel`
            Sky model
        """
        models = []

        # geom = self.maps["counts"].geom

        psf = self.maps["psf"]
        edisp = self.maps["edisp"]

        # loop over model components and set up geometries and IRFs
        for component in self.model.skymodels:

            # geom to evaluate the model component on must be aligned with counts geom
            exposure = self.maps["exposure"].cutout(component.position, 2 * component.evaluation_radius)

            # upsample if finer spatial binning is needed
            # if component.upsample_factor:
            #    geom_comp = geom_comp.upsample(component.upsample_factor)

            # upsampled energy binnig
            # if component.energies:
            #    geom_comp = geom_comp.axes["energy"].center = e_true

            # exposure = self.maps["exposure"].reproject_to_geom(geom_comp)

            # psf kernel and psf matrix are computed such that after application
            # npred matches the counts geometry
            # psf = self.maps["psf"].get_psf_kernel(geom, geom_comp, component.position)
            # edisp = self.maps["edisp"].get_edisp_matrix(geom, geom_comp, component.position)

            model = IRFSkyModel(component, exposure, psf, edisp)
            models.append(model)

        self.irf_models = models

    @property
    def npred(self):
        """Compute total predicted number of counts."""
        if self.background:
            npred_total = self.background.evaluate()
        else:
            npred_total = Map.from_geom(self.counts.geom)

        for model in self.irf_models:
            npred = model.evaluate()
            npred_total.fill_by_coord(model.coords, npred.data)

        return npred_total

    @property
    def likelihood_per_bin(self):
        """Likelihood per bin given the current model parameters"""
        return self._likelihood(self.maps["counts"].data, mu_on=self.npred.data)

    @property
    def likelihood(self):
        """Total likelihood given the current model parameters"""
        # update parameters
        if self.mask:
            stat = self.likelihood_per_bin[self.mask.data]
        else:
            stat = self.likelihood_per_bin
        return np.sum(stat, dtype=np.float64)

    # TODO: is this the right place for the method?
    def _likelihood_to_fit(self, *factors):
        """Callback function for the fitter"""
        self.parameters.set_parameter_factors(factors)
        return self.likelihood

    # TODO: is this the right place for the method?
    def _likelihood_to_fit_sherpa(self, factors):
        """Callback function for the fitter"""
        self.parameters.set_parameter_factors(factors)
        return self.likelihood, self.likelihood_per_bin


class Datasets(object):
    """Join multiple datasets

    Parameters
    ----------
    datasets : list
        List of `Datasets` objects.
    """
    def __init__(self, datasets):
        self.datasets = datasets

    @property
    def parameters(self):
        return Parameters([dataset.parameters for dataset in self.datasets])

    @property
    def likelihood(self):
        """Compute joint likelihood"""
        total_likelihood = 0
        # TODO: add parallel evaluation of likelihoods
        for dataset in self.datasets:
            likelihood = dataset.likelihood
            total_likelihood += likelihood
        return total_likelihood


class FluxPointDataset(object):
    """Flux point dataset

    Parameters
    ----------
    flux_points : `FluxPoints`
        Flux point object.
    model : `SpectralModel`
        Spectral model object.
    mask : `numpy.ndarray`
        Mask to exclude bins form the likelihood computation.
    likelihood : {"chi2", "chi2assym"}
        Choice of likelihood.
    """
    def __init__(self, flux_points, model, mask=None, likelihood="chi2"):
        self.flux_points = flux_points
        self.model = model

    def likelihood_per_bin():
        pass

    def likelihood(self, parameters):
        pass


class SpectrumDataset(object):

    def __init__(self, data, model, edisp, likelihood="wstat"):
        self.edisp = edisp
        self.model = model

    def likelihood_per_bin():
        pass

    def likelihood(self, parameters):
        pass
