# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
from astropy.nddata.utils import NoOverlapError
from astropy.coordinates import Angle
from ..maps import Map, WcsGeom
from .counts import fill_map_counts
from .exposure import make_map_exposure_true_energy, _map_spectrum_weight
from .background import make_map_background_irf

__all__ = ["MapMaker", "MapMakerObs"]

log = logging.getLogger(__name__)




class IRFMapEstimator:
    """Estimate coarse maps for PSF, EDISP, exposure and background.

    Parameters
    ----------
    geom : `MapGeom`
        Coarse map geometry.
    """

    def __init__(self, observation, geom, theta=None, migra=None, time_interval=None):
        self.observation = observation
        self.geom = geom
        self.time_interval = time_interval
        self.theta = theta
        self.migra = migra

    @property
    def energy(self):
        energy_axis = self.geom.get_axis_by_name("energy")
        return energy_axis.center * energy_axis.unit

    @property
    def energy_edges(self):
        energy_axis = self.geom.get_axis_by_name("energy")
        return energy_axis.edges * energy_axis.unit

    @property
    def geom_image(self):
        return self.geom.to_image()

    @lazyproperty
    def offsets(self):
        """Get offset values"""
        return self.geom_image.separation(self.observation.pointing_radec)

    @lazyproperty
    def fov_lon(self):
        pass

    @lazyproperty
    def fov_lat(self):
        pass

    def estimate_psf(self):
        """"""
        psf_values = self.observation.psf.evaluate(
            offset=self.offsets,
            energy=self.energy,
            rad=self.theta)

        # Re-order axes to be consistent with expected geometry
        psf_values = np.transpose(psf_values, axes=(2, 0, 1))

        psf_map = Map.from_geom(self.psf_geom, unit="sr-1")
        psf_map.data = psf_values.to_value("sr-1")
        return PSFMap(psf_map)

    def estimate_edisp(self, observation):
        """"""
        edisp_values = self.observation.edisp.data.evaluate(
            offset=self.offsets,
            e_true=self.energy,
            migra=self.migra,
        )

        # Re-order axes to be consistent with expected geometry
        edisp_values = np.transpose(edisp_values, axes=(1, 0, 2))

        # Create Map and fill relevant entries
        edisp_map = Map.from_geom(self.edisp_geom, unit="")
        edisp_map.data = edisp_values.to_value("")
        return EDispMap(edisp_map)

    def estimate_background(self):
        return Map()

    def estimate_exposure(self):
        """"""
        aeff_values = self.observation.aeff.data.evaluate(
            offset=self.offset,
            energy=self.energy,
        )

        exposure_values = (aeff_values * self.observation.livetime).to("m2 s")

        exposure_map = Map.from_geom(self.geom, unit="")
        exposure_map.data = exposure_values.to_value("")
        return exposure_map


    def run(self):
        """Estimate observation maps"""
        maps = {}

        maps["exposure"] = self.estimate_exposure(self)
        maps["background"] = self.estimate_background(self)

        maps["psf"] = self.estimate_psf(self)
        maps["psf"].exposure = maps["exposure"]

        maps["edisp"] = self.estimate_edisp(self)
        maps["edips"].exposure = maps["edisp"]
        return maps

    @staticmethod
    def write(self, observation, maps, folder="analysis", overwrite=False):
        """Write maps to disk.

        Parameters
        ----------
        observation : `DataStoreObservation`


        """
        maps = self.run(observation)
        for name, map in maps.items():
            path = Path(folder) / "obs_{}".format(id=observation.obs_id)
            path.mkdir(parent=True, exists_ok=True)
            filename = "{name}.fits.gz".format(name=name)
            map.write(path / filename, overwrite=overwrite)

            observation.events.write(path / "events.fits.gz")


class MapEstimator:
    """Estimate `MapDataset`"""

    def __init__(self, geom, offset_max, exclusion_mask=None):
        self.geom = geom
        self.offset_max = Angle(offset_max)
        self.exclusion_mask = exclusion_mask

    def _apply_offset_max(self, map_):
        """"""
        return

    def cutout_geom(self, ):
        """"""
        position =
        return self.geom.cutout(position, width=self.offset_max)

    def estimate_counts(self, events):
        """"""
        geom = self.cutout_geom()
        counts = Map.from_geom(geom)
        fill_map_counts(counts, events)
        return counts

    def estimate_background(self, background):
        """"""
        pass


    def join(self):
        pass

    def stack(self):
        """"""
        counts = Map.from_geom(self.geom)

        for observation in observations:

            coords = self.cutout_geom().coords


        return maps

    def write(self):
        pass



class MapMaker:
    """Make maps from IACT observations.

    Parameters
    ----------
    geom : `~gammapy.maps.WcsGeom`
        Reference image geometry in reco energy
    offset_max : `~astropy.coordinates.Angle`
        Maximum offset angle
    geom_true : `~gammapy.maps.WcsGeom`
        Reference image geometry in true energy, used for exposure maps and PSF.
        If none, the same as geom is assumed
    exclusion_mask : `~gammapy.maps.Map`
        Exclusion mask
    """

    def __init__(self, geom, offset_max, geom_true=None, exclusion_mask=None):
        if not isinstance(geom, WcsGeom):
            raise ValueError("MapMaker only works with WcsGeom")

        if geom.is_image:
            raise ValueError("MapMaker only works with geom with an energy axis")

        self.geom = geom
        self.geom_true = geom_true if geom_true else geom
        self.offset_max = Angle(offset_max)
        self.maps = {}

        # Some background estimation methods need an exclusion mask.
        if exclusion_mask is not None:
            self.maps["exclusion"] = exclusion_mask


    def run(self, observations, selection=None):
        """
        Run MapMaker for a list of observations to create
        stacked counts, exposure and background maps

        Parameters
        --------------
        observations : `~gammapy.data.Observations`
            Observations to process
        selection : list
            List of str, selecting which maps to make.
            Available: 'counts', 'exposure', 'background'
            By default, all maps are made.

        Returns
        -----------
        maps: dict of stacked counts, background and exposure maps.
        """
        selection = _check_selection(selection)

        maps = {}

        # Initialise zero-filled maps
        for name in selection:
            if name == "exposure":
                maps[name] = Map.from_geom(self.geom_true, unit="m2 s")
            else:
                maps[name] = Map.from_geom(self.geom, unit="")

        for obs in observations:
            cutout = self.geom.cutout()

            try:
                maps_obs = self._process_obs(obs, selection)
            except NoOverlapError:
                log.info(
                    "Skipping observation {}, no overlap with map.".format(obs.obs_id)
                )
                continue

            # Stack observation maps to total
            for name in selection:
                data = maps_obs[name].quantity.to_value(maps[name].unit)

                if name == "exposure":
                    self.maps[name].fill_by_coord(coords_etrue, data)
                else:
                    self.maps[name].fill_by_coord(coords, data)


        return self.maps

    def _process_obs(self, obs, selection):
        # Compute cutout geometry and slices to stack results back later
        cutout_geom = self.geom.cutout(position=obs.pointing_radec, width=2 * self.offset_max, mode="trim")
        cutout_geom_etrue = self.geom_true.cutout(position=obs.pointing_radec, width=2 * self.offset_max, mode="trim")
        log.info("Processing observation: OBS_ID = {}".format(obs.obs_id))

        # Compute field of view mask on the cutout
        coords = cutout_geom.get_coord()
        offset = coords.skycoord.separation(obs.pointing_radec)
        fov_mask = offset >= self.offset_max

        # Compute field of view mask on the cutout in true energy
        coords_etrue = cutout_geom_etrue.get_coord()
        offset_etrue = coords_etrue.skycoord.separation(obs.pointing_radec)
        fov_mask_etrue = offset_etrue >= self.offset_max

        # Only if there is an exclusion mask, make a cutout
        # Exclusion mask only on the background, so only in reco-energy
        exclusion_mask = self.maps.get("exclusion", None)
        if exclusion_mask is not None:
            exclusion_mask = exclusion_mask.cutout(
                position=obs.pointing_radec, width=2 * self.offset_max, mode="trim"
            )

        # Make maps for this observation
        maps_obs = MapMakerObs(
            observation=obs,
            geom=cutout_geom,
            geom_true=cutout_geom_etrue,
            fov_mask=fov_mask,
            fov_mask_etrue=fov_mask_etrue,
            exclusion_mask=exclusion_mask,
        ).run(selection)

        return maps_obs

    def make_images(self, spectrum=None, keepdims=False):
        """Create images by summing over the energy axis.

        Exposure is weighted with an assumed spectrum,
        resulting in a weighted mean exposure image.

        Parameters
        ----------
        spectrum : `~gammapy.spectrum.models.SpectralModel`
            Spectral model to compute the weights.
            Default is power-law with spectral index of 2.

        keepdims : bool, optional
            If this is set to True, the energy axes is kept with a single bin.
            If False, the energy axes is removed

        Returns
        -------
        images : dict of `~gammapy.maps.Map`
        """
        images = {}
        for name, map in self.maps.items():
            if name == "exposure":
                map = _map_spectrum_weight(map, spectrum)

            images[name] = map.sum_over_axes(keepdims=keepdims)

        return images


class MapMakerObs:
    """Make maps for a single IACT observation.

    Parameters
    ----------
    observation : `~gammapy.data.DataStoreObservation`
        Observation
    geom : `~gammapy.maps.WcsGeom`
        Reference image geometry
    geom_true : `~gammapy.maps.WcsGeom`
        Reference image geometry in true energy, used for exposure maps and PSF.
        If none, the same as geom is assumed
    fov_mask : `~numpy.ndarray`
        Mask to select pixels in field of view
    exclusion_mask : `~gammapy.maps.Map`
        Exclusion mask (used by some background estimators)
    """

    def __init__(
            self,
            observation,
            geom,
            geom_true=None,
            fov_mask=None,
            fov_mask_etrue=None,
            exclusion_mask=None,
    ):
        self.observation = observation
        self.geom = geom
        self.geom_true = geom_true if geom_true else geom
        self.fov_mask = fov_mask
        self.fov_mask_etrue = fov_mask_etrue
        self.exclusion_mask = exclusion_mask
        self.maps = {}

    def run(self, selection=None):
        """Make maps.

        Returns dict with keys "counts", "exposure" and "background".

        Parameters
        ----------
        selection : list
            List of str, selecting which maps to make.
            Available: 'counts', 'exposure', 'background'
            By default, all maps are made.
        """
        selection = _check_selection(selection)

        for name in selection:
            getattr(self, "_make_" + name)()

        return self.maps

    def _make_counts(self):
        counts = Map.from_geom(self.geom)
        fill_map_counts(counts, self.observation.events)
        if self.fov_mask is not None:
            counts.data[..., self.fov_mask] = 0
        self.maps["counts"] = counts

    def _make_exposure(self):
        exposure = make_map_exposure_true_energy(
            pointing=self.observation.pointing_radec,
            livetime=self.observation.observation_live_time_duration,
            aeff=self.observation.aeff,
            geom=self.geom_true,
        )
        if self.fov_mask_etrue is not None:
            exposure.data[..., self.fov_mask_etrue] = 0
        self.maps["exposure"] = exposure

    def _make_background(self):
        background = make_map_background_irf(
            pointing=self.observation.fixed_pointing_info,
            ontime=self.observation.observation_time_duration,
            bkg=self.observation.bkg,
            geom=self.geom,
        )
        if self.fov_mask is not None:
            background.data[..., self.fov_mask] = 0

        # TODO: decide what background modeling options to support
        # Extra things like FOV norm scale or ring would go here.

        self.maps["background"] = background


def _check_selection(selection):
    """Handle default and validation of selection"""
    available = ["counts", "exposure", "background"]

    if selection is None:
        selection = available

    if not isinstance(selection, list):
        raise TypeError("Selection must be a list of str")

    for name in selection:
        if name not in available:
            raise ValueError("Selection not available: {!r}".format(name))

    return selection
