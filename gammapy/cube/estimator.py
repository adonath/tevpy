from .fit import MapDataset

import numpy as np
from astropy.utils import lazyproperty
from astropy.coordinates import Angle, SkyOffsetFrame
from pathlib import Path
from ..maps import Map, MapAxis
from ..maps.utils import edges_from_lo_hi
from .edisp_map import EDispMap
from .psf_map import PSFMap
from .models import BackgroundModel
from ..utils.coordinates import sky_to_fov


class MapDatasetEstimator:
    """Create a `MapDataset` for one observation.

    Parameters
    ----------
    geom : `~gammapy.maps.MapGeom`
        Map geometry used for counts and background map. Defines the
        energy axes in reconstructed energy and typically has fine
        spatial resolution.
    name : str
        Dataset name.
    energy_axis_true: `~MapAxis`
        True energy axis.
    rad_axis : `MapAxis`
        Rad axis for PSF map.
    migra_axis : `MapAxis`
        Migra axis for EDISP map.
    binsz_irf : str or `Quantity`
        Spatial bin size for the PSF and EDISP map.
    background_oversampling : int
        Background oversampling factor in energy.
    offset_max : `~astropy.units.Quantity`
        Maximum offset cut.
    safe_energy_range: `~astropy.units.Quantity`
        Safe energy range.
    cutout_observation : bool
        Make a cutout of the geometry for the observation.
    """
    default_units = {
        "exposure": "m2 s",
        "psf": "sr-1"
    }

    def __init__(
            self,
            geom,
            name="map-dataset",
            energy_axis_true=None,
            rad_axis=None,
            migra_axis=None,
            binsz_irf="0.2 deg",
            background_oversampling=None,
            offset_max="3 deg",
            safe_energy_range="aeff 10%",
            cutout_observation=True
    ):
        self.geom = geom
        self.name = name
        self.energy_axis_true = energy_axis_true
        self.rad_axis = rad_axis
        self.migra_axis = migra_axis
        self.binsz_irf = Angle(binsz_irf)
        self.background_oversampling = background_oversampling
        self.offset_max = Angle(offset_max)
        self.safe_energy_range = safe_energy_range
        self.cutout_observation = cutout_observation

    @lazyproperty
    def geom_image(self):
        return self.geom.to_image()

    @lazyproperty
    def geom_image_irf(self):
        return self.geom.to_image()

    @lazyproperty
    def _skycoord(self):
        coord = self.geom_image.get_coord()
        return coord.skycoord

    @lazyproperty
    def _skycoord_irf(self):
        coord = self.geom_image_irf.get_coord()
        return coord.skycoord

    @lru_cache(maxsize=1)
    def offset(self, observation):
        return self.geom.separation(observation.pointing_radec)

    @lru_cache(maxsize=1)
    def offset_irf(self, observation):
        """Offset array"""
        slices = self.geom_irf.cutout_slices(position=observation.pointing_radec, width=self.offset_max)
        skycoord = self.skycoord_irf[slices]
        return skycoord.separation(observation.pointing_radec)

    @lru_cache(maxsize=1)
    def fov_coords(self, observation):
        pointing = observation.pointing_radec
        frame = SkyOffsetFrame(origin=pointing)
        pseudo_fov_coord = self.skycoord.transform_to(frame)
        fov_lon = pseudo_fov_coord.lon
        fov_lat = pseudo_fov_coord.lat
        return fov_lon, fov_lat

    @lru_cache(maxsize=1)
    def fov_coords_alt_az(self, observation):
        """FoV coordinates"""
        pointing = observation.fixed_pointing_info
        coord_altaz = self.skycoord.transform_to(pointing.altaz_frame)

        # Compute FOV coordinates of map relative to pointing
        return sky_to_fov(
                coord_altaz.az, coord_altaz.alt, pointing.altaz.az, pointing.altaz.alt
            )

    def prepare_counts(self, observation):
        """Prepare counts map.

        Parameters
        ----------
        observation : `DataStoreObservation`
            Observation to compute counts map for.

        Returns
        -------
        counts : `Map`
            Counts map.
        """
        geom = self._cutout_geom(self.geom, observation)
        counts = Map.from_geom(geom)
        coord = observation.events._get_coord_from_geom(geom)
        counts.fill_by_coord(coord)
        return counts

    def prepare_background(self, observation):
        """Estimate background map.

        Returns
        -------
        background : `Map`
            Background map.
        """
        # Get altaz coords for map
        energies = self.geom.get_axis_by_name("energy").edges

        bkg_coordsys = observation.bkg.meta.get("FOVALIGN", "ALTAZ")

        if bkg_coordsys == "ALTAZ":
            fov_lon, fov_lat = self.fov_coords_alt_az(observation)
        elif bkg_coordsys == "RADEC":
            fov_lon, fov_lat = self.fov_coords(observation)
        else:
            raise ValueError(
                'Found unknown background coordinate system definition: "{}". '
                'Should be "ALTAZ" or "RADEC".'.format(bkg_coordsys)
            )

        bkg_de = observation.bkg.evaluate_integrate(
            fov_lon=fov_lon,
            fov_lat=fov_lat,
            energy_reco=energies[:, np.newaxis, np.newaxis],
        )

        d_omega = self.geom.solid_angle()
        ontime = observation.observation_live_time_duration
        data = (bkg_de * d_omega * ontime).to_value("")

        background = Map.from_geom(self.geom, data=data)
        return background

    def prepare_exposure(self, observation):
        """Estimate exposure map.

        Returns
        -------
        exposure : `Map`
            Exposure map.
        """
        geom = self.geom_exposure

        energy = geom.get_axis_by_name("energy").center
        exposure = observation.aeff.data.evaluate(
            offset=self.offset_irf, energy=energy[:, np.newaxis, np.newaxis]
        )

        unit = self.default_units["exposure"]
        data = (exposure * self.ontime).to_value(unit)
        exposure = Map.from_geom(geom=self.geom_irf, data=data, unit=unit)
        return exposure

    def estimate_psf(self):
        """Estimate PSF map.

        Returns
        -------
        psf : `PSFMap`
            PSF map.
        """
        # TODO: simplify implementation avoid using np.where and transposing
        #  and using MapAxis for PSF3D
        psf = self.observation.psf
        energy_axis = self.geom_irf.get_axis_by_name("energy")
        edges = edges_from_lo_hi(psf.rad_lo, psf.rad_hi)
        rad_axis = MapAxis.from_edges(edges=edges, unit="deg", name="theta")

        valid = np.where((self.offset_irf < self.offset_max))

        # Compute PSF values
        psf_values = psf.evaluate(offset=self.offset_irf[valid], energy=energy_axis.center, rad=rad_axis.center)

        # Re-order axes to be consistent with expected geometry
        psf_values = np.transpose(psf_values, axes=(2, 0, 1))

        geom = self.geom_irf.to_image().to_cube([rad_axis, energy_axis])
        # Create Map and fill relevant entries
        psfmap = Map.from_geom(geom, unit="sr-1")
        psfmap.data[:, :, valid[0], valid[1]] += psf_values.to_value(psfmap.unit)
        return PSFMap(psfmap)

    def estimate_edisp(self):
        """Estimate EDisp map.

        Returns
        -------
        edisp : `EdispMap`
            Edisp map.
        """
        edisp = self.observation.edisp
        energy_axis = self.geom_irf.get_axis_by_name("energy")

        migra_axis = edisp.data.axis("migra")

        # Compute separations with pointing position
        valid = np.where(self.offset_irf < self.offset_max)

        # Compute EDisp values
        edisp_values = edisp.data.evaluate(
            offset=self.offset_irf[valid],
            e_true=energy_axis.center[:, np.newaxis],
            migra=migra_axis.center[:, np.newaxis, np.newaxis],
        )

        # Re-order axes to be consistent with expected geometry
        edisp_values = np.transpose(edisp_values, axes=(1, 0, 2))

        geom = self.geom_irf.to_image().to_cube([migra_axis, energy_axis])
        # Create Map and fill relevant entries
        edispmap = Map.from_geom(geom, unit="")
        edispmap.data[:, :, valid[0], valid[1]] += edisp_values.to_value(edispmap.unit)
        return EDispMap(edispmap)

    def estimate_mask_safe(self):
        """Estimate safe data range mask.

        Returns
        -------
        mask : `~numpy.ndarray`
            Safe data range mask.
        """
        offset_mask = self.offset < self.offset_max

        # TODO: offset dependent safe energy range? E.g. from edisp map?
        emin, emax = None, None
        energy_mask = self.geom.energy_mask(emin=emin, emax=emax)
        return offset_mask & energy_mask

    def run(self, observation, steps="all"):
        """"""
        kwargs = {}

        if steps == "all":
            steps = ["counts", "exposure", "background", "psf", "edisp"]

        if "counts" in steps:
            kwargs["counts"] = self.prepare_counts(observation)

        if "exposure" in steps:
            kwargs["exposure"] = self.prepare_exposure(observation)

        if "background" in steps:
            background = self.estimate_background()
            kwargs["background_model"] = BackgroundModel(background)

        if "psf" in steps:
            psf = self.estimate_psf()
            psf.exposure_map = kwargs["exposure"]
            kwargs["psf"] = psf
            # TODO: set reference to exposure map?

        if "edisp" in steps:
            edisp = self.estimate_edisp()
            edisp.exposure_map = kwargs["exposure"]
            kwargs["edisp"] = edisp
            # TODO: set reference to exposure map?

        kwargs["mask_safe"] = self.estimate_mask_safe()
        kwargs["gti"] = observation.gti
        return MapDataset(**kwargs)


class EstimatorChain:
    """Estimator chain.

    Parameters
    ----------
    estimator : list of `Estimator`
        List of estimators to execute in a chain.

    """

    def __init__(self, estimators):
        self.estimators = estimators

    def run(self, observation):
        """Run estimator chain.

        """

        dataset = self.estimators[0].run(observation)

        for estimator in self.estimators[1:]:
            dataset = estimator.run(dataset)

        return dataset


class MapDatasetToSpectrumDatasetEstimator:
    """Reduce a map dataset to a spectrum dataset.

    Parameters
    ----------
    region : `SkyRegion`
        REgion to integrate over.
    apply_containment_correction : bool
        Whether to apply the containment correction of the PSF.

    """
    def __init__(self, region, apply_containment_correction=True):
        self.region = region
        self.apply_containment_correction = apply_containment_correction

    def run(self, dataset):
        """"""

        spectrum_dataset = dataset

        return spectrum_dataset


class MapDatasetToImageEstimator:
    """Prepare images.

    Parameters
    ----------
    spectrum : `SpectralModel`
        Spectral model used for weighting.
    keepdims : bool
        Keep extra dimensions.

    """
    def __init__(self, spectrum, keepdims=True):
        self.spectrum = spectrum
        self.keepdims = keepdims

    def run(self, dataset):
        """Run image preparator.

        Parameters
        ----------
        dataset : `MapDataset`
            Input map dataset.

        Returns
        -------
        dataset : `MapDataset`
            Output map dataset.
        """
        return dataset


class DatasetsEstimator:
    """Datasets estimator.

    Book-keeping class to handle dataset preparation across
    multiple observations and stacking.

    Parameters
    ----------
    geom : `~gammapy.maps.MapGeom`
        Map geometry for the counts and background map.
    geom_irf : `~gammapy.maps.MapGeom`
        Map geometry used for exposure, edisp and PSF map.
    """

    def __init__(self, preparators, time_intervals=None):
        self.preparators = preparators
        self.time_intervals = time_intervals

    def stack(self, datasets):
        """Stack datasets.

        Parameters
        ----------
        datasets : list of `MapDataset` or `SpectrumDataset`
            List of dataset objects

        Returns
        -------
        dataset_stacked : `MapDataset` or `SpectrumDataset`
            Stacked dataset.
        """
        dataset_stacked = MapDataset.create(geom=self.geom, geom_irf=self.geom_irf)

        for dataset in datasets:
            dataset_stacked.stack(dataset)

        return dataset_stacked

    def stack_read(self, obs_ids, path="datasets"):
        """Read dataset from disk and stack them.

        Parameters
        ----------
        obs_ids : list of obs_ids
            List of observations.
        filenames : list of str
            List of dataset file-names.

        Returns
        -------
        dataset_stacked : `MapDataset` or `SpectrumDataset`
            Stacked dataset.

        """
        dataset_stacked = MapDataset.from_geoms(geom, geom_irf)

        for obs_id in obs_ids:
            filename = Path(path) / {"obs_{}.fits".format(obs_id)}
            dataset = MapDataset.read(filename)
            dataset_stacked.stack(dataset)

        return dataset_stacked

    def _run_map_estimator(self, obs, steps):
        position = obs.pointing_radec
        width = 2 * self.offset_max
        geom = self.geom.cutout(position=position, width=width)
        geom_irf = self.geom_irf.cutout(position=position, width=width)
        estimator = MapDatasetEstimator(observation=obs, geom=geom, geom_irf=geom_irf, offset_max=self.offset_max)
        return estimator.run(steps=steps)

    def run(self, observations, steps="all", n_jobs=4):
        """Run datasets estimator.

        Parameters
        ----------
        observations : `~gammapy.data.Observations`
            List of observations.

        Returns
        -------
        datasets : list of `Dataset` or `Datasets`
            List of datasets.
        """
        datasets = []

        if self.time_intervals is not None:
            for time_interval in self.time_intervals:
                obs = observations.select_time(time_interval)
                dataset = self._run_map_estimator(obs, steps)
                datasets.append(dataset)
        else:
            for obs in observations:
                dataset = self._run_map_estimator(obs, steps)
                datasets.append(dataset)

        return datasets

    def run_write(self, observations, steps="all", path="datasets", overwrite=False):
        """Run datasets estimator and write datasets to disk.

        Parameters
        ----------
        observations : `~gammapy.data.Observations`
            List of observations.
        path : str
            Base path to write the datasets.
        overwrite : bool
            Overwrite existing files.
        n_jobs : int
            Number of jobs to submit.

        """
        if self.time_intervals is not None:
            for idx, time_interval in enumerate(self.time_intervals):
                obs = observations.select_time(time_interval)
                dataset = self._run_map_estimator(obs, steps)
                filename = Path(path) / "obs_{}.fits".format(idx)
                dataset.write(filename, overwrite=overwrite)
        else:
            for obs in observations:
                dataset = self._run_map_estimator(obs, steps)
                filename = Path(path) / "obs_{}.fits".format(obs.obs_id)
                dataset.write(filename, overwrite=overwrite)


class BackgroundNormEstimator:
    """

    """

    def __init__(self, exclusion_mask=None, method="fit", source_model=None):
        self.exclusion_mask = exclusion_mask
        self.method = method
        self.source_model = source_model

    def fit(self, dataset):
        # fit background norm
        dataset.parameters.freeze_all()
        dataset.background_model.norm.frozen = False
        dataset.mask_fit = None
        fit = Fit(dataset)
        result = fit.optimize()
        return dataset

    def scale(self, dataset):
        # scale background outside exclusion regions
        return dataset

    def run(self, dataset):
        """Run estimator
        """
        if self.method == "fit":
            dataset = self.fit(dataset)
        elif self.method == "scale":
            dataset = self.scale(dataset)
        return dataset