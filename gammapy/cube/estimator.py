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
    geom_irf : `~gammapy.maps.MapGeom`
        Map geometry used for the exposure, energy and psf map, defines
        the energy axis in true energy and typically has a coarse
        spatial resolution.
    offset_max : `~astropy.units.Quantity`
        Maximum offset cut.
    safe_energy_range: `~astropy.units.Quantity`
        Safe energy range.
    """
    default_units = {
        "exposure" : "m2 s",
        "psf": "sr-1"
    }

    def __init__(self, observation, geom, geom_irf=None, offset_max=None, safe_energy_range=None):
        self.observation = observation
        self.geom = geom
        self.geom_irf = geom_irf
        self.offset_max = Angle(offset_max)
        self.safe_energy_range = safe_energy_range

    @property
    def ontime(self):
        """On time"""
        return self.observation.observation_live_time_duration

    def _offset(self, geom):
        geom_image = geom.to_image()
        coords = geom_image.get_coord()
        pointing = self.observation.pointing_radec
        offset = pointing.separation(coords.skycoord)
        return offset

    @lazyproperty
    def offset(self):
        return self._offset(self.geom)

    @lazyproperty
    def offset_irf(self):
        """Offset array"""
        return self._offset(self.geom_irf)

    @lazyproperty
    def skycoord(self):
        coord = self.geom.to_image().get_coord()
        return coord.skycoord

    @lazyproperty
    def fov_coords(self):
        pointing = self.observation.pointing_radec
        frame = SkyOffsetFrame(origin=pointing)
        pseudo_fov_coord = self.skycoord.transform_to(frame)
        fov_lon = pseudo_fov_coord.lon
        fov_lat = pseudo_fov_coord.lat
        return fov_lon, fov_lat

    @lazyproperty
    def fov_coords_alt_az(self):
        """FoV coordinates"""
        pointing = self.observation.fixed_pointing_info
        coord_altaz = self.skycoord.transform_to(pointing.altaz_frame)

        # Compute FOV coordinates of map relative to pointing
        return sky_to_fov(
                coord_altaz.az, coord_altaz.alt, pointing.altaz.az, pointing.altaz.alt
            )

    def estimate_counts(self):
        """Estimate counts map.

        Parameters
        ----------
        events : `EventList`
            List of events.
        geom :

        Returns
        -------
        counts : `Map`
            Counts map.
        """
        counts = Map.from_geom(self.geom)
        coord = self.observation.events._get_coord_from_geom(self.geom)
        counts.fill_by_coord(coord)
        return counts

    def estimate_background(self):
        """Estimate background map.

        Returns
        -------
        background : `Map`
            Background map.
        """
        # Get altaz coords for map
        energies = self.geom.get_axis_by_name("energy").edges

        bkg_coordsys = self.observation.bkg.meta.get("FOVALIGN", "ALTAZ")

        if bkg_coordsys == "ALTAZ":
            fov_lon, fov_lat = self.fov_coords_alt_az
        elif bkg_coordsys == "RADEC":
            fov_lon, fov_lat = self.fov_coords
        else:
            raise ValueError(
                'Found unknown background coordinate system definition: "{}". '
                'Should be "ALTAZ" or "RADEC".'.format(bkg_coordsys)
            )

        bkg_de = self.observation.bkg.evaluate_integrate(
            fov_lon=fov_lon,
            fov_lat=fov_lat,
            energy_reco=energies[:, np.newaxis, np.newaxis],
        )

        d_omega = self.geom.solid_angle()
        data = (bkg_de * d_omega * self.ontime).to_value("")

        background = Map.from_geom(self.geom, data=data)
        return background

    def estimate_exposure(self):
        """Estimate exposure map.

        Returns
        -------
        exposure : `Map`
            Exposure map.
        """
        energy = self.geom_irf.get_axis_by_name("energy").center
        exposure = self.observation.aeff.data.evaluate(
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

    def run(self, steps="all"):
        """"""
        kwargs = {}

        if steps == "all":
            steps = ["counts", "exposure", "background", "psf", "edisp", "mask_safe"]

        if "counts" in steps:
            kwargs["counts"] = self.estimate_counts()

        if "exposure" in steps:
            kwargs["exposure"] = self.estimate_exposure()

        if "background" in steps:
            background = self.estimate_background()
            kwargs["background_model"] = BackgroundModel(background)

        if "psf" in steps:
            kwargs["psf"] = self.estimate_psf()
            # TODO: set reference to exposure map?

        if "edisp" in steps:
            kwargs["edisp"] = self.estimate_edisp()
            # TODO: set reference to exposure map?

        if "mask_safe" in steps:
            kwargs["mask_safe"] = self.estimate_mask_safe()

        return MapDataset(**kwargs)


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

    def __init__(self, geom, geom_irf, offset_max=None, background_estimator=None):
        self.geom = geom
        self.geom_irf = geom_irf
        self.offset_max = offset_max
        self.background_estimator = background_estimator

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
        dataset_stacked = MapDataset.from_geoms(geom, geom_irf)

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
        # TODO: add parallelisation using multiprocessing.Pool
        datasets = []
        for obs in observations:
            # for now assume aligned WCS, but later reprojection can be used
            geom = self.geom.cutout()
            geom_irf = self.geom_irf.cutout()
            estimator = MapDatasetEstimator(geom, geom_irf)

            if self.background_estimator is not None:
                dataset = estimator.run(dataset)

            datasets.append(dataset)

        return datasets

    def _run_map_dataset_estimator(config, steps):
        estimator = MapDatasetEstimator(**config)
        return estimator.run(steps=steps)

    def run_write(self, observations, steps="all", path="datasets", overwrite=False, n_jobs=4):
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
        # TODO: add parallelisation using multiprocessing.Pool
        for obs in observations:
            geom = self.geom.cutout()
            estimator = MapDatasetEstimator(geom, geom_irf)
            dataset = estimator.run(obs=obs, steps=steps)
            filename = Path(path) / {"obs_{}.fits".format(obs.obs_id)}
            dataset.write(filename, overwrite=overwrite)


# needed?
class BackgroundNormEstimator:
    ""

    def __init__(self, exclusion_mask=None, method="fit"):
        self.exclusion_mask = exclusion_mask

    @staticmethod
    def fit(dataset):
        # fit background norm
        dataset.parameters.freeze_all()
        dataset.background_model.norm.frozen = False
        dataset.mask_fit = None
        fit = Fit(dataset)
        result = fit.optimize()
        return dataset

    @staticmethod
    def scale(dataset):
        # scale background outside exclusion regions
        return dataset

    def run(self, dataset):
        """Run estimator
        """
        datsaet = self.fit(dataset)
        dataset = self.scale(dataset)
        return dataset