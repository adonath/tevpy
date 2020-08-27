# Licensed under a 3-clause BSD style license - see LICENSE.rst
import abc
import numpy as np
import logging
from pathlib import Path
from gammapy.datasets import Datasets
from astropy.nddata import NoOverlapError

__all__ = ["Maker"]


log = logging.getLogger(__name__)


class Maker(abc.ABC):
    """Abstract maker base class."""

    @property
    @abc.abstractmethod
    def tag(self):
        pass

    @abc.abstractmethod
    def run(self):
        pass

    def __str__(self):
        s = f"{self.__class__.__name__}\n"
        s += "-" * (len(s) - 1) + "\n\n"

        names = self.__init__.__code__.co_varnames

        max_len = np.max([len(_) for _ in names]) + 1

        for name in names:
            value = getattr(self, name, "not available")

            if value == "not available":
                continue
            else:
                s += f"\t{name:{max_len}s}: {value}\n"

        return s.expandtabs(tabsize=2)


class DatasetsMaker:
    """Run makers in a chain

    Parameters
    ----------
    makers : list of `Maker` objects
        Makers
    path : str or `Path`
        Path
    overwrite : bool
        Whether to overwrite
    naming_scheme : str
        Naming scheme including the placeholders "obs_id" and "idx".
    constant_irfs : bool
        Option to stack IRFs if they are constant. This avoids re-computing
        the IRFs per observation, but just scale them to the stacked livetime.
    """
    def __init__(
            self,
            makers,
            path=None,
            overwrite=True,
            cutout_mode="partial",
            cutout_width="5 deg",
            n_jobs=None,
            stack=False,
            naming_scheme=None,
            constant_irfs=False
    ):
        self.makers = makers

        if path:
            path = Path(path)

        self.path = path
        self.overwrite = overwrite
        self.cutout_mode = cutout_mode
        self.cutout_width = cutout_width
        self.n_jobs = n_jobs
        self.stack = stack
        self.naming_scheme = naming_scheme
        self.constant_irfs = constant_irfs

    def make_dataset(self, dataset, observation):
        """Make single dataset.

        Parameters
        ----------
        dataset : `MapDataset`
            Map dataset
        observation : `Observation`
            Observation
        """
        log.info(f"Computing dataset for observation {observation.obs_id}")

        if self.cutout_width is not None:
            cutouts_kwargs = {
                "position": observation.pointing_radec,
                "width": self.cutout_width,
                "mode": self.cutout_mode,
            }

            dataset = dataset.cutout(cutouts_kwargs)

        for maker in self.makers:
            log.info(f"Running {maker.tag}")
            dataset = maker.run(dataset=dataset, observation=observation)

        return dataset

    def run(self, dataset, observations):
        """Make all datasets.

        Parameters
        ----------
        dataset : `MapDataset`
            Reference map dataset
        observation : `Observation`
            Observation

        Returns
        -------
        datasets : `Datasets`
            Datasets
        """
        datasets = Datasets()

        for obs in observations:
            try:
                dataset_per_obs = self.make_dataset(dataset, observation=obs)
            except NoOverlapError:
                log.info(f"No overlap for {obs.obs_id}, skipping...")
                continue

            if self.stack:
                dataset.stack(dataset_per_obs)
            else:
                datasets.append(dataset)

        return datasets

    def run_write_obs(self, observation):
        """Run and write

        Parameters
        ----------
        observation : `Observation`
            Observation
        """
        log.info(f"Computing dataset for observation {observation.obs_id}")

        try:
            # currently dataset object cannot be pickled fix...
            dataset = self.make_dataset(observation=observation)
        except NoOverlapError:
            log.info(f"No overlap for {observation.obs_id}, skipping...")

        path = self.path / f"obs-{observation.obs_id}"
        path.mkdir(exist_ok=True)

        filename = path / f"dataset.fits"
        log.info(f"Writing {filename}")
        dataset.write(filename, overwrite=self.overwrite)

        filename = path / f"bkg-model.yaml"
        log.info(f"Writing {filename}")
        dataset.models.write(
            filename, overwrite=self.overwrite, write_covariance=False
        )

    def run_write(self, observations):
        """Run and write

        Parameters
        ----------
        observations : `Observations`
            Observations
        """
        with contextlib.closing(Pool(processes=self.n_jobs)) as pool:
           log.info("Using {} jobs.".format(self.n_jobs))
           pool.map(self.run_write_obs, observations)

        pool.join()
