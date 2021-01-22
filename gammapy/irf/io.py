# Licensed under a 3-clause BSD style license - see LICENSE.rst
import  abc
from gammapy.utils.scripts import make_path
from gammapy.maps import RegionNDMap, MapAxis
from astropy.table import Table
from astropy.io import fits


__all__ = [
    "load_cta_irfs",
    "GTPSFMapReader",
    "GTPSFMapWriter",
    "GADFIRFMapReader",
    "GADFIRFMapWriter"
]


IRF_DL3_AXES_SPECIFICATION = {
    "THETA": {"name": "offset", "interp": "lin"},
    "ENERG": {"name": "energy_true", "interp": "log"},
    "ETRUE": {"name": "energy_true", "interp": "log"},
    "RAD": {"name": "rad", "interp": "lin"},
    "DETX": {"name": "fov_lon", "interp": "lin"},
    "DETY": {"name": "fov_lat", "interp": "lin"},
    "MIGRA": {"name": "migra", "interp": "lin"},
}


# The key is the class tag.
# TODO: extend the info here with the minimal header info
IRF_DL3_HDU_SPECIFICATION = {
    "bkg_3d": {
        "extname": "BACKGROUND",
        "column_name": "BKG",
        "hduclas2": "BKG",
    },
    "bkg_2d": {
        "extname": "BACKGROUND",
        "column_name": "BKG",
        "hduclas2": "BKG",
    },
    "edisp_2d": {
        "extname": "ENERGY DISPERSION",
        "column_name": "MATRIX",
        "hduclas2": "EDISP",
    },
    "psf_table": {
        "extname": "PSF_2D_TABLE",
        "column_name": "RPSF",
        "hduclas2": "PSF",
    },
    "aeff_2d": {
        "extname": "EFFECTIVE AREA",
        "column_name": "EFFAREA",
        "hduclas2": "EFF_AREA",
    }
}


IRF_MAP_HDU_SPECIFICATION = {
    "edisp_kernel_map": "edisp",
    "edisp_map": "edisp",
    "psf_map": "psf"
}


def load_cta_irfs(filename):
    """load CTA instrument response function and return a dictionary container.

    The IRF format should be compliant with the one discussed
    at http://gamma-astro-data-formats.readthedocs.io/en/latest/irfs/.

    The various IRFs are accessible with the following keys:

    - 'aeff' is a `~gammapy.irf.EffectiveAreaTable2D`
    - 'edisp'  is a `~gammapy.irf.EnergyDispersion2D`
    - 'psf' is a `~gammapy.irf.EnergyDependentMultiGaussPSF`
    - 'bkg' is a  `~gammapy.irf.Background3D`

    Parameters
    ----------
    filename : str
        the input filename. Default is

    Returns
    -------
    cta_irf : dict
        the IRF dictionary

    Examples
    --------
    Access the CTA 1DC IRFs stored in the gammapy datasets

    .. code-block:: python

        from gammapy.irf import load_cta_irfs
        cta_irf = load_cta_irfs("$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits")
        print(cta_irf['aeff'])
    """
    from .background import Background3D
    from .effective_area import EffectiveAreaTable2D
    from .energy_dispersion import EnergyDispersion2D
    from .psf.gauss import EnergyDependentMultiGaussPSF

    aeff = EffectiveAreaTable2D.read(filename, hdu="EFFECTIVE AREA")
    bkg = Background3D.read(filename, hdu="BACKGROUND")
    edisp = EnergyDispersion2D.read(filename, hdu="ENERGY DISPERSION")
    psf = EnergyDependentMultiGaussPSF.read(filename, hdu="POINT SPREAD FUNCTION")

    return dict(aeff=aeff, bkg=bkg, edisp=edisp, psf=psf)


class IRFMapWriter:
    """"""
    def __init__(self, filename, overwrite=False):
        filename = make_path(filename)
        filename.parent.mkdir(exist_ok=True, parents=True)

        self.filename = filename
        self.overwrite = overwrite

    @abc.abstractmethod
    def to_hdulist(self, irf_map):
        """"""
        pass

    def write(self, irf_map):
        """"""
        hdulist = self.to_hdulist(irf_map)
        hdulist.writeto(self.filename, overwrite=self.overwrite)


class FitsReader:
    """"""
    def __init__(self, filename):
        self.filename = make_path(filename)

    def from_hdulist(self):

    def read(self):
        pass

class GADFIRFMapWriter(IRFMapWriter):
    tag = "gadf"

    @staticmethod
    def to_hdulist(irf_map):
        """Convert to `~astropy.io.fits.HDUList`.

        Returns
        -------
        hdu_list : `~astropy.io.fits.HDUList`
            HDU list.
        """
        hdu = IRF_MAP_HDU_SPECIFICATION[irf_map.tag]
        hdulist = irf_map._irf_map.to_hdulist(hdu=hdu)
        exposure_hdu = hdu + "_exposure"

        if irf_map.exposure_map is not None:
            new_hdulist = irf_map.exposure_map.to_hdulist(hdu=exposure_hdu)
            hdulist.extend(new_hdulist[1:])

        return hdulist


class GadfIRFMapReader:
    tag = "gadf"
    pass


class GtPSFMapWriter(IRFMapWriter):
    """PSFMap FITS writer"""
    tag = "gtpsf"

    @staticmethod
    def to_hdulist(psf_map):
        """Convert PSFMap to hdulist

        Parameters
        ----------
        psf_map : `PSFMap`
            PSF map

        Returns
        -------
        hdulist : `~astropy.io.fits.HDUList`
            HDU list
        """
        psf, exposure = psf_map.psf_map, psf_map.exposure_map

        theta_hdu = psf.geom.axes["rad"].to_table_hdu(format="gtpsf")
        psf_table = psf.geom.axes["energy_true"].to_table(format="gtpsf")

        psf_table["Exposure"] = exposure.quantity.to("cm^2 s")
        psf_table["Psf"] = psf.quantity.to("sr^-1")
        psf_hdu = fits.BinTableHDU(data=psf_table, name="PSF")
        return fits.HDUList([fits.PrimaryHDU(), theta_hdu, psf_hdu])


class GtPSFMapFitsReader:
    """Read PSFMap in gtpsf format"""
    tag = "gtpsf"

    @staticmethod
    def from_hdulist(hdulist):
        """Create `EnergyDependentTablePSF` from ``gtpsf`` format HDU list.

        Parameters
        ----------
        hdulist : `~astropy.io.fits.HDUList`
            HDU list with ``THETA`` and ``PSF`` extensions.

        Returns
        -------
        psf_map : `PSFMap`
            PSF map
        """
        from .psf import PSFMap
        rad_axis = MapAxis.from_table_hdu(hdulist["THETA"], format="gtpsf")

        table = Table.read(hdulist["PSF"])
        energy_axis_true = MapAxis.from_table(table, format="gtpsf")

        # TODO: it would be good to know the position the PSF was created for
        psf_map = RegionNDMap.create(region=None, axes=[rad_axis, energy_axis_true], unit="sr-1")
        psf_map.data = table["Psf"].data

        exposure_map = RegionNDMap.create(region=None, axes=[energy_axis_true], unit="cm2 s")
        exposure_map.data = table["Exposure"].data
        return PSFMap(psf_map=psf_map, exposure_map=exposure_map)

    def read(self):
        """Read PSFMap in gtpsf format"""
        hdulist = fits.open(self.filename)
        return self.from_hdulist(hdulist)