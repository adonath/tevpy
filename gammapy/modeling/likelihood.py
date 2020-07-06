# Licensed under a 3-clause BSD style license - see LICENSE.rst

__all__ = ["Likelihood"]


# TODO: get rid of this wrapper class? Or use it in a better way?
class Likelihood:
    """Wrapper of the likelihood function used by the optimiser.

    This might become superfluous if we introduce a
    generic ``Likelihood`` interface and use that directly,
    or change the ``Fit`` class to work with ``Model``
    and ``Likelihood`` objects.

    For now, this class does the translation of parameter
    values and the parameter factors the optimiser sees.

    Parameters
    ----------
    parameters : `~gammapy.modeling.Parameters`
        Parameters with starting values
    function : callable
        Likelihood function
    """


    def fcn(self, factors):
        self.parameters.set_parameter_factors(factors)
        return self.function()
