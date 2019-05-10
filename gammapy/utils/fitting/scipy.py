# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from scipy.optimize import minimize
from .likelihood import Likelihood


__all__ = ["optimize_scipy", "covariance_scipy"]


def optimize_scipy(parameters, function, **kwargs):
    method = kwargs.pop("method", "Nelder-Mead")
    pars = [par.factor for par in parameters.free_parameters]

    bounds = []
    for par in parameters.free_parameters:
        parmin = par.factor_min if not np.isnan(par.factor_min) else None
        parmax = par.factor_max if not np.isnan(par.factor_max) else None
        bounds.append((parmin, parmax))

    likelihood = Likelihood(function, parameters)
    result = minimize(likelihood.fcn, pars, bounds=bounds, method=method, **kwargs)

    factors = result.x
    info = {"success": result.success, "message": result.message, "nfev": result.nfev}
    optimizer = None

    return factors, info, optimizer


def covariance_scipy(parameters, function, **kwargs):
    from numdifftools import Hessian

    likelihood = Likelihood(function, parameters)
    hessian = Hessian(likelihood.fcn, **kwargs)

    parameter_factors = np.array([par.factor for par in parameters])
    hesse_matrix = hessian(parameter_factors)

    success, message = True, "Covariance estimation successful"

    try:
        covariance_factors = 2 * np.linalg.inv(hesse_matrix)
    except np.linalg.linalg.LinAlgError:
        success, message = False, "Hesse matrix inversion failed."
        # If normal inverse fails, try pseudo inverse
        N = len(parameters.parameters)
        covariance_factors = np.nan * np.ones((N, N))

    return covariance_factors, {"success": success, "message": message}

