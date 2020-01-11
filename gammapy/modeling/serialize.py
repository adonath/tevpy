# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utilities to serialize models."""
from gammapy.modeling.models import SkyModel, SkyDiffuseCube

__all__ = ["models_to_dict", "dict_to_models"]


def models_to_dict(models):
    """Convert list of models to dict.

    Parameters
    ----------
    models : list
        Python list of Model objects
    """
    # update shared parameters names for serialization
    _rename_shared_parameters(models)

    models_data = []
    for model in models:
        model_data = model.to_dict()
        # De-duplicate if model appears several times
        if model_data not in models_data:
            models_data.append(model_data)

    # restore shared parameters names after serialization
    _restore_shared_parameters(models)

    return {"components": models_data}


def _rename_shared_parameters(models):
    params_list = []
    params_shared = []
    for model in models:
        for param in model.parameters:
            if param not in params_list:
                params_list.append(param)
            elif param not in params_shared:
                params_shared.append(param)
    for k, param in enumerate(params_shared):
        param.name = param.name + "@shared_" + str(k)


def _restore_shared_parameters(models):
    for model in models:
        for param in model.parameters:
            param.name = param.name.split("@")[0]


def dict_to_models(data, link=True):
    """De-serialise model data to Model objects.

    Parameters
    ----------
    data : dict
        Serialised model information
    link : bool
        check for shared parameters and link them
    """
    models = []
    for component in data["components"]:
        # background models are created separately
        if component["type"] == "BackgroundModel":
            continue

        if component["type"] == "SkyDiffuseCube":
            model = SkyDiffuseCube.from_dict(component)

        if component["type"] == "SkyModel":
            model = SkyModel.from_dict(component)

        models.append(model)

    if link:
        _link_shared_parameters(models)
    return models


def _link_shared_parameters(models):
    shared_register = {}
    for model in models:
        for param in model.parameters:
            name = param.name
            if "@" in name:
                if name in shared_register:
                    new_param = shared_register[name]
                    model.parameters.link(name, new_param)
                    if isinstance(model, SkyModel):
                        spatial_params = model.spatial_model.parameters
                        spectral_params = model.spectral_model.parameters
                        if name in spatial_params.names:
                            spatial_params.link(name, new_param)
                        elif name in spectral_params.names:
                            spectral_params.link(name, new_param)
                else:
                    param.name = name.split("@")[0]
                    shared_register[name] = param
