from .defaults import _C as cfg

__all__ = ["cfg", "get_config"]


def get_config(config_file=None, overload_parameters=None):
    if config_file is not None:
        cfg.merge_from_file(config_file)

    if overload_parameters is not None:
        cfg.merge_from_list(overload_parameters)

    return cfg



