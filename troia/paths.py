import os
from pathlib import Path

import tdpy


PATH_ENV_VAR = "TROIA_PATH"


def get_repository_path() -> Path:
    path_value = os.environ.get(PATH_ENV_VAR)
    if not path_value or not path_value.strip():
        raise EnvironmentError(f"{PATH_ENV_VAR} is required and cannot be empty.")
    return Path(path_value).expanduser().resolve()


def get_data_path() -> Path:
    return get_repository_path() / "data"


def get_visuals_path() -> Path:
    return get_repository_path() / "visuals"


def retr_pathgaia(pathbase=None):
    """Return normalized base/data/image paths for Troia Gaia side scripts."""

    if pathbase is None:
        pathbase = tdpy.retr_pathbase('troia')
    else:
        pathbase = tdpy.ensr_path(pathbase)

    return {
        'pathbase': pathbase,
        'pathdata': tdpy.ensr_path(os.path.join(pathbase, 'data')),
        'pathimag': tdpy.ensr_path(os.path.join(pathbase, 'imag')),
    }