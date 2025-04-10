"""A package for satellite image AI data prep."""

import warnings
from importlib.metadata import version


__version__ = version(__name__)

__all__ = ['__version__']

warnings.filterwarnings('ignore', module='^zarr\\.')
