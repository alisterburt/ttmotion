"""Inter-frame motion estimation and correction for cryo-EM images."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ttmotion")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Alister Burt"
__email__ = "alisterburt@gmail.com"
