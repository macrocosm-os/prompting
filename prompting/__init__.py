__version__ = "2.5.1"
version_split = __version__.split(".")
__spec_version__ = (10000 * int(version_split[0])) + (100 * int(version_split[1])) + (1 * int(version_split[2]))
