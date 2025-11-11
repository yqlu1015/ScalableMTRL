# Ensure vendored distutils from setuptools is available under Python 3.12+.
# Some third-party dependencies (e.g., dockerpycreds via wandb) still import
# the legacy `distutils` module, which was removed from the standard library.
# Importing setuptools patches sys.modules to make distutils available.
try:
    import setuptools  # noqa: F401
except ImportError:
    pass  # setuptools should be in dependencies, but fail gracefully if not


