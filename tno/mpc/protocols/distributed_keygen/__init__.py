"""
Distributed key generation using Paillier homomorphic encryption.
"""

# Explicit re-export of all functionalities, such that they can be imported properly. Following
# https://www.python.org/dev/peps/pep-0484/#stub-files and
# https://mypy.readthedocs.io/en/stable/command_line.html#cmdoption-mypy-no-implicit-reexport
from .distributed_keygen import DistributedPaillier as DistributedPaillier

__version__ = "3.1.4"
