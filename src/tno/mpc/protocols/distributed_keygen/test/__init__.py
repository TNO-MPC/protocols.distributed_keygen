"""
Testing module of the tno.mpc.protocols.distributed_keygen library
"""

# Explicit re-export of all functionalities, such that they can be imported properly. Following
# https://www.python.org/dev/peps/pep-0484/#stub-files and
# https://mypy.readthedocs.io/en/stable/command_line.html#cmdoption-mypy-no-implicit-reexport
from tno.mpc.protocols.distributed_keygen.test.test_distributed_keygen import (
    fixture_distributed_schemes as fixture_distributed_schemes,
)
