"""
Constants.

Contains constants used to setup collective groups.
"""
import hashlib
import os
from enum import Enum, auto


def get_store_name(group_name):
    """Generate the unique name for the NCCLUniqueID store (named actor).

    Args:
        group_name (str): unique user name for the store.
    Return:
        str: MD5-hexlified name for the store.
    """
    if not group_name:
        raise ValueError("group_name is None.")
    return hashlib.md5(group_name.encode()).hexdigest()


class ENV(Enum):
    """Environment variables."""

    NCCL_USE_MULTISTREAM = auto(), lambda v: (v or "True") == "True"

    @property
    def val(self):
        """Return the output of the lambda against the system's env value."""
        _, default_fn = self.value  # pylint: disable=unpacking-non-sequence
        return default_fn(os.getenv(self.name))
