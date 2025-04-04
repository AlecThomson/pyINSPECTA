"""SDHDF Error classes"""

from __future__ import annotations


class VerificationError(Exception):
    """Error raised if Verification fails"""

    def __init__(self, msg):
        super().__init__(self, msg)
