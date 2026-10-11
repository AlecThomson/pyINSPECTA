"""SDHDF Error classes"""

from __future__ import annotations


class VerificationError(Exception):
    """Error raised if Verification fails"""

    def __init__(self, msg: str) -> None:
        super().__init__(self, msg)
