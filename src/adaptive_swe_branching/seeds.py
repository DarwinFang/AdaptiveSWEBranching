from __future__ import annotations

import hashlib


def derive_seed(root_seed: int, *identity: object) -> int:
    """Derive a stable 31-bit seed independent of execution order."""
    text = "\0".join([str(root_seed), *(str(part) for part in identity)])
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)
