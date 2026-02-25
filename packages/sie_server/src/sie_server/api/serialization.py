"""Custom serialization for SIE Server.

Provides msgpack serialization with native numpy support for efficient wire format.
Falls back to JSON for debugging when Accept header requests it.

Per DESIGN.md Section 5.9:
- Use msgpack with msgpack-numpy for arrays (not list comprehensions)
- Single call serialization, no Python loops
"""

from typing import Any

import msgpack
import msgpack_numpy as m
import numpy as np
from fastapi import Response

# Patch msgpack for numpy support
m.patch()


class MsgPackResponse(Response):
    """FastAPI Response that serializes to msgpack with numpy support.

    Content-Type: application/msgpack
    """

    media_type = "application/msgpack"

    def __init__(
        self,
        content: Any = None,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        media_type: str | None = None,
        background: Any = None,
    ) -> None:
        # Process dict (TypedDict) for msgpack serialization
        if isinstance(content, dict):
            content = _prepare_for_msgpack(content)

        # Serialize with msgpack-numpy
        body = msgpack.packb(content, use_bin_type=True)
        super().__init__(
            content=body,
            status_code=status_code,
            headers=headers,
            media_type=media_type or self.media_type,
            background=background,
        )


def _prepare_for_msgpack(d: dict[str, Any]) -> dict[str, Any]:
    """Prepare dict for msgpack serialization, preserving numpy arrays.

    TypedDict (which we use for responses) is already a dict.
    This just handles nested structures.
    """
    result: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _prepare_for_msgpack(v)
        elif isinstance(v, list):
            result[k] = [_prepare_for_msgpack(item) if isinstance(item, dict) else item for item in v]
        else:
            # numpy arrays are preserved as-is for msgpack-numpy
            result[k] = v
    return result


def _convert_for_json(d: dict[str, Any]) -> dict[str, Any]:
    """Convert dict for JSON serialization, converting numpy arrays to lists.

    JSON doesn't support numpy arrays natively, so we need to convert them.
    """
    result: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            result[k] = v.tolist()
        elif isinstance(v, dict):
            result[k] = _convert_for_json(v)
        elif isinstance(v, list):
            result[k] = [_convert_for_json(item) if isinstance(item, dict) else item for item in v]
        else:
            result[k] = v
    return result


def deserialize_msgpack(data: bytes) -> dict[str, Any]:
    """Deserialize msgpack data with numpy support.

    Args:
        data: msgpack bytes.

    Returns:
        Deserialized dict with numpy arrays restored.
    """
    result: dict[str, Any] = msgpack.unpackb(data, raw=False)
    return result
