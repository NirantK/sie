"""Input types for SIE Server API (wire format).

These types define the structure of items received over the wire after msgpack
deserialization. The SDK converts flexible Python types (PIL.Image, numpy arrays,
file paths) to these wire format types before transport.

Using TypedDict for zero runtime overhead - validation is done manually where needed.
"""

from typing import Any, TypedDict, TypeGuard


class ImageInput(TypedDict, total=False):
    """Image input for multimodal models (wire format).

    On the wire, images are sent as bytes with format hint.

    Attributes:
        data: Image data as bytes.
        format: Image format hint: 'jpeg', 'png', etc. Inferred if not provided.
    """

    data: bytes
    format: str | None


class AudioInput(TypedDict, total=False):
    """Audio input for audio models (wire format).

    On the wire, audio is sent as bytes with format and sample rate metadata.

    Attributes:
        data: Audio data as bytes.
        format: Audio format: 'wav', 'mp3', etc.
        sample_rate: Sample rate in Hz.
    """

    data: bytes
    format: str | None
    sample_rate: int | None


class VideoInput(TypedDict, total=False):
    """Video input for video models (wire format).

    On the wire, video is sent as bytes with format hint.

    Attributes:
        data: Video data as bytes.
        format: Video format: 'mp4', 'webm', etc.
    """

    data: bytes
    format: str | None


class Item(TypedDict, total=False):
    """A single item to encode, score, or extract from (wire format).

    Items can contain text, images, audio, or video. Most models operate on text only,
    but multimodal models (ColPali, CLIP) can process images alongside text.

    Attributes:
        id: Optional identifier for this item. Returned in responses.
        text: Text content to process.
        images: Images for multimodal models.
        audio: Audio for audio models.
        video: Video for video models.
        metadata: Arbitrary metadata. Not used by models, returned in responses.
    """

    id: str
    text: str
    images: list[ImageInput]
    audio: AudioInput
    video: VideoInput
    metadata: dict[str, Any]


# =============================================================================
# Type Guards
# =============================================================================
# Python 3.12 removed isinstance() support for TypedDict.
# These TypeGuard functions provide runtime validation while maintaining
# type narrowing for static type checkers.


def is_image_input(obj: Any) -> TypeGuard[ImageInput]:
    """Check if obj is a valid ImageInput dict.

    TypedDict does not support isinstance() in Python 3.12+.
    This type guard validates structure and enables type narrowing.

    Args:
        obj: Object to validate.

    Returns:
        True if obj is a dict with 'data' key containing bytes.
    """
    return isinstance(obj, dict) and "data" in obj and isinstance(obj.get("data"), bytes)


def is_audio_input(obj: Any) -> TypeGuard[AudioInput]:
    """Check if obj is a valid AudioInput dict.

    Args:
        obj: Object to validate.

    Returns:
        True if obj is a dict with 'data' key containing bytes.
    """
    return isinstance(obj, dict) and "data" in obj and isinstance(obj.get("data"), bytes)


def is_video_input(obj: Any) -> TypeGuard[VideoInput]:
    """Check if obj is a valid VideoInput dict.

    Args:
        obj: Object to validate.

    Returns:
        True if obj is a dict with 'data' key containing bytes.
    """
    return isinstance(obj, dict) and "data" in obj and isinstance(obj.get("data"), bytes)


def is_item(obj: Any) -> TypeGuard[Item]:
    """Check if obj is a valid Item dict.

    Args:
        obj: Object to validate.

    Returns:
        True if obj is a dict (Item has all optional fields).
    """
    return isinstance(obj, dict)
