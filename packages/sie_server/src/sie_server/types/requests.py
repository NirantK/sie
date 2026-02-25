"""Request types for SIE Server API.

These types define the structure of API requests for encode, score, and extract endpoints.
Using TypedDict for zero runtime overhead - validation is done manually where needed.
"""

from typing import Any, Literal, TypedDict

from sie_server.types.inputs import Item

# Supported output types for encode endpoint
OutputType = Literal["dense", "sparse", "multivector"]

# Supported dtype for output
DType = Literal["float32", "float16", "bfloat16", "int8", "uint8", "binary", "ubinary"]


class EncodeParams(TypedDict, total=False):
    """Parameters for encode requests.

    Attributes:
        output_types: Which output types to return: 'dense', 'sparse', 'multivector'.
        instruction: Task instruction for instruction-tuned models.
        output_dtype: Output dtype: 'float32', 'float16', 'int8', 'binary'.
        options: Runtime options to override defaults.
    """

    output_types: list[OutputType]
    instruction: str | None
    output_dtype: DType | None
    options: dict[str, Any] | None


class EncodeRequest(TypedDict, total=False):
    """Request body for POST /v1/encode/{model}.

    Attributes:
        items: Items to encode (required, must be non-empty).
        params: Encoding parameters.
    """

    items: list[Item]
    params: EncodeParams | None


class ScoreRequest(TypedDict, total=False):
    """Request body for POST /v1/score/{model}.

    Used for reranking: scores each item against the query.

    Attributes:
        query: Query item to score against (required).
        items: Items to score (required, must be non-empty).
        instruction: Task instruction for instruction-tuned rerankers.
        options: Runtime options to override defaults.
    """

    query: Item
    items: list[Item]
    instruction: str | None
    options: dict[str, Any] | None


class ExtractParams(TypedDict, total=False):
    """Parameters for extract requests.

    Attributes:
        labels: Entity types for NER: ['person', 'organization', 'date'].
        output_schema: JSON schema for structured extraction.
        instruction: Task instruction for extraction.
        options: Runtime options to override defaults.
    """

    labels: list[str] | None
    output_schema: dict[str, Any] | None
    instruction: str | None
    options: dict[str, Any] | None


class ExtractRequest(TypedDict, total=False):
    """Request body for POST /v1/extract/{model}.

    Used for NER and structured extraction.

    Attributes:
        items: Items to extract from (required, must be non-empty).
        params: Extraction parameters.
    """

    items: list[Item]
    params: ExtractParams | None
