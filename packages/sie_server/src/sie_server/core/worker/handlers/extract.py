"""Extract operation handler.

Handles entity extraction (NER, RE) from text and images.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sie_server.core.inference_output import ExtractOutput
from sie_server.core.worker.handlers.base import OperationHandler, make_hashable

if TYPE_CHECKING:
    from sie_server.adapters.base import ModelAdapter
    from sie_server.core.batcher import HasCost
    from sie_server.core.worker.types import RequestMetadata
    from sie_server.types.inputs import Item


class ExtractHandler(OperationHandler[ExtractOutput]):
    """Handler for extract (NER, RE) operations.

    Supports:
    - Entity extraction with configurable labels
    - Structured extraction with output schema
    - Optional instructions for instruction-tuned models
    """

    def make_config_key(self, metadata: RequestMetadata) -> tuple[Any, ...]:
        """Create config key for batching extract requests.

        Items with the same (labels, instruction, options) can be batched together.

        Args:
            metadata: Request metadata.

        Returns:
            Hashable tuple for grouping.
        """
        labels_key = tuple(sorted(metadata.labels)) if metadata.labels else None
        options_key = make_hashable(metadata.options) if metadata.options else None
        return (
            labels_key,
            metadata.instruction,
            options_key,
        )

    def run_inference(
        self,
        adapter: ModelAdapter,
        items: list[Item],
        config_key: tuple[Any, ...],
        prepared_items: list[HasCost] | None,
        metadata_list: list[RequestMetadata],
    ) -> ExtractOutput:
        """Run extract inference.

        Args:
            adapter: The model adapter.
            items: Items to extract from.
            config_key: Config key tuple.
            prepared_items: Pre-processed items.
            metadata_list: Metadata (unused for extract).

        Returns:
            ExtractOutput with entities.
        """
        labels_tuple, instruction, options_tuple = config_key
        labels = list(labels_tuple) if labels_tuple else None
        options = dict(options_tuple) if options_tuple else None

        return adapter.extract(
            items,
            labels=labels,
            instruction=instruction,
            options=options,
            prepared_items=prepared_items,
        )

    def slice_output(self, output: ExtractOutput, index: int) -> ExtractOutput:
        """Extract single item from batched extract output.

        Args:
            output: Batched output.
            index: Index to extract.

        Returns:
            Single-item ExtractOutput.
        """
        return ExtractOutput(
            entities=[output.entities[index]],
            batch_size=1,
        )

    def assemble_output(
        self,
        partials: dict[int, ExtractOutput],
        batch_size: int,
    ) -> ExtractOutput:
        """Assemble partial outputs into full extract output.

        Args:
            partials: Dict mapping index to single-item output.
            batch_size: Total batch size.

        Returns:
            Full ExtractOutput.
        """
        if not partials:
            return ExtractOutput(entities=[], batch_size=0)

        entities = [partials[i].entities[0] for i in range(batch_size)]
        return ExtractOutput(entities=entities, batch_size=batch_size)

    @classmethod
    def format_output(cls, output: ExtractOutput) -> list[dict[str, Any]]:
        """Convert ExtractOutput to per-item dicts for API response."""
        return [{"entities": list(ents)} for ents in output.entities]
