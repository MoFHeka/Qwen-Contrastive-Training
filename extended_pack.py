"""Extended token packing utilities for specific project requirements.

This module provides:
- Max Sample packing: Greedy packing with sample count limits and no-truncation logic
"""

from __future__ import annotations

from __future__ import annotations

import typing as tp
from dataclasses import dataclass
import numpy as np

if tp.TYPE_CHECKING:
    from collections.abc import Iterator

from easydel.data.transforms.pack import (
    GreedyPacker,
    PackedShardedSource,
    PackedSequence,
    PoolPacker,
    FirstFitPacker,
)
from easydel.data import ShardedDataSource
from eformer.loggings import get_logger

logger = get_logger(__name__)


@dataclass
class ContrastivePackedSequence(PackedSequence):
    """Extended PackedSequence containing contrastive learning specific fields."""

    triplet_type: np.ndarray | None = None
    position_ids: np.ndarray | None = None

    def to_dict(self) -> dict[str, np.ndarray]:
        """Convert to dictionary for training."""
        result = super().to_dict()
        if self.triplet_type is not None:
            result["triplet_type"] = self.triplet_type
        if self.position_ids is not None:
            result["position_ids"] = self.position_ids
        return result


class ContrastiveMaxSampleGreedyPacker(GreedyPacker):
    """
    Custom Packer for Contrastive Learning.

    Features:
    1. Max Sample Limit: Flushes if max_samples is reached.
    2. No Truncation: Entire sample must fit, or it starts a new pack.
    3. No Auto-EOS: Assumes input already has EOS if needed.
    4. Segment Types: Tracks specific segment types (Anchor=0, Pos=1, Neg=2...).
    5. Segment IDs: Indicates which sample each token belongs to (all tokens in a segment have the same segment_id).
    6. Attention Mask: 1D mask indicating valid (non-padding) positions, shape [seq_len].
    """

    def __init__(
        self,
        seq_length: int,
        eos_token_id: int,
        pad_token_id: int = 0,
        include_segment_ids: bool = True,
        max_samples: int = 128,
    ):
        """
        Initialize ContrastiveMaxSampleGreedyPacker.

        Args:
            seq_length: Target sequence length.
            eos_token_id: EOS token ID (kept for signature compatibility, unused for insertion).
            pad_token_id: Padding token ID.
            include_segment_ids: Whether to track segment IDs.
            max_samples: Maximum number of samples allowed per packed sequence.
        """
        super().__init__(
            seq_length=seq_length,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            include_segment_ids=include_segment_ids,
        )
        self.max_samples = max_samples

        # Additional buffers specific to this packer
        self._triplet_type_buffer: list[int] = []
        self._sample_index_buffer: list[
            int
        ] = []  # Used to generate segment_ids (indicates which sample each token belongs to)

    def _create_independent_pack(
        self, tokens: list[int], triplet_types: list[int], source_id: str | None = None
    ) -> ContrastivePackedSequence:
        """
        Create an independent pack from tokens without affecting the current buffer.

        This method uses temporary buffers to create a pack, leaving the main buffer untouched.
        """
        # Use temporary buffers
        temp_buffer = list(tokens)
        temp_triplet_type_buffer = list(triplet_types)
        temp_sample_index_buffer = [0] * len(tokens)  # Single sample, segment_id = 0
        temp_source_ids = [source_id] if source_id else []

        # Create pack using the same logic as flush_final but with temporary data
        current_len = len(temp_buffer)
        pad_len = self.seq_length - current_len

        # 1. Input IDs: Pad with pad_token_id
        input_ids = np.array(
            temp_buffer + [self.pad_token_id] * pad_len, dtype=np.int32
        )

        # 2. Segment Type: Pad with -1
        triplet_type = np.array(
            temp_triplet_type_buffer + [-1] * pad_len, dtype=np.int32
        )

        # 3. Segment IDs: All tokens belong to segment 0, padding uses -1
        # Following MaskInfo convention: -1 for padding tokens
        segment_ids = np.array(
            temp_sample_index_buffer + [-1] * pad_len, dtype=np.int32
        )

        # 4. Attention Mask
        attention_mask = np.array([1] * current_len + [0] * pad_len, dtype=np.int32)

        # 5. Position IDs: Start from 0 for the single sample
        position_ids = np.zeros(self.seq_length, dtype=np.int32)
        for i in range(current_len):
            position_ids[i] = i

        # Create Result
        result = ContrastivePackedSequence(
            input_ids=input_ids,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            triplet_type=triplet_type,
            position_ids=position_ids,
            source_ids=temp_source_ids.copy() if temp_source_ids else None,
            num_segments=1,
        )

        return result

    def add(
        self, tokens: list[int], triplet_types: list[int], source_id: str | None = None
    ) -> ContrastivePackedSequence | None:
        """
        Add tokens and segment types to the packer.

        Note: If a sample exceeds seq_length, it will be truncated to seq_length
        and packed independently (as a separate pack). The current buffer is NOT affected
        and can continue accumulating samples normally.

        Returns:
            ContrastivePackedSequence or None. Returns a pack when:
            - Buffer is flushed due to capacity limits (normal case)
            - A long sample is packed independently (independent pack path)
            Returns None when sample is added to buffer without flushing.
        """
        result = None
        sample_tokens_len = len(tokens)

        # Validate input lengths match
        if len(triplet_types) != sample_tokens_len:
            raise ValueError(
                f"tokens and triplet_types length mismatch: "
                f"tokens={sample_tokens_len}, triplet_types={len(triplet_types)}"
            )

        # Handle case where sample exceeds seq_length
        if sample_tokens_len > self.seq_length:
            logger.warning(
                f"Sample length ({sample_tokens_len}) exceeds seq_length ({self.seq_length}). "
                f"Truncating to sample_tokens_len and packing independently. "
                f"Consider using max_sample_length parameter in upstream processing."
            )

            # Truncate the long sample
            truncated_tokens = tokens[: self.seq_length]
            truncated_triplet_types = triplet_types[: self.seq_length]

            # Create independent pack without affecting current buffer
            # This allows the buffer to continue accumulating samples normally
            independent_pack = self._create_independent_pack(
                truncated_tokens, truncated_triplet_types, source_id
            )

            # Return the independent pack
            # Note: Current buffer is NOT affected, can continue adding samples normally
            return independent_pack

        # Normal case: sample fits within seq_length
        # 1. Check Fit Conditions
        # Note: We do NOT add +1 for EOS here (Requirement 2)
        fits_in_length = (len(self._buffer) + sample_tokens_len) <= self.seq_length
        fits_in_samples = self._current_segment + 1 <= self.max_samples

        # 2. Flush if it doesn't fit
        if self._buffer and (not fits_in_length or not fits_in_samples):
            result = self.flush_final()
            assert self._current_segment == 0 and len(self._buffer) == 0

        # 3. Add to buffers
        self._buffer.extend(tokens)
        self._triplet_type_buffer.extend(triplet_types)

        # Track sample index for segment_ids generation
        # All tokens in this segment (anchor, positive, negatives) belong to the same sample
        # segment_ids indicates which sample each token belongs to
        self._sample_index_buffer.extend([self._current_segment] * sample_tokens_len)

        if source_id:
            self._source_ids.append(source_id)

        self._current_segment += 1

        return result

    def flush_final(self) -> ContrastivePackedSequence | None:
        """Flush with custom padding and mask generation."""
        if not self._buffer:
            return None

        current_len = len(self._buffer)

        # Handle case where buffer exceeds seq_length (should not happen after add validation, but be safe)
        if current_len > self.seq_length:
            logger.warning(
                f"Buffer length ({current_len}) exceeds seq_length ({self.seq_length}). "
                f"Truncating to seq_length. This should not happen if add() validation works correctly."
            )
            # Truncate buffers to seq_length
            self._buffer = self._buffer[: self.seq_length]
            self._triplet_type_buffer = self._triplet_type_buffer[: self.seq_length]
            self._sample_index_buffer = self._sample_index_buffer[: self.seq_length]
            current_len = self.seq_length

        pad_len = self.seq_length - current_len

        # 1. Input IDs: Pad with pad_token_id
        input_ids = np.array(
            self._buffer + [self.pad_token_id] * pad_len, dtype=np.int32
        )

        # 2. Segment Type: Pad with -1
        triplet_type = np.array(
            self._triplet_type_buffer + [-1] * pad_len, dtype=np.int32
        )

        # 3. Segment IDs: Indicates which sample each token belongs to
        # All tokens in a segment (anchor, positive, negatives) have the same segment_id
        # Pad with -1 to indicate padding positions (following MaskInfo convention)
        # According to MaskInfo.from_segments documentation:
        # https://github.com/erfanzar/ejkernel/blob/d0c6af1ee534aa4dddce8abaeb04a661b602cf3b/ejkernel/types/mask.py#L677
        #   - Non-negative integers: segment membership (0, 1, 2, ...)
        #   - -1: padding tokens
        segment_ids = np.array(
            self._sample_index_buffer + [-1] * pad_len, dtype=np.int32
        )

        # 4. Attention Mask: 1D mask indicating valid (non-padding) positions
        # 1 for valid tokens, 0 for padding
        # Shape: [seq_len]
        attention_mask = np.array([1] * current_len + [0] * pad_len, dtype=np.int32)

        # 5. Position IDs: Each sample resets position_id starting from 0
        # For each unique segment_id, position_ids start from 0 and increment
        # Padding positions have position_id = 0
        position_ids = np.zeros(self.seq_length, dtype=np.int32)
        if self._sample_index_buffer:
            # Track position counter for each segment_id
            segment_positions: dict[int, int] = {}
            for i, seg_id in enumerate(self._sample_index_buffer):
                if seg_id not in segment_positions:
                    segment_positions[seg_id] = 0
                position_ids[i] = segment_positions[seg_id]
                segment_positions[seg_id] += 1

        # Create Result
        result = ContrastivePackedSequence(
            input_ids=input_ids,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            triplet_type=triplet_type,
            position_ids=position_ids,
            source_ids=self._source_ids.copy() if self._source_ids else None,
            num_segments=self._current_segment,
        )

        # Reset all buffers
        self._buffer = []
        self._triplet_type_buffer = []
        self._sample_index_buffer = []
        self._segment_ids = []  # Reset base class buffer just in case
        self._source_ids = []
        self._current_segment = 0

        return result


class ContrastivePackedShardedSource(PackedShardedSource):
    """
    ShardedSource that uses ContrastiveMaxSampleGreedyPacker.
    It assumes the input stream already contains 'input_ids' and 'triplet_type'.
    """

    def __init__(
        self,
        source: ShardedDataSource[dict],
        seq_length: int,
        eos_token_id: int,
        pad_token_id: int = 0,
        strategy: str = "greedy",
        num_packers: int = 4,
        include_segment_ids: bool = True,
        input_field: str = "input_ids",
        shuffle: bool = True,
        shuffle_buffer_factor: int = 10,
        seed: int | None = None,
        max_samples: int = 128,
    ):
        """
        Initialize ContrastivePackedShardedSource with full parameter parity.

        Args:
            source: Source to pack.
            seq_length: Target sequence length.
            eos_token_id: EOS token ID.
            pad_token_id: Padding token ID.
            strategy: Packing strategy (Ignored, enforced to greedy internally).
            num_packers: Number of packers (Ignored).
            include_segment_ids: Whether to include segment IDs.
            input_field: Field name containing input IDs.
            shuffle: Whether to shuffle packed sequences.
            shuffle_buffer_factor: Buffer size multiplier for shuffling.
            seed: Random seed.
            max_samples: Maximum number of samples allowed per packed sequence.
        """
        # Initialize base with compatible params
        super().__init__(
            source=source,
            seq_length=seq_length,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            strategy=strategy,
            num_packers=num_packers,
            include_segment_ids=include_segment_ids,
            input_field=input_field,
            shuffle=shuffle,
            shuffle_buffer_factor=shuffle_buffer_factor,
            seed=seed,
        )
        self._max_samples = max_samples

    def _create_packer(self):
        """Create the custom ContrastiveMaxSampleGreedyPacker."""
        if self._strategy == "greedy":
            sample_limit = (
                self._max_samples if self._max_samples is not None else 1000000
            )
            return ContrastiveMaxSampleGreedyPacker(
                seq_length=self._seq_length,
                eos_token_id=self._eos_token_id,
                pad_token_id=self._pad_token_id,
                include_segment_ids=self._include_segment_ids,
                max_samples=sample_limit,
            )
        elif self._strategy == "pool":
            return PoolPacker(
                seq_length=self._seq_length,
                eos_token_id=self._eos_token_id,
                pad_token_id=self._pad_token_id,
                include_segment_ids=self._include_segment_ids,
            )
        elif self._strategy == "first_fit":
            return FirstFitPacker(
                seq_length=self._seq_length,
                eos_token_id=self._eos_token_id,
                pad_token_id=self._pad_token_id,
                include_segment_ids=self._include_segment_ids,
            )
        else:
            raise ValueError(f"Invalid strategy: {self._strategy}")

    def open_shard(self, shard_name: str) -> Iterator[dict]:
        """
        Overridden open_shard to handle 'triplet_type' extraction.
        """
        import random  # Ensure local import availability

        if self._seed is not None:
            random.seed(self._seed)

        packer = self._create_packer()
        shuffle_buffer = []
        max_buffer = self._shuffle_buffer_factor * 100

        def emit(packed: ContrastivePackedSequence):
            result = packed.to_dict()
            if self._shuffle:
                if len(shuffle_buffer) < max_buffer:
                    shuffle_buffer.append(result)
                    return None
                else:
                    idx = random.randrange(0, max_buffer)
                    out = shuffle_buffer[idx]
                    shuffle_buffer[idx] = result
                    return out
            return result

        for source_shard in self._source.shard_names:
            for example in self._source.open_shard(source_shard):
                # We expect the transform to have flattened the structure
                tokens = example.get(self._input_field, [])  # Use input_field param
                triplet_types = example.get("triplet_type", [])

                if not tokens:
                    continue

                source_id = example.get("__source__")

                # Custom add call with triplet_types
                result = packer.add(list(tokens), list(triplet_types), source_id)

                if result is not None:
                    out = emit(result)
                    if out is not None:
                        yield out

        # Flush final
        final = packer.flush_final()
        if final is not None:
            out = emit(final)
            if out is not None:
                yield out

        # Emit buffer
        if self._shuffle:
            random.shuffle(shuffle_buffer)
            yield from shuffle_buffer
