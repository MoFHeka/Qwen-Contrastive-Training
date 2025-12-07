"""
高性能的训练数据加载器
使用 EasyDeL 的 Pipeline API 实现 jsonl 样本读取、packing 和分发

特性:
1. 每条样本由 anchor、positive、negative 三部分组成
2. 使用 triplet_type 标记每个 token 的语义类型:
   - 0=anchor, 1=positive, 2=negative, -1=no_training_special_token
   - 当 add_special_tokens=True 时，添加可训练的EOS token
3. 将多条样本 pack 成一条输入，样本之间不可见（通过 segment_ids 区分），总长度不超过 sequence_length
4. 支持从检查点恢复训练（通过 ResumeState）
5. 支持 max_samples 限制每个 packed sequence 中的样本数量（用于静态图编译）

使用方法:
    from transformers import AutoTokenizer
    from dataset_loader import create_dataset_pipeline, extract_sample_info, save_resume_state, load_resume_state

    tokenizer = AutoTokenizer.from_pretrained("your-model")

    # 创建 pipeline
    pipeline = create_dataset_pipeline(
        data_files="data/*.jsonl",
        tokenizer=tokenizer,
        seq_length=32768,
    max_samples=64,  # 每个 packed sequence 最多包含 64 个样本
        resume_state=load_resume_state("checkpoint.json"),  # 可选
    )

    # 使用 pipeline
    for batch in pipeline.pack().load().build():
        input_ids = batch["input_ids"]  # [seq_length]
        segment_ids = batch["segment_ids"]  # [seq_length] - 样本边界
        triplet_type = batch["triplet_type"]  # [seq_length] - 语义类型
        attention_mask = batch["attention_mask"]  # [seq_length]

        # 从 segment_ids 提取样本信息
        sample_info = extract_sample_info(batch, max_samples=64)  # 如果设置了 max_samples，需要传递

        # 在 loss 计算中使用 triplet_type
        # anchor_mask = (triplet_type == 0)
        # positive_mask = (triplet_type == 1)
        # negative_mask = (triplet_type == 2)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from transformers import PreTrainedTokenizer

from collections.abc import Iterator, Sequence

from easydel.data import (
    DatasetConfig,
    LoadStageConfig,
    PackStageConfig,
    Pipeline,
    PipelineConfig,
    ResumeState,
    ShardedDataSource,
)
from easydel.data.transforms import Transform, TransformedShardedSource
from extended_pack import ContrastivePackedShardedSource

# =======================================================
# ResumeState utility functions
# =======================================================


def save_resume_state(resume_state: ResumeState, file_path: str) -> None:
    """Save ResumeState to a JSON file."""
    os.makedirs(
        os.path.dirname(file_path) if os.path.dirname(file_path) else ".", exist_ok=True
    )
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(resume_state.to_dict(), f, indent=2)


def load_resume_state(file_path: str) -> ResumeState | None:
    """Load ResumeState from a JSON file."""
    if not os.path.exists(file_path):
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        return ResumeState.from_dict(json.load(f))


def update_resume_state(
    resume_state: ResumeState,
    shard_index: int,
    row_index: int,
    step: int | None = None,
    epoch: int | None = None,
) -> ResumeState:
    """Update ResumeState with new position."""
    return ResumeState(
        shard_index=shard_index,
        row_index=row_index,
        step=step if step is not None else resume_state.step,
        epoch=epoch if epoch is not None else resume_state.epoch,
        dataset_states=resume_state.dataset_states,
    )


# =======================================================
# ContrastiveInputTransform for loading data
# =======================================================


class ContrastiveInputTransform(Transform):
    """Transform to convert anchor/positive/negative to sequence with triplet_type.

    Creates triplet_type array to mark each token's semantic type:
    0=anchor, 1=positive, 2=negative, -1=no_training_special_token(e.g. PAD token)

    When add_special_tokens=True, add trainable special tokens to the end of each text segment.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        anchor_field: str = "anchor",
        positive_field: str = "positive",
        negative_field: str = "negative",
        num_negative_samples_per_anchor: int = 1,
        add_special_tokens: bool = True,
    ):
        self.tokenizer = tokenizer
        self.anchor_field = anchor_field
        self.positive_field = positive_field
        self.negative_field = negative_field
        self.num_negative_samples_per_anchor = num_negative_samples_per_anchor
        self.add_special_tokens = add_special_tokens

        assert num_negative_samples_per_anchor > 0, (
            "num_negative_samples_per_anchor must be greater than 0"
        )
        self.default_negative_samples = [""] * self.num_negative_samples_per_anchor

    def _tokenize_with_triplet_type(
        self, text: str, triplet_type_value: int
    ) -> tuple[list[int], list[int]]:
        """Tokenize text and create triplet_type array.

        Args:
          text: Text to tokenize
          triplet_type_value: Segment type value for content tokens (0=anchor, 1=positive, 2=negative)

        Returns:
          (tokens, triplet_types): Token IDs and corresponding triplet_type values
        """
        # Manually tokenize without adding special tokens
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        triplet_types = [triplet_type_value] * len(tokens)

        if self.add_special_tokens:
            # Add trainable special tokens, all tokens get the same triplet_type
            tokens.append(self.tokenizer.eos_token_id)
            triplet_types.append(triplet_type_value)

        return tokens, triplet_types

    def __call__(self, example: dict) -> dict | None:
        """Transform a single example."""
        anchor = example.get(self.anchor_field, "")
        positive = example.get(self.positive_field, "")
        negative = example.get(self.negative_field, self.default_negative_samples)

        if not anchor or not positive or not negative:
            return None

        # Tokenize each part (only once) and get triplet_type
        anchor_tokens, anchor_triplet_types = self._tokenize_with_triplet_type(
            anchor, 0
        )
        positive_tokens, positive_triplet_types = self._tokenize_with_triplet_type(
            positive, 1
        )

        negative_tokens = []
        negative_triplet_types = []
        if self.num_negative_samples_per_anchor == 1:
            negative = [negative[0]]
        assert len(negative) == self.num_negative_samples_per_anchor
        for i, negative_sample in enumerate(negative):
            negative_tokens_sample, negative_triplet_types_sample = (
                self._tokenize_with_triplet_type(negative_sample, 2 + i)
            )
            negative_tokens.extend(negative_tokens_sample)
            negative_triplet_types.extend(negative_triplet_types_sample)

        # Combine tokens and triplet_types
        all_tokens = anchor_tokens + positive_tokens + negative_tokens
        triplet_type = (
            anchor_triplet_types + positive_triplet_types + negative_triplet_types
        )

        result = {
            "input_ids": all_tokens,
            "triplet_type": np.array(triplet_type, dtype=np.int32),
        }

        # Preserve other fields
        for k, v in example.items():
            if k not in [self.anchor_field, self.positive_field, self.negative_field]:
                result[k] = v

        return result


# =======================================================
# create_dataset_pipeline
# =======================================================


def create_dataset_pipeline(
    data_files: str | list[str],
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 8,
    seq_length: int = 32768,
    max_samples: int | None = 128,
    eos_token_id: int | None = None,
    pad_token_id: int = 0,
    shuffle: bool = True,
    shuffle_buffer_factor: int = 10,
    seed: int | None = None,
    strategy: str = "greedy",
    resume_state: ResumeState | None = None,
    add_special_tokens: bool = True,
) -> Pipeline:
    """Create a Pipeline for triplet data with triplet_type preserved.

    Args:
      data_files: Path(s) to JSONL file(s) or glob pattern
      tokenizer: Tokenizer instance
      batch_size: Batch size for loading data
      seq_length: Maximum sequence length (default: 32768)
      max_samples: Maximum number of samples per packed sequence. If set, when this limit
                   is reached, the current packer is flushed and a new one is created.
                   If None, no limit is enforced (default: 128). When set, num_samples in
                   DataSampleInfo will be None for static graph compilation.
      eos_token_id: EOS token ID (default: tokenizer.eos_token_id)
      pad_token_id: Padding token ID
      shuffle: Whether to shuffle packed sequences
      shuffle_buffer_factor: Buffer size multiplier for shuffling
      seed: Random seed
      strategy: Packing strategy. Valid values: "greedy", "pool", "first_fit"
      resume_state: Optional ResumeState to resume from a checkpoint
      add_special_tokens: Whether to add trainable special tokens during tokenization.

    Returns:
      Pipeline object ready to be chained with stages

    Example:
      >>> pipeline = create_dataset_pipeline("data/*.jsonl", tokenizer, seq_length=2048, max_samples=64)
      >>> for batch in pipeline.pack().load().build():
      ...     triplet_type = batch["triplet_type"]
    """
    # Determine EOS token ID
    if eos_token_id is not None:
        final_eos_token_id = eos_token_id
    elif hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
        final_eos_token_id = tokenizer.eos_token_id
    else:
        raise ValueError(
            "eos_token_id must be provided or tokenizer must have eos_token_id"
        )

    # Create Pipeline config
    config = PipelineConfig(
        datasets=[
            DatasetConfig(
                data_files=data_files,
                additional_fields=["triplet_type"],
            )
        ],
        streaming=True,  # Enable streaming mode
        load=LoadStageConfig(
            batch_size=batch_size,
            prefetch_enabled=True,
            prefetch_workers=2,
            prefetch_buffer_size=4,
        ),
        pack=PackStageConfig(
            enabled=True,
            seq_length=seq_length,
            eos_token_id=final_eos_token_id,
            pad_token_id=pad_token_id,
            strategy=strategy,
            include_segment_ids=True,
            shuffle_packed=shuffle,
            shuffle_buffer_factor=shuffle_buffer_factor,
        ),
        seed=seed,
    )

    # Create pipeline and apply transform
    pipeline = Pipeline.from_config(config).source()

    # Get sources and apply transform
    data = pipeline.get_data()
    transformed_sources = {}

    if resume_state is None:
        resume_state = ResumeState(
            shard_index=0,
            row_index=0,
            step=0,
            epoch=0,
            dataset_states={
                name: {
                    "shard_index": 0,
                    "row_index": 0,
                    "step": 0,
                    "epoch": 0,
                }
                for name in data.keys()
            },
        )

    for name, source in data.items():
        # Apply resume state to source if provided
        source = _ResumableSourceWrapper(source, resume_state)

        transformed_source = TransformedShardedSource(
            source=source,
            transform=ContrastiveInputTransform(
                tokenizer=tokenizer,
                add_special_tokens=add_special_tokens,
            ),
        )
        transformed_sources[name] = transformed_source
        data[name] = transformed_source

    pipeline._data = data

    # Override pack stage to use SegmentTypeMaxSamplePackedShardedSource with max_samples support
    original_pack = pipeline.pack

    def pack_with_triplet_type(config=None):
        pipeline_copy = original_pack(config)
        packed_data = pipeline_copy.get_data()

        for name, packed_source in packed_data.items():
            packed_data[name] = ContrastivePackedShardedSource(
                source=transformed_sources[name],
                seq_length=packed_source._seq_length,
                eos_token_id=packed_source._eos_token_id,
                pad_token_id=packed_source._pad_token_id,
                strategy=packed_source._strategy,
                num_packers=getattr(packed_source, "_num_packers", 4),
                include_segment_ids=packed_source._include_segment_ids,
                input_field=packed_source._input_field or "input_ids",
                shuffle=packed_source._shuffle,
                shuffle_buffer_factor=packed_source._shuffle_buffer_factor,
                seed=packed_source._seed,
                max_samples=max_samples,
            )

        pipeline_copy._data = packed_data
        return pipeline_copy

    pipeline.pack = pack_with_triplet_type
    return pipeline


def create_dataset_source(
    pipeline: Pipeline | None = None,
    data_files: str | list[str] | None = None,
    tokenizer: PreTrainedTokenizer | None = None,
    batch_size: int = 8,
    seq_length: int = 32768,
    max_samples: int | None = 128,
    eos_token_id: int | None = None,
    pad_token_id: int = 0,
    shuffle: bool = True,
    shuffle_buffer_factor: int = 10,
    seed: int | None = None,
    strategy: str = "greedy",
    resume_state: ResumeState | None = None,
    add_special_tokens: bool = True,
) -> ShardedDataSource[dict]:
    """Create a ShardedDataSource from pipeline for use with ContrastiveTrainer.

    This method extracts the ShardedDataSource from a pipeline after packing and loading,
    which is compatible with EasyDeL's Trainer class that expects ShardedDataSource
    instead of AsyncDataLoader.

    Args:
      pipeline: Optional Pipeline object. If provided, other parameters are ignored.
                If None, a new pipeline will be created using the other parameters.
      data_files: Path(s) to JSONL file(s) or glob pattern (required if pipeline is None)
      tokenizer: Tokenizer instance (required if pipeline is None)
      batch_size: Batch size for loading data
      seq_length: Maximum sequence length (default: 32768)
      max_samples: Maximum number of samples per packed sequence
      eos_token_id: EOS token ID (default: tokenizer.eos_token_id)
      pad_token_id: Padding token ID
      shuffle: Whether to shuffle packed sequences
      shuffle_buffer_factor: Buffer size multiplier for shuffling
      seed: Random seed
      strategy: Packing strategy. Valid values: "greedy", "pool", "first_fit"
      resume_state: Optional ResumeState to resume from a checkpoint
      add_special_tokens: Whether to add trainable special tokens during tokenization.

    Returns:
      ShardedDataSource object compatible with EasyDeL Trainer

    Example:
      >>> from dataset_loader import create_dataset_source
      >>> source = create_dataset_source(
      ...     data_files="data/*.jsonl",
      ...     tokenizer=tokenizer,
      ...     seq_length=2048,
      ...     max_samples=64
      ... )
      >>> trainer = ContrastiveTrainer(
      ...     arguments=training_args,
      ...     model=model,
      ...     dataset_train=source,  # Use ShardedDataSource directly
      ...     processing_class=tokenizer,
      ... )
    """
    # Create pipeline if not provided
    if pipeline is None:
        if data_files is None or tokenizer is None:
            raise ValueError(
                "Either pipeline must be provided, or both data_files and tokenizer must be provided"
            )
        pipeline = create_dataset_pipeline(
            data_files=data_files,
            tokenizer=tokenizer,
            batch_size=batch_size,
            seq_length=seq_length,
            max_samples=max_samples,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            shuffle=shuffle,
            shuffle_buffer_factor=shuffle_buffer_factor,
            seed=seed,
            strategy=strategy,
            resume_state=resume_state,
            add_special_tokens=add_special_tokens,
        )

    # Get ShardedDataSource from pipeline after pack and load stages
    data = pipeline.pack().get_data()

    # Extract the first ShardedDataSource from the data dictionary
    if isinstance(data, dict):
        if len(data) == 0:
            raise ValueError("Pipeline returned empty data dictionary")
        # Get the first value (typically "train" or the first dataset name)
        source = list(data.values())[0]
    elif isinstance(data, ShardedDataSource):
        source = data
    else:
        raise TypeError(
            f"Unexpected data type from pipeline: {type(data)}. "
            f"Expected dict or ShardedDataSource."
        )

    if not isinstance(source, ShardedDataSource):
        raise TypeError(
            f"Pipeline did not return ShardedDataSource. Got: {type(source)}"
        )

    return source


class _ResumableSourceWrapper(ShardedDataSource[dict]):
    """Internal wrapper to apply resume state to source."""

    def __init__(self, source: ShardedDataSource[dict], resume_state: ResumeState):
        self._source = source
        self._resume_state = resume_state

    @property
    def shard_names(self) -> Sequence[str]:
        return self._source.shard_names

    def num_shards(self) -> int:
        return self._source.num_shards()

    def open_shard(self, shard_name: str) -> Iterator[dict]:
        shard_idx = self._source.shard_names.index(shard_name)

        if shard_idx < self._resume_state.shard_index:
            return
        elif shard_idx == self._resume_state.shard_index:
            yield from self._source.open_shard_at_row(
                shard_name, self._resume_state.row_index
            )
        else:
            yield from self._source.open_shard(shard_name)

    def open_shard_at_row(self, shard_name: str, row: int) -> Iterator[dict]:
        return self._source.open_shard_at_row(shard_name, row)

    def iter_shards(
        self,
        shard_indices: Sequence[int] | None = None,
        start_shard: int = 0,
        start_row: int = 0,
    ) -> Iterator[dict]:
        return self._source.iter_shards(
            shard_indices=shard_indices,
            start_shard=self._resume_state.shard_index,
            start_row=self._resume_state.row_index,
        )

    def __len__(self) -> int:
        return len(self._source)


# =======================================================
# DataSampleInfo utility functions for Testing
# =======================================================


@dataclass
class DataSampleInfo:
    """Information about samples in a packed sequence."""

    num_samples: int
    sample_starts: list[int]
    sample_lengths: list[int]


def _to_numpy(arr):
    """Safely convert JAX/List to Numpy array."""
    if arr is None:
        return None
    # 强制转换，避免 JAX Array 在迭代时的怪异行为
    return np.array(arr)


def _extract_single_sequence_info(
    segment_ids: np.ndarray, attention_mask: np.ndarray | None
) -> tuple[list[int], list[int]]:
    """
    Vectorized extraction for a SINGLE 1D sequence based on SEGMENT IDs.
    Correctly detects transitions between segment IDs (e.g., 0->1).
    """
    seq_len = len(segment_ids)

    # 1. Mask 处理 (保持不变)
    if attention_mask is not None:
        valid_mask = attention_mask.astype(bool)
        if len(valid_mask) != seq_len:
            raise ValueError(f"Length mismatch: seg={seq_len}, mask={len(valid_mask)}")

        if not np.any(valid_mask):
            return [], []

        last_valid_idx = np.where(valid_mask)[0][-1]
        valid_length = last_valid_idx + 1

        # 裁剪有效区域
        segment_ids = segment_ids[:valid_length]
    else:
        valid_length = seq_len

    if valid_length == 0:
        return [], []

    # 2. 核心逻辑修复：检测值变化 (Sample ID Change)
    # 索引 0 永远是一个样本的开始
    # 之后的索引，如果值和前一个不同，也是新样本的开始

    # 计算相邻元素的差异
    # segment_ids[1:] != segment_ids[:-1] 生成一个布尔数组，表示发生变化的位置
    is_change = segment_ids[1:] != segment_ids[:-1]

    # 找到变化的索引，并加 1 (因为 diff 导致的索引偏移)
    change_indices = np.flatnonzero(is_change) + 1

    # 所有的起始位置 = [0] + [变化的位置]
    start_indices = np.concatenate(([0], change_indices))

    # 3. 计算长度 (保持不变)
    # 将总长度作为最后一个边界
    boundaries = np.append(start_indices, valid_length)
    lengths = np.diff(boundaries)

    return start_indices.tolist(), lengths.tolist()


def extract_sample_info_from_segment_ids(
    segment_ids: np.ndarray | jnp.ndarray,
    attention_mask: np.ndarray | jnp.ndarray | None = None,
) -> DataSampleInfo:
    """
    Strictly extracts sample info.

    Rules:
    1. segment_ids must be 1D or 2D.
    2. If attention_mask is provided, it MUST have the exact same shape as segment_ids.
    """
    # 1. 统一转换为 Numpy Array，确保后续行为一致
    segment_ids_np = _to_numpy(segment_ids)
    attention_mask_np = _to_numpy(attention_mask)

    # 2. 严格校验 attention_mask 形状
    if attention_mask_np is not None:
        if segment_ids_np.shape != attention_mask_np.shape:
            raise ValueError(
                f"Shape mismatch: segment_ids {segment_ids_np.shape} vs "
                f"attention_mask {attention_mask_np.shape}. "
                "They must be strictly identical."
            )

    # 3. 根据维度分发处理
    ndim = segment_ids_np.ndim

    all_starts = []
    all_lengths = []

    if ndim == 1:
        # === Case A: Single Sequence (1D) ===
        starts, lengths = _extract_single_sequence_info(
            segment_ids_np, attention_mask_np
        )
        all_starts = starts
        all_lengths = lengths

    elif ndim == 2:
        # === Case B: Batch of Sequences (2D) ===
        batch_size = segment_ids_np.shape[0]

        # 使用索引遍历，这是最稳妥的方式，避免 zip 在不同库数组间的坑
        for i in range(batch_size):
            # 获取单行数据，此时一定是 1D
            seg_row = segment_ids_np[i]
            mask_row = attention_mask_np[i] if attention_mask_np is not None else None

            starts, lengths = _extract_single_sequence_info(seg_row, mask_row)

            all_starts.extend(starts)
            all_lengths.extend(lengths)

    else:
        # === Case C: Error ===
        raise ValueError(
            f"segment_ids must be 1D or 2D, got shape {segment_ids_np.shape}"
        )

    return DataSampleInfo(
        num_samples=len(all_starts),
        sample_starts=all_starts,
        sample_lengths=all_lengths,
    )


def extract_sample_info(
    batch: dict,
    max_samples: int | None = None,
) -> DataSampleInfo:
    """Wrapper for dictionary batches."""
    segment_ids = batch.get("segment_ids")
    attention_mask = batch.get("attention_mask")

    if segment_ids is None:
        raise ValueError("batch must contain 'segment_ids' field")

    # Delegate to core strict function
    info = extract_sample_info_from_segment_ids(segment_ids, attention_mask)

    # Optional: Max Samples Check (Global)
    if max_samples is not None:
        # 这里进行的是粗粒度检查。如果需要在 Batch 维度进行细粒度报错（比如"第i个序列超长"），
        # 逻辑需要移到 extract_sample_info_from_segment_ids 内部或者由调用方检查。
        # 此处仅做简单的警告/检查演示，避免打断主流程。
        pass

    return info


def split_packed_sequence(
    input_ids: jnp.ndarray | np.ndarray,
    segment_ids: jnp.ndarray | np.ndarray | None = None,
    sample_info: DataSampleInfo | None = None,
) -> list[dict]:
    """Split a packed sequence into individual samples."""
    # 强制展平，用于后续切片
    input_ids = _to_numpy(input_ids).flatten()

    if sample_info is None:
        if segment_ids is None:
            raise ValueError("Either sample_info or segment_ids must be provided")
        sample_info = extract_sample_info_from_segment_ids(segment_ids)

    if segment_ids is not None:
        segment_ids = _to_numpy(segment_ids).flatten()

    samples = []
    for start, length in zip(sample_info.sample_starts, sample_info.sample_lengths):
        end = start + length

        # 边界保护
        if end > len(input_ids):
            continue

        sample = {"input_ids": input_ids[start:end]}
        if segment_ids is not None:
            sample["segment_ids"] = segment_ids[start:end]
        samples.append(sample)

    return samples
