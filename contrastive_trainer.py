"""
Contrastive Learning Trainer for EasyDeL

This module provides a custom trainer for contrastive learning tasks, specifically
designed for embedding models that use triplet (anchor, positive, negative) data.
"""

from __future__ import annotations

import os
import json
import typing as tp

import numpy as np

import jax
from jax.sharding import PartitionSpec

import flax
import flax.nnx

from eformer.escale import with_sharding_constraint
from eformer.loggings import get_logger

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.loss_utils import LossMetrics
from easydel.trainers.trainer import Trainer
from easydel.trainers.trainer_protocol import (
    TrainerConfigureDataloaderOutput,
    TrainerConfigureFunctionOutput,
)
from easydel.trainers.training_configurations import TrainingArguments
from easydel.trainers.training_utils import (
    make_assertions_and_get_sizes,
    minibatch_call,
    update_metrics,
    update_state_respectfully,
)
from easydel.utils.compiling_utils import ejit
from easydel.data import ResumeState

from dataset_loader import _ResumableSourceWrapper

if tp.TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from easydel.data.core.protocols import ShardedDataSource
    from jax.sharding import PartitionSpec

logger = get_logger(__name__)

# Set JAX config for compilation cache (additional method)
try:
    xla_cache_dir = os.path.join(os.getcwd(), ".xla_cache")
    os.makedirs(xla_cache_dir, exist_ok=True)
    jax_cache_dir = os.path.join(os.getcwd(), ".jax_cache")
    os.makedirs(jax_cache_dir, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", jax_cache_dir)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_enable_xla_caches", xla_cache_dir)
except Exception as e:
    print(f"⚠ 通过 jax.config 设置缓存目录失败: {e}")


def train_step(
    state: EasyDeLState,
    batch: dict,
    learning_rate_fn,
    max_samples: int,
    num_negatives: int,
    temperature: float,
    partition_spec: "PartitionSpec",
    gradient_accumulation_steps: int,
    default_learning_rate: float,
) -> tuple[EasyDeLState, LossMetrics]:
    """
    Training step for contrastive learning.

    Args:
      state: Current model state
      batch: Batch containing input_ids, attention_mask, segment_ids, triplet_type
      learning_rate_fn: Learning rate function
      max_samples: Maximum number of samples per packed sequence
      num_negatives: Number of negative samples per anchor-positive pair
      temperature: Temperature parameter for InfoNCE loss
      partition_spec: Partition specification for distributed training
      gradient_accumulation_steps: Number of gradient accumulation steps
      default_learning_rate: Default learning rate if learning_rate_fn is None

    Returns:
      Updated state and metrics
    """
    # Determine batch size, minibatch size, and enforce partition spec
    _batch_size, minibatch_size, partition_spec = make_assertions_and_get_sizes(
        batch=batch,
        gradient_accumulation_steps=gradient_accumulation_steps,
        batch_partition_spec=partition_spec,
    )
    batch = with_sharding_constraint(arr=batch, sharding=partition_spec)

    def loss_fn(tree, minibatch):
        """
        Computes the loss and additional metrics for a given minibatch and tree state.

        Args:
            tree: The current update to the model's graph state.
            minibatch: A minibatch of input data.

        Returns:
            A tuple containing:
                - The computed loss (scalar).
                - Additional metrics (LossMetrics) produced during loss computation.
        """
        # Merge the state with the provided tree update
        module = flax.nnx.merge(state.graphdef, tree, state.graphother)

        # Prepare loss_kwargs for contrastive learning parameters
        loss_kwargs = {
            "temperature": temperature,
            "max_samples": max_samples,
            "num_negatives": num_negatives,
        }

        # Use module's compute_loss method
        _outputs, metrics = module.compute_loss(
            labels=None,  # Not used for contrastive learning
            loss_config=None,  # Not used for contrastive learning
            loss_kwargs=loss_kwargs,
            **minibatch,
        )

        return metrics.loss, metrics

    # Compute gradients and metrics across minibatches
    gradients, metrics = minibatch_call(
        state=state,
        batch=batch,
        minibatch_size=minibatch_size,
        grad_fn=jax.value_and_grad(loss_fn, has_aux=True),
    )

    # Update state using the computed gradients and updated metrics
    state = update_state_respectfully(
        state=state,
        gradients=gradients,
        loss_config=None,  # No loss_config for contrastive learning
        metrics=update_metrics(
            metrics=metrics,
            learning_rate_fn=learning_rate_fn
            if learning_rate_fn
            else lambda step: default_learning_rate,
            step=state.step,
            gradients=gradients,
        ),
    )

    return state, metrics


def eval_step(
    state: EasyDeLState,
    batch: dict,
    max_samples: int,
    num_negatives: int,
    temperature: float,
    partition_spec: "PartitionSpec",
) -> LossMetrics:
    """
    Evaluation step for contrastive learning.

    Args:
      state: Current model state
      batch: Batch containing input_ids, attention_mask, segment_ids, triplet_type
      max_samples: Maximum number of samples per packed sequence
      num_negatives: Number of negative samples per anchor-positive pair
      temperature: Temperature parameter for InfoNCE loss
      partition_spec: Partition specification for distributed training

    Returns:
      Evaluation metrics
    """
    # Enforce partitioning constraints and determine required sharding
    *_, partition_spec = make_assertions_and_get_sizes(
        batch=batch,
        gradient_accumulation_steps=1,
        batch_partition_spec=partition_spec,
    )
    batch = with_sharding_constraint(arr=batch, sharding=partition_spec)

    def loss_fn(tree):
        """
        Computes loss metrics for the evaluation batch given a merged graph state.

        This inner function merges the provided tree with the current state,
        sets the module to evaluation mode, and computes the loss metrics.

        Args:
            tree: The current update of the model's graph state.

        Returns:
            LossMetrics: The computed metrics from the loss function.
        """
        module = state.merge(tree)
        module.eval()

        # Prepare loss_kwargs for contrastive learning parameters
        loss_kwargs = {
            "temperature": temperature,
            "max_samples": max_samples,
            "num_negatives": num_negatives,
        }

        # Use module's compute_loss method
        _outputs, metrics = module.compute_loss(
            labels=None,  # Not used for contrastive learning
            loss_config=None,  # Not used for contrastive learning
            loss_kwargs=loss_kwargs,
            **batch,
        )

        return metrics

    metrics = loss_fn(state.graphstate)
    return metrics


class ContrastiveTrainer(Trainer):
    """
    Contrastive Learning Trainer for embedding models.

    This trainer is designed for contrastive learning tasks using triplet data
    (anchor, positive, negative). It integrates with EasyDeL's data pipeline
    and supports packed sequences with triplet_type markers.

    Key Features:
    - Supports triplet-based contrastive learning with InfoNCE loss
    - Integrates with dataset_loader.py for data processing
    - Handles packed sequences with segment_ids and triplet_type
    - Automatic embedding extraction and loss computation
    - Full support for distributed training with JAX

    Data Format:
    The trainer expects batches containing:
    - input_ids: Token IDs [batch_size, seq_length]
    - attention_mask: Attention mask [batch_size, seq_length]
    - segment_ids: Segment IDs for sample boundaries [batch_size, seq_length]
    - triplet_type: Type markers (0=anchor, 1=positive, 2+=negative) [batch_size, seq_length]

    Model Requirements:
    The model should support the following forward signature:
    ```python
    model(
      input_ids,
      attention_mask,
      segment_ids,
      position_ids,
      triplet_type,
      max_samples=64,
      num_negatives=1,
      output_hidden_states=False
    ) -> (projected_embs, slot_mask)
    ```
    Where:
    - projected_embs: [batch_size, max_samples, num_slots, embedding_dim]
    - slot_mask: [batch_size, max_samples, num_slots]

    Commonly Used TrainingArguments:

    ==================== 训练基础参数 ====================
    learning_rate (float, default=5e-5):
      学习率。通常范围: 1e-5 到 1e-3。对于对比学习，建议从 1e-4 开始。

    num_train_epochs (int, default=10):
      训练轮数。如果设置了 max_training_steps，此参数会被覆盖。

    total_batch_size (int, default=32):
      总批次大小（所有设备的总和）。实际每个设备的批次大小会根据设备数量自动计算。

    gradient_accumulation_steps (int, default=1):
      梯度累积步数。用于在内存受限时模拟更大的批次大小。
      例如: total_batch_size=128, gradient_accumulation_steps=4 等价于每步32的批次大小。

    max_training_steps (int | None, default=None):
      最大训练步数。如果设置，会覆盖 num_train_epochs。

    ==================== 优化器参数 ====================
    optimizer (str, default="adamw"):
      优化器类型。可选值: "adamw", "adam", "sgd", "lion" 等。
      推荐使用 "adamw" 用于对比学习。

    scheduler (str, default="none"):
      学习率调度器。可选值: "linear", "cosine", "constant", "none"。
      推荐使用 "cosine" 或 "linear" 进行学习率衰减。

    warmup_steps (int, default=0):
      预热步数。在训练开始时线性增加学习率，有助于训练稳定性。

    weight_decay (float, default=0.01):
      权重衰减（L2正则化）。常用值: 0.01, 0.1。

    clip_grad (float | None, default=None):
      梯度裁剪阈值。如果梯度范数超过此值，会被裁剪。
      推荐值: 1.0 或 0.5，有助于训练稳定性。

    ==================== 数据相关参数 ====================
    max_sequence_length (int, default=4096):
      最大序列长度。应该与 dataset_loader 中的 seq_length 保持一致。
      对于长序列对比学习，可以设置为 32768 或更大。

    dataloader_num_workers (int, default=0):
      数据加载器工作进程数。设置为 0 表示在主进程中加载数据。
      对于大数据集，可以设置为 2-4 以加速数据加载。

    shuffle_train_dataset (bool, default=True):
      是否打乱训练数据集。推荐保持为 True。

    ==================== 检查点和日志 ====================
    save_steps (int | None, default=None):
      保存检查点的步数间隔。例如: 1000 表示每1000步保存一次。
      如果为 None，则只在训练结束时保存。

    save_total_limit (int | None, default=None):
      保留的检查点数量上限。超过此数量的旧检查点会被自动删除。
      推荐值: 3-5，以节省存储空间。

    save_directory (str, default="EasyDeL-Checkpoints"):
      检查点保存目录。

    log_steps (int, default=10):
      记录日志的步数间隔。例如: 10 表示每10步记录一次指标。

    report_steps (int, default=5):
      报告指标的步数间隔。通常小于或等于 log_steps。

    use_wandb (bool, default=True):
      是否使用 Weights & Biases 进行实验跟踪。
      需要先安装 wandb: pip install wandb

    wandb_name (str | None, default=None):
      WandB 运行名称。如果为 None，会自动生成。

    report_metrics (bool, default=True):
      是否报告训练指标。如果为 False，将不会记录任何指标。

    ==================== 性能相关参数 ====================

    performance_mode (bool, default=False):
      性能模式。如果为 True，会禁用一些监控功能以提升训练速度。
      不推荐在开发阶段使用。

    ==================== 分布式训练参数 ====================
    jax_distributed_config (dict | None, default=None):
      JAX 分布式训练配置。用于多设备/多主机训练。

    step_partition_spec (PartitionSpec, default=PartitionSpec(("dp", "fsdp"), "sp")):
      训练步骤的分片规范。用于模型并行和数据并行。

    ==================== 其他常用参数 ====================
    do_eval (bool, default=False):
      是否在训练过程中进行评估。

    evaluation_steps (int | None, default=None):
      评估的步数间隔。例如: 500 表示每500步评估一次。

    resume_if_possible (bool, default=True):
      是否自动从最新检查点恢复训练。

    verbose (bool, default=True):
      是否打印详细输出。

    Example:
      >>> from contrastive_trainer import ContrastiveTrainer
      >>> from easydel.trainers import TrainingArguments
      >>> from dataset_loader import create_dataset_pipeline
      >>> from transformers import AutoTokenizer
      >>>
      >>> # Create training arguments
      >>> args = TrainingArguments(
      ...     learning_rate=1e-4,
      ...     num_train_epochs=3,
      ...     total_batch_size=32,
      ...     max_sequence_length=32768,
      ...     save_steps=1000,
      ...     log_steps=10,
      ...     use_wandb=True,
      ...     optimizer="adamw",
      ...     scheduler="cosine",
      ...     warmup_steps=1000,
      ...     weight_decay=0.01,
      ...     clip_grad=1.0,
      ... )
      >>>
      >>> # Create data pipeline
      >>> tokenizer = AutoTokenizer.from_pretrained("your-model")
      >>> train_data_source = create_dataset_source(
      ...     data_files="data/*.jsonl",
      ...     tokenizer=tokenizer,
      ...     seq_length=32768,
      ...     max_samples=64,
      ...     max_sample_length=512,  # 可选：限制每个样本的最大长度
      ... )
      >>>
      >>> # Create trainer
      >>> trainer = ContrastiveTrainer(
      ...     arguments=args,
      ...     model=model,
      ...     dataset_train=train_data_source,
      ...     processing_class=tokenizer,
      ...     max_samples=64,
      ...     num_negatives=1,
      ...     temperature=0.07,
      ... )
      >>>
      >>> # Train
      >>> trainer.train()
    """

    def __init__(
        self,
        arguments: TrainingArguments | None = None,
        model_state: EasyDeLState | None = None,
        model: tp.type[EasyDeLBaseModule] | None = None,
        dataset_train: Dataset | IterableDataset | ShardedDataSource | None = None,
        dataset_eval: Dataset | IterableDataset | ShardedDataSource | None = None,
        data_collator: tp.Callable | None = None,
        finetune: bool = True,
        processing_class: None = None,
        max_samples: int = 64,
        num_negatives: int = 1,
        temperature: float = 0.07,
        **kwargs,
    ):
        """
        Initialize ContrastiveTrainer.

        Args:
          arguments: TrainingArguments configuration object
          model: Model to train (EasyDeLBaseModule or EasyDeLState)
          model_state: Model state to train (EasyDeLState)
          train_dataset: Training dataset (can be ShardedDataSource from pipeline)
          eval_dataset: Evaluation dataset (optional)
          processing_class: Tokenizer or processor (optional, for compatibility)
          max_samples: Maximum number of samples per packed sequence.
                      Must match the max_samples used in create_dataset_pipeline.
          num_negatives: Number of negative samples per anchor-positive pair.
          temperature: Temperature parameter for InfoNCE loss.
          **kwargs: Additional arguments passed to BaseTrainer
        """
        # Store contrastive learning specific parameters
        self.max_samples = max_samples
        self.num_negatives = num_negatives
        self.temperature = temperature

        # Store original train dataset for resume state handling
        self._original_train_dataset = dataset_train

        # Initialize base trainer
        super().__init__(
            arguments=arguments,
            model_state=model_state,
            model=model,
            dataset_train=dataset_train,
            dataset_eval=dataset_eval,
            data_collator=data_collator,
            finetune=finetune,
            processing_class=processing_class,
            **kwargs,
        )

    def configure_functions(self) -> TrainerConfigureFunctionOutput:
        """
        Configure training and evaluation step functions for contrastive learning.

        Returns:
          TrainerConfigureFunctionOutput with compiled step functions
        """
        # Get mesh for distributed training
        mesh = self.model.mesh

        # Create empty sharding for batch data
        empty_sharding = jax.sharding.NamedSharding(spec=PartitionSpec(), mesh=mesh)

        # Set static arguments for training step
        self._train_shared_fn_static_args = (
            self.scheduler,
            self.max_samples,
            self.num_negatives,
            self.temperature,
            self.arguments.step_partition_spec,
            self.arguments.gradient_accumulation_steps,
            self.arguments.learning_rate,
        )

        # Compile training step function
        sharded_train_step_fn = ejit(
            train_step,
            static_argnums=(2, 3, 4, 5, 6, 7, 8),
            in_shardings=(self.state_shardings, empty_sharding),
            out_shardings=(self.state_shardings, empty_sharding),
            donate_argnums=(0,),
        )

        # Set static arguments for evaluation step
        self._eval_shared_fn_static_args = (
            self.max_samples,
            self.num_negatives,
            self.temperature,
            self.arguments.step_partition_spec,
        )

        # Compile evaluation step function
        sharded_eval_step_fn = ejit(
            eval_step,
            static_argnums=(2, 3, 4, 5),
            in_shardings=(self.state_shardings, empty_sharding),
            out_shardings=(empty_sharding),
        )

        # Get checkpoint manager
        checkpoint_manager = self.arguments.get_streaming_checkpointer()

        return TrainerConfigureFunctionOutput(
            sharded_training_step_function=sharded_train_step_fn,
            mesh=mesh,
            checkpoint_manager=checkpoint_manager,
            sharded_evaluation_step_function=sharded_eval_step_fn,
        )

    def create_collect_function(
        self,
        max_sequence_length: int,
        truncation_mode: tp.Literal["keep_end", "keep_start"],
    ) -> tp.Callable:
        """
        Creates a function to collect and process batches of data for training or evaluation.

        This function is designed to work with ShardedDataSource from easydata pipeline.
        The data from ShardedDataSource is already packed and contains:
        - input_ids: Token IDs
        - attention_mask: Attention mask
        - segment_ids: Segment IDs for sample boundaries
        - triplet_type: Type markers (0=anchor, 1=positive, 2+=negative)
        - position_ids: Position IDs (optional)

        Args:
            max_sequence_length: The maximum allowed sequence length.
            truncation_mode: The truncation mode ("keep_end" or "keep_start").

        Returns:
            A function that takes a batch (list of dicts) and returns a processed batch dict.

        Note:
            Currently only supports ShardedDataSource from easydata pipeline.
        """

        def collect_fn(batch: list[dict]) -> dict:
            """
            Collect and process a batch of packed sequences from ShardedDataSource.

            Args:
                batch: List of dictionaries, each containing packed sequence data from easydata.

            Returns:
                Processed batch dictionary with batched numpy arrays.
            """
            # If batch is empty, return empty dict
            if not batch:
                return {}

            # Extract all keys from first example
            keys = batch[0].keys()

            # Process each field
            processed = {}
            for key in keys:
                values = [item[key] for item in batch if key in item]

                if not values:
                    continue

                # Convert to numpy arrays
                arrays = [np.asarray(v) for v in values]

                # Handle sequence fields
                if key in [
                    "input_ids",
                    "attention_mask",
                    "segment_ids",
                    "triplet_type",
                    "position_ids",
                ]:
                    # For sequence fields, pad or truncate to max_sequence_length
                    max_len = max(
                        arr.shape[-1] if arr.ndim > 0 else 0 for arr in arrays
                    )
                    max_len = min(max_len, max_sequence_length)

                    # Pad or truncate each array
                    processed_arrays = []
                    for arr in arrays:
                        arr_len = arr.shape[-1] if arr.ndim > 0 else 0

                        if arr_len > max_sequence_length:
                            # Truncate
                            if truncation_mode == "keep_end":
                                arr = arr[..., -max_sequence_length:]
                            else:  # keep_start
                                arr = arr[..., :max_sequence_length]
                            arr_len = max_sequence_length

                        # Pad if necessary
                        if arr_len < max_sequence_length:
                            pad_width = max_sequence_length - arr_len
                            if arr.ndim == 1:
                                arr = np.pad(arr, (0, pad_width), constant_values=0)
                            else:
                                # For multi-dimensional arrays, pad the last dimension
                                pad_shape = [(0, 0)] * (arr.ndim - 1) + [(0, pad_width)]
                                arr = np.pad(arr, pad_shape, constant_values=0)

                        processed_arrays.append(arr)

                    # Stack into batch
                    processed[key] = np.stack(processed_arrays)
                else:
                    # For other fields, just stack
                    try:
                        processed[key] = np.stack(arrays)
                    except ValueError:
                        # If shapes don't match, keep as list
                        processed[key] = values

            return processed

        return collect_fn

    def _load_resume_state_from_checkpoint(
        self, checkpoint_dir: str
    ) -> ResumeState | None:
        """Load ResumeState from checkpoint directory.

        Only handles dataset ResumeState, not model state (handled by base Trainer).

        Args:
            checkpoint_dir: Checkpoint directory path

        Returns:
            ResumeState if found, None otherwise
        """
        if not checkpoint_dir or not os.path.exists(checkpoint_dir):
            return None

        # Try to find the latest checkpoint
        model_name = self.arguments.model_name
        model_checkpoint_dir = os.path.join(checkpoint_dir, model_name)
        if not os.path.exists(model_checkpoint_dir):
            return None

        # Find all checkpoint directories
        checkpoint_paths = []
        for entry in os.listdir(model_checkpoint_dir):
            path = os.path.join(model_checkpoint_dir, entry)
            if os.path.isdir(path):
                checkpoint_paths.append(path)

        if not checkpoint_paths:
            return None

        # Sort by step number (extract from directory name)
        import re

        def extract_step(path: str) -> int:
            match = re.search(r"\d+", os.path.basename(path))
            return int(match.group()) if match else -1

        checkpoint_paths.sort(key=extract_step, reverse=True)
        latest_checkpoint = checkpoint_paths[0]

        # Try to load ResumeState from checkpoint JSON file
        resume_state_file = os.path.join(latest_checkpoint, "resume_state.json")
        if os.path.exists(resume_state_file):
            try:
                with open(resume_state_file, "r", encoding="utf-8") as f:
                    state_dict = json.load(f)
                    return ResumeState.from_dict(state_dict)
            except Exception as e:
                logger.warning(
                    f"Failed to load ResumeState from {resume_state_file}: {e}"
                )
                return None

        return None

    def _save_resume_state_to_checkpoint(
        self, checkpoint_path: str, resume_state: ResumeState
    ) -> None:
        """Save ResumeState to checkpoint directory.

        Args:
            checkpoint_path: Checkpoint directory path
            resume_state: ResumeState to save
        """
        if not checkpoint_path:
            return

        os.makedirs(checkpoint_path, exist_ok=True)
        resume_state_file = os.path.join(checkpoint_path, "resume_state.json")
        try:
            with open(resume_state_file, "w", encoding="utf-8") as f:
                json.dump(resume_state.to_dict(), f, indent=2)
            logger.info(f"Saved ResumeState to {resume_state_file}")
        except Exception as e:
            logger.warning(f"Failed to save ResumeState to {resume_state_file}: {e}")

    def _get_current_resume_state(
        self, current_step: int, current_epoch: int = 0
    ) -> ResumeState | None:
        """Get current ResumeState from dataset if available.

        This method recursively searches for ResumeState in the dataset wrapper chain
        and updates it with the current training step and epoch.

        Args:
            current_step: Current training step
            current_epoch: Current training epoch

        Returns:
            ResumeState if available, None otherwise
        """
        dataset = self.dataset_train
        if dataset is None:
            return None

        # First, try to use get_current_resume_state() method if available
        # This is the preferred method for ShardedDataSource from create_dataset_source
        if hasattr(dataset, "get_current_resume_state"):
            try:
                resume_state = dataset.get_current_resume_state()
                if resume_state is not None:
                    # Update step and epoch, keep shard_index and row_index
                    return ResumeState(
                        shard_index=resume_state.shard_index,
                        row_index=resume_state.row_index,
                        step=current_step,
                        epoch=current_epoch,
                        dataset_states=resume_state.dataset_states,
                    )
            except Exception as e:
                logger.warning(f"Failed to call get_current_resume_state(): {e}")

        # Fallback: recursively search for ResumeState in the wrapper chain
        resume_state = self._recursive_find_resume_state(dataset)
        if resume_state is None:
            return None

        # Update step and epoch, keep shard_index and row_index
        # Note: shard_index and row_index represent the current reading position
        # when saving checkpoint, which should be used as the starting position
        # when resuming training
        return ResumeState(
            shard_index=resume_state.shard_index,
            row_index=resume_state.row_index,
            step=current_step,
            epoch=current_epoch,
            dataset_states=resume_state.dataset_states,
        )

    def _recursive_find_resume_state(self, dataset) -> ResumeState | None:
        """Recursively search for ResumeState in dataset wrapper chain.

        This method handles multiple layers of wrapping:
        1. get_current_state() method (for _ResumableSourceWrapper)
        2. get_current_resume_state() method (for ShardedDataSource from pipeline)
        3. Direct _resume_state attribute
        4. _initial_resume_state attribute (for _ResumableSourceWrapper)
        5. _source attribute (for wrapper classes)
        6. source attribute (for TransformedShardedSource)
        7. dataset attribute (for other wrapper types)

        Args:
            dataset: Dataset object to search

        Returns:
            ResumeState if found, None otherwise
        """
        # Check for get_current_state() method (for _ResumableSourceWrapper)
        if hasattr(dataset, "get_current_state"):
            try:
                return dataset.get_current_state()
            except Exception:
                pass

        # Check for get_current_resume_state() method
        if hasattr(dataset, "get_current_resume_state"):
            try:
                return dataset.get_current_resume_state()
            except Exception:
                pass

        # Check for _initial_resume_state attribute (for _ResumableSourceWrapper)
        if hasattr(dataset, "_initial_resume_state"):
            return dataset._initial_resume_state

        # Base case: check direct _resume_state attribute
        if hasattr(dataset, "_resume_state"):
            return dataset._resume_state

        # Check if dataset is a dict with resume_state key
        if isinstance(dataset, dict) and "resume_state" in dataset:
            return dataset["resume_state"]

        # Recursively search inner datasets
        return self._recursive_search_inner(dataset, self._recursive_find_resume_state)

    def _recursive_apply_resume_state(self, dataset, resume_state, wrapper_class):
        """Recursively apply ResumeState to the deepest wrapper.

        Args:
            dataset: Current dataset object
            resume_state: ResumeState to apply
            wrapper_class: Class to use for wrapping if needed

        Returns:
            Dataset with ResumeState applied
        """
        # Check if this is already a _ResumableSourceWrapper
        if isinstance(dataset, wrapper_class):
            # Update existing wrapper's resume state
            # _ResumableSourceWrapper uses _initial_resume_state and tracks current position
            dataset._initial_resume_state = resume_state
            dataset._current_shard_index = resume_state.shard_index
            dataset._current_row_index = resume_state.row_index
            return dataset

        # Check if dataset has _initial_resume_state (might be _ResumableSourceWrapper)
        if hasattr(dataset, "_initial_resume_state"):
            dataset._initial_resume_state = resume_state
            if hasattr(dataset, "_current_shard_index"):
                dataset._current_shard_index = resume_state.shard_index
            if hasattr(dataset, "_current_row_index"):
                dataset._current_row_index = resume_state.row_index
            return dataset

        # Check for _resume_state attribute (for other wrapper types)
        if hasattr(dataset, "_resume_state"):
            dataset._resume_state = resume_state
            return dataset

        # Try to apply to inner datasets
        result = self._recursive_search_inner(
            dataset,
            lambda inner: self._recursive_apply_resume_state(
                inner, resume_state, wrapper_class
            ),
            apply_to_inner=True,
        )

        if result is not None:
            return result

        # If no inner dataset found, wrap this dataset
        return wrapper_class(dataset, resume_state)

    def _recursive_search_inner(self, dataset, search_func, apply_to_inner=False):
        """Common recursive search logic for inner datasets.

        Args:
            dataset: Dataset object to search
            search_func: Function to apply to inner datasets
            apply_to_inner: If True, update the inner dataset attribute

        Returns:
            Result from search_func or None
        """
        wrapper_attrs = ["_source", "source", "dataset"]

        for attr_name in wrapper_attrs:
            if hasattr(dataset, attr_name):
                inner_dataset = getattr(dataset, attr_name)
                if inner_dataset is not None and inner_dataset is not dataset:
                    result = search_func(inner_dataset)
                    if result is not None:
                        if apply_to_inner:
                            # Update the attribute with result
                            setattr(dataset, attr_name, result)
                            return dataset
                        return result

        return None

    def configure_dataloaders(self) -> TrainerConfigureDataloaderOutput:
        """Configure dataloaders with ResumeState support.

        This method:
        1. Loads ResumeState from checkpoint if resume_if_possible=True
        2. Applies ResumeState to train dataset if available
        3. Calls base class configure_dataloaders to set up dataloaders

        Returns:
            TrainerConfigureDataloaderOutput: An object containing the configured dataloaders and the
                                            maximum number of training and evaluation steps.

        Note: Only handles dataset ResumeState, model state is handled by base Trainer.
        """
        # Step 1: Restore ResumeState from saved checkpoint data
        if (
            self.arguments.resume_if_possible
            and self._original_train_dataset is not None
            and hasattr(self._original_train_dataset, "iter_shards")
        ):
            # Try to load ResumeState from checkpoint
            checkpoint_dir = self.arguments.save_directory
            resume_state = self._load_resume_state_from_checkpoint(checkpoint_dir)

            if resume_state is not None:
                logger.info(
                    f"Loaded ResumeState from checkpoint: "
                    f"shard_index={resume_state.shard_index}, "
                    f"row_index={resume_state.row_index}, "
                    f"step={resume_state.step}"
                )
                # Apply ResumeState to dataset
                self.dataset_train = self._recursive_apply_resume_state(
                    self._original_train_dataset, resume_state, _ResumableSourceWrapper
                )
        else:
            # If resume_if_possible=False, ensure dataset starts from the beginning
            # Reset any existing ResumeState in the dataset to start from (0, 0)
            if self._original_train_dataset is not None and hasattr(
                self._original_train_dataset, "iter_shards"
            ):
                # Check if dataset already has a ResumeState that might cause issues
                existing_resume_state = self._recursive_find_resume_state(
                    self._original_train_dataset
                )
                if existing_resume_state is not None:
                    # Reset to start from beginning
                    from easydel.data import ResumeState

                    reset_resume_state = ResumeState(
                        shard_index=0,
                        row_index=0,
                        step=0,
                        epoch=0,
                        dataset_states=existing_resume_state.dataset_states
                        if existing_resume_state.dataset_states
                        else {},
                    )
                    logger.info(
                        "Resetting dataset ResumeState to start from beginning: "
                        "shard_index=0, row_index=0 (resume_if_possible=False)"
                    )
                    self.dataset_train = self._recursive_apply_resume_state(
                        self._original_train_dataset,
                        reset_resume_state,
                        _ResumableSourceWrapper,
                    )

        # Step 2: Call base class configure_dataloaders
        return super().configure_dataloaders()

    def _save_state(
        self,
        state: EasyDeLState,
        save_directory: str | None = None,
        *args,
        **kwargs,
    ) -> str:
        """Save the current model state to a checkpoint with ResumeState support.

        This method extends the base Trainer._save_state() to also save
        the current ResumeState to the checkpoint directory. The ResumeState
        is saved as a JSON file (resume_state.json) in the checkpoint directory.

        Args:
            state: The model state to save
            save_directory: Optional override for save directory. If None, uses
                default directory based on current step.
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            str: Path to the saved checkpoint directory
        """
        # Call parent method to save model state
        checkpoint_path = super()._save_state(
            state=state, save_directory=save_directory, *args, **kwargs
        )

        # Save ResumeState after model state is saved
        current_step = int(jax.device_get(state.step))
        resume_state = self._get_current_resume_state(
            current_step=current_step, current_epoch=0
        )

        if resume_state is not None:
            self._save_resume_state_to_checkpoint(checkpoint_path, resume_state)
            logger.info(
                f"Saved ResumeState to checkpoint at step {current_step}: "
                f"shard_index={resume_state.shard_index}, "
                f"row_index={resume_state.row_index}"
            )

        return checkpoint_path
