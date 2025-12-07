"""
Model wrapper with projection head for Qwen3-Embedding

This module creates a combined EasyDeL model that includes both the base model
and projection head. The new model shares the same mesh as the base model.
"""

import os
import copy

from typing import Optional, Any, Mapping, Sequence, Callable

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec

from flax import nnx
import chex
import optax

from ejkernel.types import MaskInfo

import easydel as ed
from easydel.infra import EasyDeLBaseModule
from easydel.infra.factory import TaskType, registry
from easydel.infra.loss_utils import LossMetrics, LossConfig

from qwen3_embedding_config import Qwen3EmbeddingConfig
from loss import info_nce_loss

CUSTOM_MODEL_TYPE = "qwen3_embedding_with_projection"
BASE_MODEL_TYPE = "qwen3"
CUSTOM_ARCHITECTURE_NAME = "Qwen3EmbeddingModel"


class ProjectionHead(EasyDeLBaseModule):
    """
    GELU projection head to reduce embedding dimension to 256
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 256,
        dtype: jnp.dtype = jnp.bfloat16,
        *,
        rngs: nnx.Rngs,
    ):
        """
        Initialize projection head

        Args:
          input_dim: Input embedding dimension
          output_dim: Output embedding dimension (default 256)
          dtype: Data type
          rngs: Random number generators
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dtype = dtype

        # Linear layer: input_dim -> output_dim
        self.linear = nnx.Linear(input_dim, output_dim, dtype=dtype, rngs=rngs)

    def __call__(self, x: chex.Array) -> chex.Array:
        """
        Forward pass

        Args:
          x: Input embeddings [..., input_dim]

        Returns:
          Projected embeddings [..., output_dim]
        """
        x = self.linear(x.astype(self.dtype))
        # Apply GELU activation
        x = nnx.gelu(x)
        return x


class Qwen3EmbeddingModel(EasyDeLBaseModule):
    """
    Independent EasyDeL model that includes base model layers and projection head

    This model inherits from EasyDeLBaseModule and is a completely independent model.
    It builds the base model layers and projection head internally from config,
    without any dependency on external base_model or projection modules.

    To initialize state from an existing base_model, use the external function
    `initialize_from_base_model_state()` after creating the model.
    """

    def __init__(
        self,
        config: Any,
        embedding_dim: int = 256,
        dtype: Optional[jnp.dtype] = None,
        param_dtype: Optional[jnp.dtype] = None,
        precision: Optional[jax.lax.Precision] = None,
        rngs: Optional[nnx.Rngs] = None,
        quantization_config: Optional[Any] = None,
    ):
        """
        Initialize independent model from config

        Args:
          config: Model configuration (must be provided)
          embedding_dim: Output embedding dimension (default 256)
          dtype: Computation dtype (if None, get from config)
          param_dtype: Parameter dtype (if None, get from config)
          precision: Precision (if None, get from config or use DEFAULT)
          rngs: Random number generators (if None, create new one)
          quantization_config: Optional EasyDeLQuantizationConfig (if None, uses default)
        """
        if config is None:
            raise ValueError("Qwen3EmbeddingModel requires config to build model")

        if rngs is None:
            key = jax.random.PRNGKey(42)
            rngs = nnx.Rngs(key)

        # Set default dtype, param_dtype, precision from config if not provided
        if dtype is None:
            dtype = getattr(config, "dtype", jnp.bfloat16)
        if param_dtype is None:
            param_dtype = getattr(config, "param_dtype", jnp.bfloat16)
        if precision is None:
            precision = getattr(config, "precision", jax.lax.Precision.DEFAULT)

        # Initialize EasyDeLBaseModule with required parameters first
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        # Build base model internally from config using AutoEasyDeLModel
        base_model_type = getattr(config, "combined_base_model_type", BASE_MODEL_TYPE)
        base_architectures = getattr(
            config,
            "combined_base_architectures",
            getattr(config, "architectures", None),
        )
        base_config = copy.deepcopy(config)
        base_config.model_type = base_model_type
        if base_architectures is not None:
            base_config.architectures = list(base_architectures)

        # Create base model internally from config using AutoEasyDeLModel.from_config
        self.base_model = ed.AutoEasyDeLModel.from_config(
            config=base_config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        # Create projection head internally from config parameters
        hidden_size = getattr(config, "hidden_size", 2560)
        embedding_dim = getattr(config, "embedding_dim", embedding_dim)
        self.projection = ProjectionHead(
            input_dim=hidden_size, output_dim=embedding_dim, dtype=dtype, rngs=rngs
        )
        self.embedding_dim = embedding_dim

    def set_model_mesh(self, new_mesh: Optional[jax.sharding.Mesh]) -> None:
        """
        Set model mesh using config.set_model_mesh

        This method updates the mesh configuration through EasyDeLBaseModule's config,
        and also updates the model's mesh attribute for compatibility.

        Args:
          new_mesh: New JAX mesh to set, or None to clear mesh
        """
        # Use config.set_model_mesh to set mesh through EasyDeLBaseModule's config
        try:
            self.config.set_model_mesh(new_mesh)
            self.base_model.config.set_model_mesh(new_mesh)
            self.projection.config.set_model_mesh(new_mesh)
        except Exception as e:
            print(f"Error setting mesh for projection: {e}")
            pass

    def __call__(
        self,
        input_ids: chex.Array,
        attention_mask: Optional[chex.Array] = None,
        segment_ids: Optional[chex.Array] = None,
        position_ids: Optional[chex.Array] = None,
        triplet_type: Optional[chex.Array] = None,
        max_samples: int = 64,
        num_negatives: int = 1,
        output_hidden_states: bool = False,
        **kwargs,
    ) -> Any:
        """
        Forward pass through base model and projection

        Args:
          input_ids: Token IDs [batch_size, seq_len] or [seq_len]
          attention_mask: Attention mask [batch_size, seq_len] or [seq_len]
          segment_ids: Segment IDs array [seq_len] for extracting sample boundaries
          triplet_type: Segment type array [seq_len] where 0=anchor, 1=positive, 2=negative, -1=special_token
          max_samples: Maximum number of samples per packed sequence. Must be a static value for JIT compilation.
                      Should match the max_samples value used in create_dataset_pipeline. Default: 64.
          num_negatives: Number of negative samples per sample. Default: 1.
          output_hidden_states: Whether to return hidden states
          **kwargs: Additional arguments for base model

        Returns:
          If triplet_type is provided:
            Tuple of (projected_embs, slot_mask) where:
              - projected_embs: [batch_size, max_samples, num_slots, embedding_dim]
              - slot_mask: [batch_size, max_samples, num_slots]
          Otherwise:
            Projected embeddings [batch_size, seq_len, embedding_dim] or ModelOutput with hidden_states
        """

        # 在方法开头进行防御性检查
        # 确保 input_ids 只能是 1D 或 2D，防止传入了 3D 数据导致后续逻辑错误
        chex.assert_rank(input_ids, {1, 2})
        if input_ids.ndim == 1:
            if segment_ids is not None:
                chex.assert_rank(segment_ids, {1})
            if triplet_type is not None:
                chex.assert_rank(triplet_type, {1})
            if attention_mask is not None:
                chex.assert_rank(attention_mask, {1})
        elif input_ids.ndim == 2:
            if segment_ids is not None:
                chex.assert_rank(segment_ids, {2})
            if triplet_type is not None:
                chex.assert_rank(triplet_type, {2})
            if attention_mask is not None:
                chex.assert_rank(attention_mask, {2})
        else:
            raise ValueError(f"Unexpected input_ids dimension: {input_ids.ndim}")

        if attention_mask is not None:
            assert segment_ids is not None
            assert triplet_type is not None
            assert position_ids is not None
            assert position_ids.shape == segment_ids.shape
            assert position_ids.shape == attention_mask.shape
            assert position_ids.shape == triplet_type.shape

        # All computations within the same mesh for distributed training
        # EasyDeL models are designed for SPMD (jit + Mesh), not pmap
        mask_info = MaskInfo.from_segments(
            q_segment_ids=segment_ids,
            kv_segment_ids=segment_ids,
            q_positions=position_ids,
            kv_positions=position_ids,
        )

        # Forward through base model
        base_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=None,
            mask_info=mask_info,
            output_hidden_states=True,  # Always get hidden states for extraction
            **kwargs,
        )

        # Extract hidden states
        if hasattr(base_outputs, "last_hidden_state"):
            hidden_states = base_outputs.last_hidden_state
        elif isinstance(base_outputs, tuple):
            hidden_states = base_outputs[0]
        else:
            hidden_states = base_outputs

        if triplet_type is not None:
            # Extract and project embeddings for triplet-based training
            # max_samples is a static parameter for JIT compilation
            # Should match the max_samples value used in create_dataset_pipeline

            # 1. 提取 (得到大宽表)
            # combined: [B, S, 2+N, H]
            # mask:     [B, S, 2+N]
            combined_embs, slot_mask = self._extract_last_token_embeddings(
                hidden_states=hidden_states,
                segment_ids=segment_ids,
                triplet_type=triplet_type,
                attention_mask=attention_mask,
                max_samples=max_samples,
                num_negatives=num_negatives,
            )

            # 2. 投影 (一次性投影所有向量，效率最高)
            projected_embs = self.projection(combined_embs)

            # Return projected embeddings and slot mask for external loss computation
            if output_hidden_states:
                return (projected_embs, slot_mask, hidden_states)
            else:
                return (projected_embs, slot_mask)
        else:
            # Standard forward pass without triplet extraction
            projected_embeddings = self.projection(hidden_states)

            if output_hidden_states:

                class ModelOutput:
                    def __init__(self, last_hidden_state, projected_embeddings):
                        self.last_hidden_state = last_hidden_state
                        self.projected_embeddings = projected_embeddings

                return ModelOutput(
                    last_hidden_state=hidden_states,
                    projected_embeddings=projected_embeddings,
                )
            else:
                return projected_embeddings

    def compute_loss(
        self,
        *,
        labels: chex.Array | None = None,
        loss_config: LossConfig | None = None,
        loss_kwargs: dict | None = None,
        **batch,
    ) -> tuple[Any, LossMetrics]:
        """
        Computes the InfoNCE loss for contrastive learning.

        This method performs a forward pass to get projected embeddings,
        then calculates the InfoNCE loss using anchor, positive, and negative embeddings.

        Args:
            labels: Not used for contrastive learning (kept for compatibility with EasyDeLBaseModule interface).
            loss_config: Optional LossConfig (not used for contrastive learning, kept for compatibility).
            loss_kwargs: Optional dictionary containing contrastive learning specific parameters:
                - temperature (float): Temperature parameter for InfoNCE loss. Default: 0.07
                - max_samples (int): Maximum number of samples per packed sequence. Default: 64
                - num_negatives (int): Number of negative samples per anchor-positive pair. Default: 1
            **batch: Keyword arguments representing the input batch:
                - input_ids: Token IDs [batch_size, seq_len]
                - attention_mask: Attention mask [batch_size, seq_len]
                - segment_ids: Segment IDs for sample boundaries [batch_size, seq_len]
                - triplet_type: Type markers (0=anchor, 1=positive, 2+=negative) [batch_size, seq_len]
                - position_ids: Position IDs [batch_size, seq_len] (optional, will be generated if not provided)

        Returns:
            tuple[Any, LossMetrics]: A tuple containing:
                - Model output (projected_embs, slot_mask)
                - LossMetrics object containing the computed loss and metrics

        Raises:
            ValueError: If required batch fields (input_ids, attention_mask, segment_ids, triplet_type) are missing.
        """
        # Extract contrastive learning parameters from loss_kwargs or use defaults
        loss_kwargs = loss_kwargs or {}
        temperature = loss_kwargs.get("temperature", 0.07)
        max_samples = loss_kwargs.get("max_samples", 64)
        num_negatives = loss_kwargs.get("num_negatives", 1)

        # Extract required batch fields
        input_ids = batch.get("input_ids")
        if input_ids is None:
            raise ValueError("input_ids is required for compute_loss")

        attention_mask = batch.get("attention_mask")
        segment_ids = batch.get("segment_ids")
        triplet_type = batch.get("triplet_type")
        position_ids = batch.get("position_ids")

        # Validate required fields for contrastive learning
        if attention_mask is None or segment_ids is None or triplet_type is None:
            raise ValueError(
                "attention_mask, segment_ids, and triplet_type are required for contrastive learning"
            )

        # Generate position_ids if not provided
        if position_ids is None:
            seq_len = input_ids.shape[-1]
            position_ids = jnp.arange(seq_len, dtype=jnp.int32)
            if input_ids.ndim == 2:
                position_ids = jnp.broadcast_to(position_ids[None, :], input_ids.shape)

        # Forward pass through model to get projected embeddings
        model_output = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            position_ids=position_ids,
            triplet_type=triplet_type,
            max_samples=max_samples,
            num_negatives=num_negatives,
            output_hidden_states=False,
        )

        # Model returns (projected_embs, slot_mask)
        if isinstance(model_output, tuple):
            projected_embs, slot_mask = model_output
        else:
            raise ValueError(
                "Model must return (projected_embs, slot_mask) tuple for contrastive learning"
            )

        # Split embeddings and masks
        # Slot structure: [Anchor(0), Positive(1), Negatives(2...)]
        anchor_emb = projected_embs[..., 0, :]  # [B, S, H]
        anchor_mask = slot_mask[..., 0]  # [B, S]
        positive_emb = projected_embs[..., 1, :]  # [B, S, H]
        pos_mask = slot_mask[..., 1]  # [B, S]
        negatives_emb = projected_embs[..., 2:, :]  # [B, S, N, H]
        negs_mask = slot_mask[..., 2:]  # [B, S, N]

        # Compute InfoNCE loss and metrics
        loss, contrastive_metrics = info_nce_loss(
            anchor_emb=anchor_emb,
            positive_emb=positive_emb,
            negatives_emb=negatives_emb,
            anchor_mask=anchor_mask,
            pos_mask=pos_mask,
            negs_mask=negs_mask,
            temperature=temperature,
            return_metrics=True,
        )

        # Convert contrastive metrics to LossMetrics format
        metrics = LossMetrics(
            loss=loss,
            accuracy=contrastive_metrics.get("accuracy"),
            other_metrics={
                "valid_samples": contrastive_metrics.get("valid_samples"),
                "pos_sim_mean": contrastive_metrics.get("pos_sim_mean"),
                "neg_sim_mean": contrastive_metrics.get("neg_sim_mean"),
                "pos_sim_std": contrastive_metrics.get("pos_sim_std"),
                "neg_sim_std": contrastive_metrics.get("neg_sim_std"),
            },
        )

        # Create output object for compatibility
        class ModelOutput:
            def __init__(self, projected_embs, slot_mask, loss):
                self.projected_embs = projected_embs
                self.slot_mask = slot_mask
                self.loss = loss

        outputs = ModelOutput(
            projected_embs=projected_embs,
            slot_mask=slot_mask,
            loss=loss,
        )

        return outputs, metrics

    def _extract_last_token_embeddings(
        self,
        hidden_states: jax.Array,  # [B, Seq, H]
        segment_ids: jax.Array,  # [B, Seq]
        triplet_type: jax.Array,  # [B, Seq] (0=A, 1=P, 2=N1, 3=N2...)
        attention_mask: jax.Array,  # [B, Seq]
        max_samples: int = 64,
        num_negatives: int = 1,  # 负样本数量
    ):
        # 1. 计算总槽位数 (Slots)
        # Anchor(1) + Positive(1) + Negatives(N)
        num_slots = 2 + num_negatives

        # --- Defensive Checks ---
        chex.assert_rank(hidden_states, {2, 3})
        hs_ndim = hidden_states.ndim
        if hs_ndim == 2:
            chex.assert_rank(segment_ids, 1)
            chex.assert_rank(triplet_type, 1)
            chex.assert_rank(attention_mask, 1)
        elif hs_ndim == 3:
            chex.assert_rank(segment_ids, 2)
            chex.assert_rank(triplet_type, 2)
            chex.assert_rank(attention_mask, 2)
        else:
            raise ValueError(f"Unexpected hidden_states dimension: {hs_ndim}")

        if attention_mask is None:
            attention_mask = jnp.ones_like(segment_ids)

        # --- 核心逻辑 (vmap 自动处理 Batch) ---
        def _process_single_batch(h_states, s_ids, s_types, mask):
            seq_len = s_ids.shape[0]

            # # A. 确定有效 Token
            # # 主要依赖 attention_mask 来过滤 padding (mask=0)
            # # 同时过滤非法 Type (<0)
            # is_valid_token = (mask > 0) & (s_types >= 0)

            # # B. 边界检查
            # # 确保 Sample ID 和 Type ID 在预设范围内
            # # Padding positions use segment_id = max(actual_segment_ids) + 1,
            # # which will be >= max_samples in most cases, but we rely on attention_mask for filtering
            # in_bounds = (s_ids < max_samples) & (s_types < num_slots)

            # total_valid_mask = is_valid_token & in_bounds

            # A + B. 快速路径，只用 attention_mask 来过滤 padding
            total_valid_mask = mask > 0

            # C. 计算 Scatter Index (扁平化索引)
            # 唯一 ID = sample_id * num_slots + type_id
            # 例如: Sample 0 的 Anchor=0, Pos=1, Neg1=2...
            #       Sample 1 的 Anchor=num_slots, ...
            scatter_indices = s_ids * num_slots + s_types

            # 将无效位置设为越界值 (max_samples * num_slots)
            out_of_bound_idx = max_samples * num_slots
            safe_scatter_indices = jnp.where(
                total_valid_mask, scatter_indices, out_of_bound_idx
            )

            # D. 寻找 EOS Token Index (Scatter Max)
            # 这里的逻辑是：对于同一个 (Sample, Type) 组，
            # index 最大的那个 token 自然就是 EOS token (因为是 causal mask，顺序排列)。
            # 我们不需要知道 EOS 具体是哪个数值，只需要取最后一个即可。

            # 初始化为 -1 (代表该槽位为空)
            init_indices = jnp.full((max_samples * num_slots,), -1, dtype=jnp.int32)
            seq_indices = jnp.arange(seq_len, dtype=jnp.int32)

            # 找到每个槽位的最大索引 (即 EOS 位置)
            eos_indices = init_indices.at[safe_scatter_indices].max(
                seq_indices, mode="drop"
            )

            # E. 提取 (Gather)
            # 如果 eos_index 是 -1，用 0 代替防止 gather 越界，随后会 mask 掉
            safe_gather_indices = jnp.maximum(eos_indices, 0)
            extracted = h_states[safe_gather_indices]  # [Total_Slots, H]

            # F. 生成 Slot Mask
            # 只有 eos_index != -1 的槽位才是真实存在的
            slot_exists_mask = eos_indices != -1

            # G. Mask 掉无效 Embedding
            final_embeddings = extracted * slot_exists_mask[..., None]

            # Reshape 回结构化形状
            # [Max_Samples, Num_Slots, Hidden]
            return (
                final_embeddings.reshape(max_samples, num_slots, -1),
                slot_exists_mask.reshape(max_samples, num_slots),
            )

        if hs_ndim == 2:
            return _process_single_batch(
                hidden_states, segment_ids, triplet_type, attention_mask
            )
        elif hs_ndim == 3:
            return jax.vmap(_process_single_batch)(
                hidden_states, segment_ids, triplet_type, attention_mask
            )
        else:
            raise ValueError(f"Unexpected hidden_states dimension: {hs_ndim}")


def _ensure_qwen3_embedding_registration() -> None:
    """Register custom Qwen3EmbeddingModel with EasyDeL so Auto loaders can locate it."""
    if getattr(ed, CUSTOM_ARCHITECTURE_NAME, None) is not Qwen3EmbeddingModel:
        setattr(ed, CUSTOM_ARCHITECTURE_NAME, Qwen3EmbeddingModel)
    try:
        existing = registry.get_module_registration(
            TaskType.BASE_MODULE, CUSTOM_MODEL_TYPE
        )
        if existing.module is Qwen3EmbeddingModel:
            return
    except AssertionError:
        pass
    registry.register_module(
        task_type=TaskType.BASE_MODULE,
        config=Qwen3EmbeddingConfig,
        model_type=CUSTOM_MODEL_TYPE,
        embedding_layer_names=["embed_tokens"],
        layernorm_names=["norm"],
    )(Qwen3EmbeddingModel)


_ensure_qwen3_embedding_registration()


def initialize_from_base_model_state(
    target_model: Qwen3EmbeddingModel,
    source_base_model: EasyDeLBaseModule,
    source_projection: Optional[ProjectionHead] = None,
) -> None:
    """
    Initialize Qwen3EmbeddingModel state from base_model state (external function)

    This function extracts state from an external base_model and projection,
    and initializes the target Qwen3EmbeddingModel's internal state.
    The target_model should be a newly created Qwen3EmbeddingModel instance.

    Args:
      target_model: Target Qwen3EmbeddingModel to initialize
      source_base_model: Source base model to extract state from
      source_projection: Optional source projection head to extract state from
                        (if None, projection will remain randomly initialized)
    """
    # Extract state from source base_model
    source_base_state = nnx.state(source_base_model)

    # Get target model's internal base_model state structure
    target_base_state = nnx.state(target_model.base_model)

    # Copy state from source to target base_model
    def _copy_state_recursive(target_dict: dict, source_dict: dict, path: str = ""):
        """Recursively copy state from source to target"""
        copied_keys = []
        skipped_keys = []
        for key in source_dict:
            if key not in target_dict:
                skipped_keys.append(f"{path}.{key}" if path else key)
                continue
            current_path = f"{path}.{key}" if path else key
            if isinstance(source_dict[key], dict) and isinstance(
                target_dict[key], dict
            ):
                sub_copied, sub_skipped = _copy_state_recursive(
                    target_dict[key], source_dict[key], current_path
                )
                copied_keys.extend(sub_copied)
                skipped_keys.extend(sub_skipped)
            else:
                try:
                    # Direct assignment for leaf nodes
                    target_dict[key] = source_dict[key]
                    copied_keys.append(current_path)
                except Exception as e:
                    skipped_keys.append(f"{current_path} (error: {e})")
        return copied_keys, skipped_keys

    # Copy base model state
    print("Copying base model state...")
    copied_keys, skipped_keys = _copy_state_recursive(
        target_base_state, source_base_state
    )

    if skipped_keys:
        print(f"  - Skipped {len(skipped_keys)} keys (not found in target or error)")

    # Update target model's base_model with copied state
    nnx.update(target_model.base_model, target_base_state)
    print(f"  - Copied {len(copied_keys)} state keys from base_model")

    # Copy projection state if provided
    if source_projection is not None:
        print("Copying projection head state...")
        source_proj_state = nnx.state(source_projection)
        target_proj_state = nnx.state(target_model.projection)
        proj_copied, proj_skipped = _copy_state_recursive(
            target_proj_state, source_proj_state
        )
        nnx.update(target_model.projection, target_proj_state)
        print(f"  - Copied {len(proj_copied)} state keys from projection")
        if proj_skipped:
            print(f"  - Skipped {len(proj_skipped)} projection keys")
    else:
        print("Projection head will remain randomly initialized")

    print("State initialization completed")


def create_from_initial_model(
    initial_model: str,
    embedding_dim: int = 256,
    dtype: jnp.dtype = jnp.bfloat16,
    param_dtype: jnp.dtype = jnp.bfloat16,
    seed: int = 42,
    quantization_config: Optional[Any] = None,
    *,
    rngs: Optional[nnx.Rngs] = None,
) -> Qwen3EmbeddingModel:
    """
    Create Qwen3EmbeddingModel from initial Qwen model on CPU

    This is a standalone function that loads the base model on CPU and creates
    a Qwen3EmbeddingModel with projection head. After creation, this function
    is no longer used - all operations (vocab extension, loading, saving) are
    performed directly on the Qwen3EmbeddingModel object.

    Args:
      initial_model: HuggingFace model name or local path to base Qwen model
      embedding_dim: Output embedding dimension
      dtype: Computation dtype
      param_dtype: Parameter dtype
      seed: Random seed for projection head initialization
      quantization_config: Optional EasyDeLQuantizationConfig (if None, uses default)
      rngs: Optional random number generators

    Returns:
      Qwen3EmbeddingModel instance (on CPU, no mesh)
    """

    if rngs is None:
        key = jax.random.PRNGKey(seed)
        rngs = nnx.Rngs(key)

    # Force CPU backend for model conversion
    backend = ed.EasyDeLBackends.CPU
    # Use single device configuration for CPU
    final_sharding_axis_dims = (1, 1, 1, 1, 1)

    # Check if local path or HuggingFace
    is_local_path = os.path.exists(initial_model) and os.path.isdir(initial_model)
    from_torch = not is_local_path

    if is_local_path:
        print(f"Loading base model from local path (CPU): {initial_model}")
    else:
        print(f"Loading base model from HuggingFace (CPU): {initial_model}")

    try:
        # Build kwargs for from_pretrained
        load_kwargs = {
            "dtype": dtype,
            "param_dtype": param_dtype,
            "precision": jax.lax.Precision.DEFAULT,
            "backend": backend,
            "from_torch": from_torch,
            "auto_shard_model": True,
            "sharding_axis_dims": final_sharding_axis_dims,
            "sharding_axis_names": ("dp", "fsdp", "ep", "tp", "sp"),
            "config_kwargs": ed.EasyDeLBaseConfigDict(
                attn_mechanism=ed.AttentionMechanisms.VANILLA,
                gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NONE,
            ),
            "verbose": True,
            "trust_remote_code": True,
        }

        base_model = ed.AutoEasyDeLModel.from_pretrained(initial_model, **load_kwargs)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load base model from {initial_model}. Error: {e}"
        ) from e

    # Get config
    if hasattr(base_model, "config"):
        config = base_model.config
    else:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(initial_model, trust_remote_code=True)

    # Create combined model config as Qwen3EmbeddingConfig instance
    if config is not None:
        # Extract all necessary parameters from original config
        combined_config = Qwen3EmbeddingConfig(
            vocab_size=getattr(config, "vocab_size", 151936),
            hidden_size=getattr(config, "hidden_size", 4096),
            intermediate_size=getattr(config, "intermediate_size", 22016),
            num_hidden_layers=getattr(config, "num_hidden_layers", 32),
            num_attention_heads=getattr(config, "num_attention_heads", 32),
            num_key_value_heads=getattr(config, "num_key_value_heads", 32),
            head_dim=getattr(config, "head_dim", 128),
            hidden_act=getattr(config, "hidden_act", "silu"),
            max_position_embeddings=getattr(config, "max_position_embeddings", 32768),
            initializer_range=getattr(config, "initializer_range", 0.02),
            rms_norm_eps=getattr(config, "rms_norm_eps", 1e-6),
            use_cache=getattr(config, "use_cache", True),
            tie_word_embeddings=getattr(config, "tie_word_embeddings", False),
            rope_theta=getattr(config, "rope_theta", 10000.0),
            rope_scaling=getattr(config, "rope_scaling", None),
            attention_bias=getattr(config, "attention_bias", False),
            use_sliding_window=getattr(config, "use_sliding_window", False),
            sliding_window=getattr(config, "sliding_window", 4096),
            max_window_layers=getattr(config, "max_window_layers", 28),
            attention_dropout=getattr(config, "attention_dropout", 0.0),
            layer_types=getattr(config, "layer_types", None),
            embedding_dim=embedding_dim,
            combined_base_model_type=getattr(config, "model_type", BASE_MODEL_TYPE),
            combined_base_architectures=getattr(config, "architectures", []),
            model_name_or_path=initial_model,
        )
    else:
        combined_config = None

    # Get dtype and precision from base_model
    model_dtype = getattr(base_model, "dtype", dtype)
    model_param_dtype = getattr(base_model, "param_dtype", param_dtype)
    precision = getattr(base_model, "precision", jax.lax.Precision.DEFAULT)

    # Create new independent model from config (without base_model dependency)
    print("Creating new Qwen3EmbeddingModel from config in CPU mesh...")
    mesh = jax.sharding.Mesh([jax.devices("cpu")[0]], axis_names=("dp",))
    with mesh:
        model = Qwen3EmbeddingModel(
            config=combined_config or config,
            embedding_dim=embedding_dim,
            dtype=model_dtype,
            param_dtype=model_param_dtype,
            precision=precision,
            rngs=rngs,
            quantization_config=quantization_config,
        )

        # Initialize model state from base_model using external function
        print("Initializing model state from base_model...")
        initialize_from_base_model_state(
            target_model=model,
            source_base_model=base_model,
            source_projection=None,  # Projection will remain randomly initialized
        )

    # Store original vocab size for later use
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(initial_model, trust_remote_code=True)
    model._original_vocab_size = len(tokenizer)
    del tokenizer

    print("Model created successfully on CPU")
    return model


# ============================================================================
# Tokenizer helper functions are now in vocab_embedding_utils.py
# Import them from there for backward compatibility
# ============================================================================
