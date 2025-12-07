"""
Generic Model I/O utilities for EasyDeL models

This module contains generic save and load functions for EasyDeL models,
separated from model-specific code to improve modularity and reduce coupling.
These functions work with any EasyDeLBaseModule instance.
"""

import os
import shutil
from typing import Optional, Any, Mapping, Sequence, Callable

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec

import optax

import easydel as ed
from easydel.infra import EasyDeLState, EasyDeLBaseModule
from easydel.infra.factory import TaskType


def load_easydel_state(
    load_directory: str,
    device: jax.Device | None = "cpu",  # type:ignore
    dtype: jnp.dtype = jnp.bfloat16,
    param_dtype: jnp.dtype = jnp.bfloat16,
    precision: jax.lax.Precision | None = None,
    sharding_axis_dims: Sequence[int] = (1, -1, 1, 1, 1),
    sharding_dcn_axis_dims: Sequence[int] | None = None,
    sharding_axis_names: Sequence[str] = ("dp", "fsdp", "ep", "tp", "sp"),
    partition_axis: ed.PartitionAxis | None = None,
    shard_fns: Mapping[tuple, Callable] | dict | None = None,
    backend: ed.EasyDeLBackends | None = None,
    platform: ed.EasyDeLPlatforms | None = None,
    config_kwargs: ed.EasyDeLBaseConfigDict | None = None,
    model_task: TaskType = TaskType.BASE_MODULE,
    auto_shard_model: bool = True,
    partition_rules: tuple[tuple[str, PartitionSpec], ...] | None = None,
    quantization_config: "ed.EasyDeLQuantizationConfig | None" = None,
    quantize_tensors: bool = True,
    verbose: bool = True,
    tx_template: optax.GradientTransformation | None = None,
    **kwargs,
) -> EasyDeLBaseModule:
    """
    Load EasyDeL model from saved path

    This function supports all parameters from EasyDeLState.load_state.
    Parameters are passed through to EasyDeLState.load_state.

    Args:
      load_directory: Path to saved model directory
      dtype: Computation dtype (default: jnp.bfloat16)
      param_dtype: Parameter dtype (default: jnp.bfloat16)
      device: JAX device to load onto (default: "cpu")
      precision: JAX precision level
      sharding_axis_dims: Sharding axis dimensions (dp, fsdp, ep, tp, sp).
                        If None, will be read from saved config or default to (1, 1, 1, 1, 1)
      sharding_dcn_axis_dims: Data-centric sharding dimensions
      sharding_axis_names: Sharding axis names. Defaults to ("dp", "fsdp", "ep", "tp", "sp")
      partition_axis: Configuration object for partitioning specific axes
      shard_fns: Optional mapping of parameter path tuples to custom sharding functions
      backend: The backend framework to use
      platform: The hardware platform
      config_kwargs: Optional dictionary of keyword arguments to override in the loaded model configuration
      model_task: The specific task type for the model (default: TaskType.BASE_MODULE)
      auto_shard_model: If True, automatically shards the loaded model (default: True)
      partition_rules: Optional tuple of partition rules (regex, PartitionSpec)
      quantization_config: Quantization configuration
      quantize_tensors: If True, applies quantization to the loaded tensors (default: True)
      verbose: If True, logs detailed information during loading (default: True)
      tx_template: Optimizer transformation template
      **kwargs: Additional keyword arguments passed to EasyDeLState.load_state

    Returns:
      Loaded EasyDeLBaseModule instance
    """
    if not os.path.exists(load_directory):
        raise ValueError(f"model_path must exist: {load_directory}")

    # For non-CPU, check if sharding_axis_dims matches device count
    if device is not None and device != "cpu":
        try:
            devices = jax.devices()
            total_devices = len(devices)
            current_total = 1
            for dim in sharding_axis_dims:
                if dim > 0:
                    current_total *= dim
                elif dim == -1:
                    # -1 means auto-allocate, will be handled by EasyDeL
                    current_total = -1
                    break

            # If sharding_axis_dims doesn't match device count, adjust it
            if current_total > 0 and current_total != total_devices:
                if verbose:
                    print(
                        f"  - Warning: sharding_axis_dims product ({current_total}) "
                        f"doesn't match device count ({total_devices})"
                    )
                    print("  - Adjusting to use all devices with fsdp")
                # Use all devices for fsdp, but ensure embedding layer is not sharded
                sharding_axis_dims = (1, total_devices, 1, 1, 1)
        except Exception as e:
            if verbose:
                print(f"  - Warning: Could not detect device count: {e}")

    if verbose:
        print(f"Loading EasyDeLState from {load_directory}...")
        if device is not None:
            print(f"  - device: {device}")
        print(f"  - sharding_axis_dims: {sharding_axis_dims}")
        print(f"  - sharding_axis_names: {sharding_axis_names}")
        if backend is not None:
            print(f"  - backend: {backend}")

    # Build load_state kwargs with all parameters
    # Note: load_directory is a positional argument, not keyword
    load_kwargs = {
        "device": device,
        "dtype": dtype,
        "param_dtype": param_dtype,
        "precision": precision,
        "model_task": model_task,
        "sharding_axis_dims": sharding_axis_dims,
        "sharding_dcn_axis_dims": sharding_dcn_axis_dims,
        "sharding_axis_names": sharding_axis_names,
        "partition_axis": partition_axis,
        "shard_fns": shard_fns,
        "backend": backend,
        "platform": platform,
        "config_kwargs": config_kwargs,
        "auto_shard_model": auto_shard_model,
        "partition_rules": partition_rules,
        "quantization_config": quantization_config,
        "quantize_tensors": quantize_tensors,
        "verbose": verbose,
        "tx_template": tx_template,
    }

    # load_directory is the first positional argument
    easydel_state = EasyDeLState.load_state(load_directory, **load_kwargs)

    return easydel_state


def load_easydel_model(
    load_directory: str,
    device: jax.Device | None = "cpu",  # type:ignore
    dtype: jnp.dtype = jnp.bfloat16,
    param_dtype: jnp.dtype = jnp.bfloat16,
    precision: jax.lax.Precision | None = None,
    sharding_axis_dims: Sequence[int] = (1, -1, 1, 1, 1),
    sharding_dcn_axis_dims: Sequence[int] | None = None,
    sharding_axis_names: Sequence[str] = ("dp", "fsdp", "ep", "tp", "sp"),
    partition_axis: ed.PartitionAxis | None = None,
    shard_fns: Mapping[tuple, Callable] | dict | None = None,
    backend: ed.EasyDeLBackends | None = None,
    platform: ed.EasyDeLPlatforms | None = None,
    config_kwargs: ed.EasyDeLBaseConfigDict | None = None,
    model_task: TaskType = TaskType.BASE_MODULE,
    auto_shard_model: bool = True,
    partition_rules: tuple[tuple[str, PartitionSpec], ...] | None = None,
    quantization_config: "ed.EasyDeLQuantizationConfig | None" = None,
    quantize_tensors: bool = True,
    verbose: bool = True,
    tx_template: optax.GradientTransformation | None = None,
    **kwargs,
) -> EasyDeLBaseModule:
    """
    Load EasyDeL model from saved path

    This function supports all parameters from EasyDeLState.load_state.
    Parameters are passed through to EasyDeLState.load_state.

    Args:
      model_path: Path to saved model directory
      dtype: Computation dtype (default: jnp.bfloat16)
      param_dtype: Parameter dtype (default: jnp.bfloat16)
      device: JAX device to load onto (default: "cpu")
      precision: JAX precision level
      sharding_axis_dims: Sharding axis dimensions (dp, fsdp, ep, tp, sp).
                        If None, will be read from saved config or default to (1, 1, 1, 1, 1)
      sharding_dcn_axis_dims: Data-centric sharding dimensions
      sharding_axis_names: Sharding axis names. Defaults to ("dp", "fsdp", "ep", "tp", "sp")
      partition_axis: Configuration object for partitioning specific axes
      shard_fns: Optional mapping of parameter path tuples to custom sharding functions
      backend: The backend framework to use
      platform: The hardware platform
      config_kwargs: Optional dictionary of keyword arguments to override in the loaded model configuration
      model_task: The specific task type for the model (default: TaskType.BASE_MODULE)
      auto_shard_model: If True, automatically shards the loaded model (default: True)
      partition_rules: Optional tuple of partition rules (regex, PartitionSpec)
      quantization_config: Quantization configuration
      quantize_tensors: If True, applies quantization to the loaded tensors (default: True)
      verbose: If True, logs detailed information during loading (default: True)
      tx_template: Optimizer transformation template
      **kwargs: Additional keyword arguments passed to EasyDeLState.load_state

    Returns:
      Loaded EasyDeLBaseModule instance
    """

    easydel_state = load_easydel_state(
        load_directory=load_directory,
        device=device,
        dtype=dtype,
        param_dtype=param_dtype,
        precision=precision,
        sharding_axis_dims=sharding_axis_dims,
        sharding_dcn_axis_dims=sharding_dcn_axis_dims,
        sharding_axis_names=sharding_axis_names,
        partition_axis=partition_axis,
        shard_fns=shard_fns,
        backend=backend,
        platform=platform,
        config_kwargs=config_kwargs,
        model_task=model_task,
        auto_shard_model=auto_shard_model,
        partition_rules=partition_rules,
        quantization_config=quantization_config,
        quantize_tensors=quantize_tensors,
        verbose=verbose,
        tx_template=tx_template,
        **kwargs,
    )

    loaded_model = easydel_state.model

    if loaded_model is None:
        raise RuntimeError(
            "Failed to load model: EasyDeLState.model is None. "
            "This may indicate a problem with the saved model state."
        )

    if not isinstance(loaded_model, EasyDeLBaseModule):
        actual_type = type(loaded_model)
        raise RuntimeError(
            f"Loaded model is not an EasyDeLBaseModule instance. "
            f"Got type: {actual_type}. "
            f"This may indicate a registration issue or model type mismatch."
        )

    if verbose:
        print("Model loaded successfully from EasyDeLState")
        print(f"  - Model type: {type(loaded_model)}")
        step_value = getattr(easydel_state, "step", None)
        if step_value is not None:
            step_int = (
                int(step_value) if isinstance(step_value, (int, float)) else step_value
            )
            print(f"  - Training step: {step_int}")

    return loaded_model


def save_easydel_state(
    model: EasyDeLBaseModule,
    save_path: str,
    optimizer: Optional[optax.GradientTransformation],
    step: int,
    overwrite: bool = False,
    dtype: Optional[jnp.dtype] = None,
    model_type: Optional[str] = None,
    architectures: Optional[list[str]] = None,
) -> None:
    """
    Save EasyDeLState to disk

    Args:
      model: EasyDeLBaseModule instance to save
      save_path: Base save path
      optimizer: Optional optimizer
      step: Training step
      overwrite: If True, overwrite existing model directory
      dtype: Computation dtype (if None, uses model.dtype)
      model_type: Optional model type to set in config (if None, preserves existing)
      architectures: Optional architectures list to set in config (if None, preserves existing)
    """
    if dtype is None:
        dtype = getattr(model, "dtype", jnp.bfloat16)

    print("Creating EasyDeLState from model...")

    # Optionally update config model_type and architectures if provided
    if hasattr(model, "config") and model.config is not None:
        if model_type is not None:
            model.config.model_type = model_type
        if architectures is not None:
            model.config.architectures = list(architectures)

    # Build state_kwargs with only supported parameters
    state_kwargs = {"model": model, "step": step}
    if optimizer is not None:
        state_kwargs["tx"] = optimizer
        state_kwargs["init_opt_state"] = True

    easydel_state = EasyDeLState.create(**state_kwargs)

    # Clean up old model directory to avoid zarr chunks mismatch errors
    if os.path.exists(save_path):
        if overwrite:
            print(f"Cleaning up old model directory: {save_path}")
            shutil.rmtree(save_path)
        else:
            raise ValueError(
                f"Model directory already exists: {save_path}. "
                f"Set overwrite=True to overwrite, or use a different save_path."
            )

    print(f"Saving EasyDeLState to {save_path}...")
    easydel_state.save_state(
        save_path,
        float_dtype=dtype,
        save_optimizer=(optimizer is not None),
        step=step,
    )
    print("EasyDeLState saved successfully")


def _save_orbax(
    model: EasyDeLBaseModule,
    save_path: str,
    optimizer: Optional[optax.GradientTransformation],
    step: int,
    overwrite: bool = False,
) -> None:
    """
    Save model using Orbax checkpoint format

    Args:
      model: EasyDeLBaseModule instance to save
      save_path: Base save path
      optimizer: Optional optimizer
      step: Training step
      overwrite: If True, overwrite existing checkpoint directory

    Note:
      This is a placeholder implementation. Full Orbax support may be added in the future.
    """
    raise NotImplementedError(
        "Orbax checkpoint saving is not yet implemented. "
        "Please use save_method='easydel' instead."
    )


def save_easydel_model(
    model: EasyDeLBaseModule,
    save_path: str,
    optimizer: Optional[optax.GradientTransformation] = None,
    step: int = 0,
    save_method: str = "easydel",
    overwrite: bool = False,
    dtype: Optional[jnp.dtype] = None,
    model_type: Optional[str] = None,
    architectures: Optional[list[str]] = None,
) -> None:
    """
    Save EasyDeL model using specified save method

    This function saves the model using either EasyDeLState or Orbax checkpoint.
    The saved state includes:
    1. Model parameters
    2. Optimizer state (if provided)
    3. Training step count

    Note: Sharding/mesh configuration is NOT saved. Sharding should always be
    initialized or set by upper layer calls when loading the model.

    Args:
      model: EasyDeLBaseModule instance to save
      save_path: Path to save the model
      optimizer: Optional optimizer transformation (for saving optimizer state)
      step: Current training step (default 0)
      save_method: Save method to use, either "easydel" or "orbax" (default "easydel")
      overwrite: If True, overwrite existing model directory. If False, raise error if directory exists (default False)
      dtype: Computation dtype (if None, uses model.dtype)
      model_type: Optional model type to set in config (if None, preserves existing)
      architectures: Optional architectures list to set in config (if None, preserves existing)
    """
    if save_method not in ["easydel", "orbax"]:
        raise ValueError(
            f"Invalid save_method: {save_method}. Must be 'easydel' or 'orbax'"
        )

    # Check if save_path already exists
    if os.path.exists(save_path) and not overwrite:
        raise ValueError(
            f"Save path already exists: {save_path}. "
            f"Set overwrite=True to overwrite existing model, or use a different path."
        )

    os.makedirs(save_path, exist_ok=True)
    print(f"Saving model to {save_path} using {save_method}...")

    try:
        # Save model state using selected method
        if save_method == "easydel":
            save_easydel_state(
                model,
                save_path,
                optimizer,
                step,
                overwrite,
                dtype,
                model_type,
                architectures,
            )
        elif save_method == "orbax":
            _save_orbax(model, save_path, optimizer, step, overwrite)

        # Print success info
        print(f"\nModel saved successfully to {save_path}")
        if save_method == "easydel":
            print(f"  - EasyDeLState: {os.path.join(save_path, 'model')}")
        elif save_method == "orbax":
            print(
                f"  - Orbax checkpoint: {os.path.join(save_path, 'orbax_checkpoint')}"
            )
        if optimizer is not None:
            if save_method == "easydel":
                print("  - Optimizer state: saved in EasyDeLState")
            elif save_method == "orbax":
                print("  - Optimizer state: saved in Orbax checkpoint")
        print(f"  - Training step: {step}")

    except Exception as e:
        import traceback

        print(f"Error: Could not save using {save_method}: {e}")
        traceback.print_exc()
        raise RuntimeError(f"Failed to save model using {save_method}: {e}") from e


# Backward compatibility aliases for Qwen3EmbeddingModel
def load_qwen3_embedding_model(
    model_path: str,
    device: jax.Device | None = "cpu",  # type:ignore
    dtype: jnp.dtype = jnp.bfloat16,
    param_dtype: jnp.dtype = jnp.bfloat16,
    precision: jax.lax.Precision | None = None,
    sharding_axis_dims: Sequence[int] = (1, -1, 1, 1, 1),
    sharding_dcn_axis_dims: Sequence[int] | None = None,
    sharding_axis_names: Sequence[str] = ("dp", "fsdp", "ep", "tp", "sp"),
    partition_axis: ed.PartitionAxis | None = None,
    shard_fns: Mapping[tuple, Callable] | dict | None = None,
    backend: ed.EasyDeLBackends | None = None,
    platform: ed.EasyDeLPlatforms | None = None,
    config_kwargs: ed.EasyDeLBaseConfigDict | None = None,
    model_task: TaskType = TaskType.BASE_MODULE,
    auto_shard_model: bool = True,
    partition_rules: tuple[tuple[str, PartitionSpec], ...] | None = None,
    quantization_config: "ed.EasyDeLQuantizationConfig | None" = None,
    quantize_tensors: bool = True,
    verbose: bool = True,
    tx_template: optax.GradientTransformation | None = None,
    **kwargs,
) -> Any:
    """
    Load Qwen3EmbeddingModel from saved path (backward compatibility wrapper)

    This is a convenience wrapper around load_easydel_model that ensures
    Qwen3EmbeddingModel registration and handles Qwen3EmbeddingModel-specific
    post-processing.

    Args:
      Same as load_easydel_model

    Returns:
      Loaded Qwen3EmbeddingModel instance
    """
    # Import here to avoid circular imports
    from qwen3_embedding_modeling import (
        Qwen3EmbeddingModel,
        _ensure_qwen3_embedding_registration,
    )

    _ensure_qwen3_embedding_registration()

    loaded_model = load_easydel_model(
        model_path=model_path,
        device=device,
        dtype=dtype,
        param_dtype=param_dtype,
        precision=precision,
        sharding_axis_dims=sharding_axis_dims,
        sharding_dcn_axis_dims=sharding_dcn_axis_dims,
        sharding_axis_names=sharding_axis_names,
        partition_axis=partition_axis,
        shard_fns=shard_fns,
        backend=backend,
        platform=platform,
        config_kwargs=config_kwargs,
        model_task=model_task,
        auto_shard_model=auto_shard_model,
        partition_rules=partition_rules,
        quantization_config=quantization_config,
        quantize_tensors=quantize_tensors,
        verbose=verbose,
        tx_template=tx_template,
        **kwargs,
    )

    if isinstance(loaded_model, Qwen3EmbeddingModel):
        # Get embedding_dim from model if not already set
        if (
            not hasattr(loaded_model, "embedding_dim")
            or loaded_model.embedding_dim is None
        ):
            if hasattr(loaded_model, "projection") and hasattr(
                loaded_model.projection, "output_dim"
            ):
                loaded_model.embedding_dim = loaded_model.projection.output_dim
            else:
                loaded_model.embedding_dim = 256  # Default value

        if verbose:
            print(f"  - Has base_model: {hasattr(loaded_model, 'base_model')}")
            print(f"  - Has projection: {hasattr(loaded_model, 'projection')}")

        return loaded_model
    else:
        actual_type = type(loaded_model)
        raise RuntimeError(
            f"Loaded model is not a Qwen3EmbeddingModel instance. "
            f"Got type: {actual_type}. "
            f"This may indicate a registration issue or model type mismatch."
        )


def save_qwen3_embedding_model(
    model: Any,
    save_path: str,
    optimizer: Optional[optax.GradientTransformation] = None,
    step: int = 0,
    save_method: str = "easydel",
    overwrite: bool = False,
    dtype: Optional[jnp.dtype] = None,
) -> None:
    """
    Save Qwen3EmbeddingModel using specified save method (backward compatibility wrapper)

    This is a convenience wrapper around save_easydel_model that handles
    Qwen3EmbeddingModel-specific configuration.

    Args:
      model: Qwen3EmbeddingModel instance to save
      save_path: Path to save the model
      optimizer: Optional optimizer transformation (for saving optimizer state)
      step: Current training step (default 0)
      save_method: Save method to use, either "easydel" or "orbax" (default "easydel")
      overwrite: If True, overwrite existing model directory. If False, raise error if directory exists (default False)
      dtype: Computation dtype (if None, uses model.dtype)
    """
    from qwen3_embedding_modeling import (
        CUSTOM_MODEL_TYPE,
        CUSTOM_ARCHITECTURE_NAME,
    )

    save_easydel_model(
        model=model,
        save_path=save_path,
        optimizer=optimizer,
        step=step,
        save_method=save_method,
        overwrite=overwrite,
        dtype=dtype,
        model_type=CUSTOM_MODEL_TYPE,
        architectures=[CUSTOM_ARCHITECTURE_NAME],
    )
