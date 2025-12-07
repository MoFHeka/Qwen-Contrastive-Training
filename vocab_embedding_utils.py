"""
Vocabulary and Embedding Utilities for Qwen3-Embedding

This module provides utilities for vocabulary extension and embedding layer operations.
It separates vocabulary/embedding concerns from the main model implementation to improve
code maintainability and reduce coupling.
"""

from typing import List, Tuple, Optional, Any

import jax
import jax.numpy as jnp
from flax import nnx
import chex

from easydel.infra import EasyDeLBaseModule

# Re-export Qwen3EmbeddingModel for type hints (avoid circular import)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qwen3_embedding_modeling import Qwen3EmbeddingModel


# ============================================================================
# Embedding Layer Utilities
# ============================================================================


def get_embedding_layer(model: EasyDeLBaseModule) -> Optional[Any]:
    """
    Get embedding layer from base model

    Args:
      model: Base model to search for embedding layer

    Returns:
      Embedding layer if found, None otherwise
    """
    # Try common embedding layer names
    if hasattr(model, "embed_tokens"):
        return model.embed_tokens
    elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens
    elif hasattr(model, "embeddings"):
        return model.embeddings

    # Search for embedding layer in attributes
    for attr_name in dir(model):
        if attr_name.startswith("_"):
            continue
        attr = getattr(model, attr_name)
        if hasattr(attr, "embedding") or (
            hasattr(attr, "kernel") and len(attr.kernel.shape) == 2
        ):
            return attr

    return None


def get_embedding_weights(embed_layer: Any) -> Optional[chex.Array]:
    """
    Get embedding weights from embedding layer

    Args:
      embed_layer: Embedding layer

    Returns:
      Embedding weights array if found, None otherwise
    """
    if hasattr(embed_layer, "embedding"):
        return embed_layer.embedding
    elif hasattr(embed_layer, "kernel"):
        return embed_layer.kernel
    else:
        try:
            if isinstance(embed_layer, nnx.Variable):
                return embed_layer.value
            else:
                state = nnx.state(embed_layer)
                if "embedding" in state:
                    return state["embedding"]
                elif "kernel" in state:
                    return state["kernel"]
        except Exception:
            pass
    return None


def set_embedding_weights(embed_layer: Any, weights: chex.Array) -> None:
    """
    Set embedding weights to embedding layer

    Args:
      embed_layer: Embedding layer
      weights: New embedding weights
    """
    if hasattr(embed_layer, "embedding"):
        embed_layer.embedding = weights
    elif hasattr(embed_layer, "kernel"):
        embed_layer.kernel = weights
    else:
        try:
            state = nnx.state(embed_layer)
            if "embedding" in state:
                state["embedding"] = weights
            elif "kernel" in state:
                state["kernel"] = weights
            else:
                raise RuntimeError("Could not update embedding weights")
        except Exception as e:
            raise RuntimeError(f"Could not update embedding weights: {e}") from e


# ============================================================================
# Vocabulary Extension Utilities
# ============================================================================


def initialize_new_embeddings(
    num_added: int,
    hidden_size: int,
    dtype: jnp.dtype,
    initialization_strategy: str,
    current_embeddings: Optional[chex.Array] = None,
    rngs: Optional[nnx.Rngs] = None,
) -> chex.Array:
    """
    Initialize new token embeddings based on strategy

    Args:
      num_added: Number of new tokens
      hidden_size: Hidden size of embeddings
      dtype: Data type
      initialization_strategy: Strategy for initialization ("average", "random", "zeros")
      current_embeddings: Current embeddings array (required for "average" strategy)
      rngs: Random number generators (required for "random" strategy)

    Returns:
      New embeddings array [num_added, hidden_size]

    Raises:
      ValueError: If strategy is invalid or required parameters are missing
    """
    if initialization_strategy == "average":
        if current_embeddings is None:
            raise ValueError(
                "current_embeddings must be provided for 'average' initialization strategy"
            )
        new_embeddings = jnp.mean(current_embeddings, axis=0, keepdims=True)
        new_embeddings = jnp.repeat(new_embeddings, num_added, axis=0)
        return new_embeddings
    elif initialization_strategy == "random":
        if rngs is None:
            raise ValueError("rngs must be provided for random initialization")
        key = rngs.params()
        return (
            jax.random.normal(key, shape=(num_added, hidden_size), dtype=dtype) * 0.02
        )
    elif initialization_strategy == "zeros":
        return jnp.zeros((num_added, hidden_size), dtype=dtype)
    else:
        raise ValueError(
            f"Unknown initialization_strategy: {initialization_strategy}. "
            "Must be one of: 'average', 'random', 'zeros'"
        )


def update_vocab_config(
    model_config: Any,
    base_model: Optional[EasyDeLBaseModule],
    new_vocab_size: int,
) -> None:
    """
    Update vocab_size in model configs

    Args:
      model_config: Main model configuration object
      base_model: Base model instance (optional, for updating internal config)
      new_vocab_size: New vocabulary size
    """
    if hasattr(model_config, "vocab_size"):
        model_config.vocab_size = new_vocab_size
    # Update internal base_model config as well
    if base_model is not None:
        if hasattr(base_model, "config") and hasattr(base_model.config, "vocab_size"):
            base_model.config.vocab_size = new_vocab_size


def extend_model_embeddings(
    base_model: EasyDeLBaseModule,
    num_new_tokens: int,
    initialization_strategy: str = "average",
    *,
    rngs: Optional[nnx.Rngs] = None,
) -> Tuple[int, int]:
    """
    Extend model embeddings by adding new token embeddings (pure EasyDeL method)

    This function operates on the embedding layer and config, without
    any dependency on transformers tokenizer. Tokenizer operations should
    be handled separately using helper functions if needed.

    Args:
      base_model: Base model containing the embedding layer
      num_new_tokens: Number of new tokens to add to vocabulary
      initialization_strategy: How to initialize new token embeddings
        - "average": Average of all existing embeddings (default)
        - "random": Random initialization
        - "zeros": Zero initialization
      rngs: Random number generators (required for random initialization)

    Returns:
      Tuple of (original_vocab_size, new_vocab_size)

    Raises:
      RuntimeError: If embedding layer cannot be found or accessed
      ValueError: If num_new_tokens is not positive
    """
    if num_new_tokens <= 0:
        raise ValueError("num_new_tokens must be positive")

    # Get embedding layer and weights
    embed_layer = get_embedding_layer(base_model)
    if embed_layer is None:
        raise RuntimeError(
            "Could not find embedding layer in model. Please check the model structure."
        )

    current_embeddings = get_embedding_weights(embed_layer)
    if current_embeddings is None:
        raise RuntimeError(
            "Could not access embedding weights. Please check the model structure."
        )

    if len(current_embeddings.shape) != 2:
        raise RuntimeError(f"Unexpected embedding shape: {current_embeddings.shape}")

    original_vocab_size = current_embeddings.shape[0]
    hidden_size = current_embeddings.shape[1]

    # Initialize new token embeddings
    new_embeddings = initialize_new_embeddings(
        num_added=num_new_tokens,
        hidden_size=hidden_size,
        dtype=current_embeddings.dtype,
        initialization_strategy=initialization_strategy,
        current_embeddings=current_embeddings,
        rngs=rngs,
    )

    # Concatenate and update
    extended_embeddings = jnp.concatenate([current_embeddings, new_embeddings], axis=0)
    set_embedding_weights(embed_layer, extended_embeddings)

    new_vocab_size = original_vocab_size + num_new_tokens

    return original_vocab_size, new_vocab_size


def extend_model_vocab(
    model: "Qwen3EmbeddingModel",
    num_new_tokens: int,
    initialization_strategy: str = "average",
    *,
    rngs: Optional[nnx.Rngs] = None,
) -> Tuple[int, int]:
    """
    Extend vocabulary by adding new token embeddings (pure EasyDeL method)

    This function operates on the embedding layer and config, without
    any dependency on transformers tokenizer. Tokenizer operations should
    be handled separately using helper functions if needed.

    Args:
      model: Qwen3EmbeddingModel instance
      num_new_tokens: Number of new tokens to add to vocabulary
      initialization_strategy: How to initialize new token embeddings
        - "average": Average of all existing embeddings (default)
        - "random": Random initialization
        - "zeros": Zero initialization
      rngs: Random number generators (required for random initialization)

    Returns:
      Tuple of (original_vocab_size, new_vocab_size)

    Raises:
      ValueError: If num_new_tokens is not positive
      RuntimeError: If embedding layer cannot be found or accessed
    """
    # Use utility function to extend embeddings
    original_vocab_size, new_vocab_size = extend_model_embeddings(
        base_model=model.base_model,
        num_new_tokens=num_new_tokens,
        initialization_strategy=initialization_strategy,
        rngs=rngs,
    )

    # Update config
    update_vocab_config(
        model_config=model.config,
        base_model=model.base_model,
        new_vocab_size=new_vocab_size,
    )

    print("Successfully extended vocabulary and embedding layer")
    print(f"  Original vocab size: {original_vocab_size}")
    print(f"  New vocab size: {new_vocab_size}")
    print(f"  Added tokens: {num_new_tokens}")
    print(f"  Initialization strategy: {initialization_strategy}")

    return original_vocab_size, new_vocab_size


# ============================================================================
# Tokenizer Utilities (require transformers)
# ============================================================================


def get_tokenizer(model_name: str):
    """
    Get tokenizer from model name or path (independent function, requires transformers)

    This is an independent function that requires transformers library.
    Use this when you need to work with tokenizer separately from the model.

    Args:
      model_name: Model name or path

    Returns:
      Tokenizer instance

    Raises:
      ImportError: If transformers library is not installed
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError(
            "transformers library is required for tokenizer operations. "
            "Please install it with: pip install transformers"
        )
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def get_tokenizer_from_model(model: "Qwen3EmbeddingModel"):
    """
    Get tokenizer from Qwen3EmbeddingModel (convenience function, requires transformers)

    This function extracts the model name from model.config.model_name_or_path
    and loads the tokenizer. This hides implementation details from users.

    Args:
      model: Qwen3EmbeddingModel instance

    Returns:
      Tokenizer instance

    Raises:
      ValueError: If model.config.model_name_or_path is not set
    """
    if not hasattr(model, "config") or model.config is None:
        raise ValueError("Model config is not available")

    model_name_or_path = getattr(model.config, "model_name_or_path", None)
    if model_name_or_path is None:
        raise ValueError(
            "model_name_or_path is not set in model config. "
            "Please ensure the model was created with a valid model_name_or_path."
        )

    return get_tokenizer(model_name_or_path)


def extend_tokenizer_vocab(tokenizer, new_tokens: List[str]):
    """
    Extend tokenizer vocabulary by adding new tokens (independent function, requires transformers)

    This is an independent function that requires transformers library.
    After extending the tokenizer, you should call extend_model_embeddings() to update
    the model's embedding layer accordingly.

    Args:
      tokenizer: Tokenizer instance
      new_tokens: List of new token strings to add

    Returns:
      Tuple of (original_vocab_size, new_vocab_size, num_added)

    Raises:
      ValueError: If new_tokens is empty
    """
    if not new_tokens:
        raise ValueError("new_tokens cannot be empty")

    original_vocab_size = len(tokenizer)
    num_added = tokenizer.add_tokens(new_tokens)
    new_vocab_size = len(tokenizer)

    if num_added == 0:
        print(
            "Warning: No new tokens were added. They may already exist in the vocabulary."
        )

    return original_vocab_size, new_vocab_size, num_added


def save_tokenizer(tokenizer, save_path: str):
    """
    Save tokenizer to a directory (independent function, requires transformers)

    This is an independent function that requires transformers library.

    Args:
      tokenizer: Tokenizer instance
      save_path: Path to save the tokenizer
    """
    tokenizer.save_pretrained(save_path)
    print(f"Tokenizer saved to {save_path}")


def extend_model_vocab_with_tokens(
    model: "Qwen3EmbeddingModel",
    new_tokens: List[str],
    model_name: Optional[str] = None,
    initialization_strategy: str = "average",
    *,
    rngs: Optional[nnx.Rngs] = None,
) -> Tuple[int, int]:
    """
    Convenience function to extend both tokenizer and model vocabulary

    This function combines tokenizer extension (requires transformers) and
    model embedding extension (pure EasyDeL). It's a helper that orchestrates
    both operations.

    Args:
      model: Qwen3EmbeddingModel instance
      new_tokens: List of new token strings to add
      model_name: Model name or path for tokenizer (if None, uses model.config.model_name_or_path)
      initialization_strategy: How to initialize new token embeddings
        - "average": Average of all existing embeddings (default)
        - "random": Random initialization
        - "zeros": Zero initialization
      rngs: Random number generators (required for random initialization)

    Returns:
      Tuple of (original_vocab_size, new_vocab_size)

    Raises:
      ValueError: If new_tokens is empty or model_name cannot be determined
    """
    if not new_tokens:
        raise ValueError("new_tokens cannot be empty")

    # Get model name from config if not provided
    if model_name is None:
        if not hasattr(model, "config") or model.config is None:
            raise ValueError("Model config is not available")
        model_name = getattr(model.config, "model_name_or_path", None)
        if model_name is None:
            raise ValueError(
                "model_name must be provided or set in model.config.model_name_or_path"
            )

    # Extend tokenizer (requires transformers)
    tokenizer = get_tokenizer(model_name)
    original_vocab_size, new_vocab_size, num_added = extend_tokenizer_vocab(
        tokenizer, new_tokens
    )

    if num_added == 0:
        print(
            "Warning: No new tokens were added. They may already exist in the vocabulary."
        )
        return original_vocab_size, original_vocab_size

    print(
        f"Extended tokenizer vocabulary: {original_vocab_size} -> {new_vocab_size} (+{num_added} tokens)"
    )

    # Extend model embedding layer (pure EasyDeL)
    model_original_vocab, model_new_vocab = extend_model_vocab(
        model=model,
        num_new_tokens=num_added,
        initialization_strategy=initialization_strategy,
        rngs=rngs,
    )

    if model_original_vocab != original_vocab_size:
        print(
            f"Warning: Model vocab size ({model_original_vocab}) doesn't match "
            f"tokenizer vocab size ({original_vocab_size}) before extension"
        )

    return original_vocab_size, new_vocab_size
