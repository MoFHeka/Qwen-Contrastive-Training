"""
InfoNCE loss function for contrastive learning
"""

import chex
import jax
import jax.numpy as jnp
from jax import custom_vjp

from typing import Tuple, Dict


def inspect_grad(x, name):
    """
    Identity Function for print gradients
    """

    @custom_vjp
    def _probe(val):
        return val

    def _probe_fwd(val):
        return val, ()

    def _probe_bwd(res, g):
        jax.debug.print(
            "[{n}] Grad Shape: {s} | Mean: {m} | Min: {mi} | Max: {ma}",
            n=name,
            s=g.shape,
            m=jnp.mean(g),
            mi=jnp.min(g),
            ma=jnp.max(g),
        )
        return (g,)

    _probe.defvjp(_probe_fwd, _probe_bwd)

    return _probe(x)


# 1. Safe L2 Normalization
def normalize(x, mask=None, eps=1e-8):
    """
    Safe normalization that avoids NaN gradients for zero vectors/masked positions.
    Crucial for JAX static shapes where padding vectors might be zeros.
    """
    # x shape: [..., H]
    # mask shape: [..., 1]

    # Handle NaN/Inf in input just in case
    x = jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x.astype(jnp.float32)

    if mask is not None:
        # Create a safe vector (ones) for positions that are masked out (Padding).
        # This prevents division by zero (norm=0) in rsqrt.
        # We restore the zero-ness later by multiplying by mask at the end.
        x_safe = jnp.where(mask > 0, x, 1.0)

        sq_sum = jnp.sum(x_safe**2, axis=-1, keepdims=True)
        inv_norm = jax.lax.rsqrt(jnp.maximum(sq_sum, eps))

        # Result is normalized vector where valid, 0.0 where masked
        # This ensures padding vectors are explicitly zeroed out and don't affect dot products.
        return (x_safe * inv_norm) * mask
    else:
        sq_sum = jnp.sum(x**2, axis=-1, keepdims=True)
        inv_norm = jax.lax.rsqrt(jnp.maximum(sq_sum, eps))
        return x * inv_norm


def _compute_metrics(
    logits, pos_sim, all_neg_sim, all_neg_mask_flat, sample_valid_flat, valid_count
):
    """
    Compute debugging metrics ignoring padding.
    """
    # Accuracy: position 0 is the positive sample
    pred_ids = jnp.argmax(logits, axis=-1)
    # Mask out padding samples from accuracy calculation
    accuracy = (
        jnp.sum((pred_ids == 0).astype(jnp.float32) * sample_valid_flat) / valid_count
    )

    # Positive Similarity Stats
    # pos_sim shape: [N, 1]
    pos_sim_val = pos_sim.squeeze(-1) * sample_valid_flat
    pos_sim_mean = jnp.sum(pos_sim_val) / valid_count

    # Negative Similarity Stats
    # all_neg_sim shape: [N, Total_Negs]
    # We only care about valid anchors vs valid negatives
    # Valid interactions = (Anchor is valid) AND (Negative is valid)

    # Broadcast anchor mask to interaction matrix: [N, 1] * [1, Total_Negs] -> [N, Total_Negs]
    valid_interactions_mask = sample_valid_flat[:, None] * all_neg_mask_flat[None, :]
    total_valid_interactions = jnp.sum(valid_interactions_mask) + 1e-8

    neg_sim_val = all_neg_sim * valid_interactions_mask
    neg_sim_mean = jnp.sum(neg_sim_val) / total_valid_interactions

    return {
        "accuracy": accuracy,
        "valid_samples": valid_count,
        "pos_sim_mean": pos_sim_mean,
        "neg_sim_mean": neg_sim_mean,
    }


def info_nce_loss(
    anchor_emb: jnp.ndarray,  # [B, S, H] (S is static MaxSamples)
    positive_emb: jnp.ndarray,  # [B, S, H]
    negatives_emb: jnp.ndarray,  # [B, S, Num_Negs, H]
    anchor_mask: jnp.ndarray,  # [B, S] (1 for valid, 0 for padding)
    pos_mask: jnp.ndarray,  # [B, S]
    negs_mask: jnp.ndarray,  # [B, S, Num_Negs]
    temperature: float = 0.07,
    return_metrics: bool = False,
) -> Tuple[jnp.ndarray, Dict]:
    """
    InfoNCE Loss with in-batch negatives and Static Shape Support (JAX).

    JAX Static Shape Handling:
    - 'S' is the static dimension (Max Packed Samples).
    - Masks (anchor_mask, pos_mask) determine actual valid data.
    - Masks input shape: Can be [B, S] or [B, S, 1]. The function normalizes them to handle both.
    - Code logic explicitly handles padding:
        1. Padding vectors are normalized to exact Zeros.
        2. Padding negatives are masked in logits (-1e9) so valid anchors ignore them.
        3. Loss is only averaged over valid anchors (sample_valid_flat).

    Logic:
    1. Flatten Batch (B) and MaxSamples (S) dimensions into N = B*S.
    2. Compute similarity between Anchor[i] and Positive[i].
    3. Compute similarity between Anchor[i] and ALL Negatives flattened (Num_Negs * N).

    Args:
        anchor_emb: [B, S, H].
        positive_emb: [B, S, H].
        negatives_emb: [B, S, K, H]. K is Num_Negs per anchor.
        anchor_mask: [B, S] or [B, S, 1].
        pos_mask: [B, S] or [B, S, 1].
        negs_mask: [B, S, K] or [B, S, K, 1]. Must be 0 for padding negatives.
        temperature: Softmax temperature.
        return_metrics: If True, returns a dictionary of metrics.

    Returns:
        loss: Scalar loss value.
        metrics: Dictionary of metrics if requested.
    """
    # --- DEBUG Backward Gradients ---
    # anchor_emb = inspect_grad(anchor_emb, "anchor_emb")
    # positive_emb = inspect_grad(positive_emb, "anchor_emb")
    # negatives_emb = inspect_grad(negatives_emb, "anchor_emb")

    # --- 1. Shape Standardization (Flattening B and S) ---
    # Ensure inputs are at least 3D [B, S, ...] for uniform handling
    if anchor_emb.ndim == 2:  # [S, H]
        anchor_emb = anchor_emb[None, ...]
        positive_emb = positive_emb[None, ...]
        negatives_emb = negatives_emb[None, ...]
        anchor_mask = anchor_mask[None, ...]
        pos_mask = pos_mask[None, ...]
        negs_mask = negs_mask[None, ...]

    B, S, H = anchor_emb.shape  # S is static MaxSamples
    K = negatives_emb.shape[2]  # Num Negatives per sample
    N = B * S  # Total static capacity (Max Total Anchors)

    # Flatten anchors/positives to [N, H]
    anchor_flat = anchor_emb.reshape(N, H)
    pos_flat = positive_emb.reshape(N, H)

    # Flatten masks to [N, 1]
    # This robustly handles both [B, S] and [B, S, 1] inputs
    anchor_mask_flat = anchor_mask.reshape(N, 1)
    pos_mask_flat = pos_mask.reshape(N, 1)

    # Flatten negatives to [Total_Negs_Pool, H] -> [N*K, H]
    negs_flat = negatives_emb.reshape(N * K, H)
    # Flatten neg masks to [Total_Negs_Pool, 1] -> [N*K] (squeeze last dim for convenience later)
    # This robustly handles both [B, S, K] and [B, S, K, 1] inputs
    negs_mask_flat = negs_mask.reshape(N * K)

    # --- 2. Determine Valid Samples ---
    # A sample is valid for loss calculation only if both Anchor and Positive are valid.
    # Padding samples (where mask=0) will result in 0 here.
    sample_valid_flat = (anchor_mask_flat.squeeze(-1) > 0) & (
        pos_mask_flat.squeeze(-1) > 0
    )
    sample_valid_flat = sample_valid_flat.astype(jnp.float32)  # [N]
    valid_count = jnp.sum(sample_valid_flat) + 1e-8

    # --- 3. Normalize ---
    # Apply normalization. Critical: Padding vectors become exact 0 vectors.
    anchor_norm = normalize(anchor_flat, anchor_mask_flat)  # [N, H]
    positive_norm = normalize(pos_flat, pos_mask_flat)  # [N, H]

    # For negatives, we need the mask in shape [N*K, 1] for normalize
    negs_norm = normalize(negs_flat, negs_mask_flat[:, None])  # [N*K, H]

    # --- 4. Compute Similarities ---

    # A. Positive Similarity: Dot product of corresponding pairs
    # [N, H] * [N, H] -> [N, 1]
    # For padding samples, this dot product is 0.0 because vectors are 0.0
    pos_sim = jnp.sum(anchor_norm * positive_norm, axis=-1, keepdims=True)

    # B. Negative Similarity: All Anchors vs All Negatives in the batch
    # [N, H] @ [H, N*K] -> [N, N*K]
    # For padding anchors (row i), the whole row is 0.0.
    # For padding negatives (col j), the whole col is 0.0.
    all_neg_sim = jnp.matmul(anchor_norm, negs_norm.T)

    # --- 5. Masking and Logits ---

    # Masking Invalid Negatives (Padding in the negative pool):
    # If a negative sample is padding, no anchor should consider it a negative.
    # negs_mask_flat is [N*K]. Broadcast to [1, N*K].
    neg_validity_mask = negs_mask_flat[None, :]

    # Replace similarity 0.0 (from padding) with -1e9 to remove from Softmax
    all_neg_sim = jnp.where(neg_validity_mask > 0, all_neg_sim, -1e9)

    # Concatenate: [Positive_Sim, All_Negative_Sims] -> [N, 1 + N*K]
    # Index 0 is always the positive class
    logits = jnp.concatenate([pos_sim, all_neg_sim], axis=-1)

    # Apply Temperature
    logits = logits / temperature

    # --- 6. Compute Loss ---
    # Target is always index 0 (the positive pair)
    log_probs = jax.nn.log_softmax(logits, axis=-1)

    # Gather log_prob of the positive class (index 0)
    pos_log_prob = log_probs[..., 0]  # [N]

    # Apply Mask to Loss:
    # 1. Padding Anchors: sample_valid_flat is 0 -> adds 0.0 to loss sum.
    # 2. Valid Anchors: sample_valid_flat is 1 -> adds correct loss.
    loss = jnp.sum(-pos_log_prob * sample_valid_flat) / valid_count

    # --- Debug Forward ---
    # jax.debug.print("loss: {}", loss)

    if return_metrics:
        metrics = _compute_metrics(
            logits, pos_sim, all_neg_sim, negs_mask_flat, sample_valid_flat, valid_count
        )
        return loss, metrics
    else:
        return loss, {}
