"""
InfoNCE loss function for contrastive learning
"""

from numpy.ma import mask_cols
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


# Safe L2 Normalization
def normalize(x, mask=None, eps=1e-8):
    """
    Safe normalization that avoids NaN gradients for zero vectors/masked positions.
    Crucial for JAX static shapes where padding vectors might be zeros.
    """
    # x shape: [..., H]
    # mask shape: [..., H]
    if mask is not None:
        chex.assert_equal_rank((x, mask))

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

    Note: The mask logic here is consistent with the combined_mask in info_nce_loss:
    - For positive: only anchor_validity_mask matters
    - For negatives: both anchor_validity_mask AND neg_validity_mask must be valid
    """
    # Accuracy: position 0 is the positive sample
    # logits are already masked (padding positions set to -1e9), but we still use
    # sample_valid_flat to ensure padding anchors are excluded from accuracy calculation
    pred_ids = jnp.argmax(logits, axis=-1)
    accuracy = (
        jnp.sum((pred_ids == 0).astype(jnp.float32) * sample_valid_flat) / valid_count
    )

    # Positive Similarity Stats
    # pos_sim shape: [N, 1]
    # Only consider valid anchors (padding anchors masked out)
    pos_sim_val = pos_sim.squeeze(-1) * sample_valid_flat
    pos_sim_mean = jnp.sum(pos_sim_val) / valid_count

    # Negative Similarity Stats
    # all_neg_sim shape: [N, N*K] - all anchors vs all negatives
    # We only care about valid anchors vs valid negatives
    # Valid interactions = (Anchor is valid) AND (Negative is valid)
    # This matches the combined_mask logic: anchor_validity_mask * neg_validity_mask
    valid_interactions_mask = (
        sample_valid_flat[:, None] * all_neg_mask_flat[None, :]
    )  # [N, N*K]
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
    sample_mask: jnp.ndarray,  # [B, S] (1 for valid, 0 for padding)
    temperature: float = 1,
    return_metrics: bool = False,
) -> Tuple[jnp.ndarray, Dict]:
    """
    InfoNCE Loss with in-batch negatives and Static Shape Support (JAX).

    JAX Static Shape Handling:
    - 'S' is the static dimension (Max Packed Samples).
    - Masks (sample_mask) determine actual valid data.
    - Masks input shape: Can be [B, S]. The function normalizes them to handle both.
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
        sample_mask: [B, S]. Must be 0 for padding samples.
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
        sample_mask = sample_mask[None, ...]

    B, S, H = anchor_emb.shape  # S is static MaxSamples
    K = negatives_emb.shape[2]  # Num Negatives per sample
    N = B * S  # Total static capacity (Max Total Anchors)

    # Flatten anchors/positives to [N, H]
    anchor_flat = anchor_emb.reshape(N, H)
    pos_flat = positive_emb.reshape(N, H)

    # Flatten sample_mask to [N, 1]
    # sample_mask: [B, S] -> [N, 1] (1 for valid, 0 for padding)
    # This robustly handles both [B, S] and [B, S, 1] inputs
    sample_mask_flat = sample_mask.reshape(N, 1)

    # Anchor and positive use the same mask (they correspond to the same samples)
    anchor_mask_flat = sample_mask_flat  # [N, 1]
    pos_mask_flat = sample_mask_flat  # [N, 1]

    # Flatten negatives to [Total_Negs_Pool, H] -> [N*K, H]
    negs_flat = negatives_emb.reshape(N * K, H)
    # For negatives: each sample has K negatives, so we expand the mask
    # sample_mask: [B, S] -> [N] -> [N, K] -> [N*K]
    # Each negative shares the same validity as its corresponding anchor sample
    # Use jnp.tile to repeat each mask value K times
    negs_mask_flat = jnp.tile(sample_mask.reshape(N)[:, None], (1, K)).reshape(N * K)

    # --- 2. Determine Valid Samples ---
    # A sample is valid for loss calculation only if both Anchor and Positive are valid.
    # Padding samples (where mask=0) will result in 0 here.
    sample_valid_flat = (anchor_mask_flat.squeeze(-1) > 0) & (
        pos_mask_flat.squeeze(-1) > 0
    )
    sample_valid_flat = sample_valid_flat.astype(jnp.float32)  # [N]
    valid_count = jnp.sum(sample_valid_flat) + 1e-8

    # --- 3. Normalize ---
    # # Apply normalization. Critical: Padding vectors become exact 0 vectors.
    # anchor_norm = normalize(anchor_flat, anchor_mask_flat)  # [N, H]
    # positive_norm = normalize(pos_flat, pos_mask_flat)  # [N, H]

    # # For negatives, we need the mask in shape [N*K, 1] for normalize
    # negs_norm = normalize(negs_flat, negs_mask_flat)  # [N*K, H]

    # Should Normalization outside the loss function
    anchor_norm = anchor_flat
    positive_norm = pos_flat
    negs_norm = negs_flat

    # --- 4. Compute Similarities ---
    # # Don't let the model learn to simply lengthen the modulus of the vector
    # scale_factor = 1.0 / jnp.sqrt(H)

    # A. Positive Similarity: Dot product of corresponding pairs
    # [N, H] * [N, H] -> [N, 1]
    pos_sim = jnp.sum(anchor_norm * positive_norm, axis=-1, keepdims=True)

    # B. Negative Similarity: All Anchors vs All Negatives in the batch
    # [N, H] @ [H, N*K] -> [N, N*K]
    all_neg_sim = jnp.matmul(anchor_norm, negs_norm.T)

    # --- 5. Masking and Logits ---

    # Concatenate: [Positive_Sim, All_Negative_Sims] -> [N, 1 + N*K]
    # Index 0 is always the positive class
    logits = jnp.concatenate([pos_sim, all_neg_sim], axis=-1)

    # Combined masking: mask both padding anchors and padding negatives in one operation
    # anchor_validity_mask: [N, 1] - masks entire row for padding anchors
    # neg_validity_mask: [1, N*K] - masks padding negatives
    anchor_validity_mask = sample_valid_flat[:, None]  # [N, 1]
    neg_validity_mask = negs_mask_flat[None, :]  # [1, N*K]

    # Create combined mask: [N, 1 + N*K]
    # For positive (index 0): only anchor_validity_mask matters
    # For negatives (index 1:): both anchor_validity_mask AND neg_validity_mask must be valid
    combined_mask = jnp.concatenate(
        [
            anchor_validity_mask,  # [N, 1] for positive
            anchor_validity_mask * neg_validity_mask,  # [N, N*K] for negatives
        ],
        axis=-1,
    )

    # Apply combined mask: mask padding anchors (entire row) and padding negatives
    logits = jnp.where(combined_mask > 0, logits, -1e9)

    # Apply Temperature
    logits = logits / temperature

    # --- 6. Compute Loss ---
    # Target is always index 0 (the positive pair)
    log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)

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
