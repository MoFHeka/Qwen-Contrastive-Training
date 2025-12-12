"""
InfoNCE loss function for contrastive learning
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Dict


def info_nce_loss(
    anchor_emb: jnp.ndarray,  # [B, PackedSamples, H]
    positive_emb: jnp.ndarray,  # [B, PackedSamples, H]
    negatives_emb: jnp.ndarray,  # [B, PackedSamples, Num_Negs, H]
    anchor_mask: jnp.ndarray,  # [B, PackedSamples]
    pos_mask: jnp.ndarray,  # [B, PackedSamples]
    negs_mask: jnp.ndarray,  # [B, PackedSamples, Num_Negs]
    temperature: float = 0.07,
    return_metrics: bool = False,
) -> Tuple[jnp.ndarray, Dict]:
    """
    InfoNCE Loss with in-batch negatives.
    Masked positions (mask=0) do not participate in forward and backward pass.
    Uses all batch negatives: each anchor's negatives include its own negatives
    plus all other samples' positives in the batch (as in-batch negatives).

    Args:
        anchor_emb: Anchor embeddings [B, PackedSamples, H]
        positive_emb: Positive embeddings [B, PackedSamples, H]
        negatives_emb: Negative embeddings [B, PackedSamples, Num_Negs, H]
        anchor_mask: Anchor mask, 1 for valid, 0 for invalid [B, PackedSamples]
        pos_mask: Positive mask, 1 for valid, 0 for invalid [B, PackedSamples]
        negs_mask: Negative mask, 1 for valid, 0 for invalid [B, PackedSamples, Num_Negs]
        temperature: Temperature parameter for softmax
        return_metrics: Whether to return additional metrics

    Returns:
        Tuple of (loss, metrics_dict)
    """
    # 1. Determine valid samples: both anchor and positive must be valid (mask > 0)
    # [B, PackedSamples] - boolean mask
    sample_valid_mask = (anchor_mask > 0) & (pos_mask > 0)
    # Convert to float32 for computation, sum to get valid count
    sample_valid_mask_f32 = sample_valid_mask.astype(jnp.float32)
    valid_count = jnp.sum(sample_valid_mask_f32) + 1e-8

    # 2. L2 Normalization
    def normalize(x, eps=1e-8):
        """
        Normalize embeddings.
        """
        # Convert to float32 for numerical stability
        x = x.astype(jnp.float32)

        # Handle NaN and Inf values: replace with 0.0
        x = jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # Compute norm along last dimension
        norm = jnp.linalg.norm(x, axis=-1, keepdims=True)

        # Normalize
        normalized = x / (norm + eps)

        return normalized

    # Normalize embeddings
    anchor_norm = normalize(anchor_emb)  # [B, PackedSamples, H]
    positive_norm = normalize(positive_emb)  # [B, PackedSamples, H]
    negatives_norm = normalize(negatives_emb)  # [B, PackedSamples, Num_Negs, H]

    # 3. Compute positive similarity
    # [B, PackedSamples, H] * [B, PackedSamples, H] -> [B, PackedSamples]
    pos_sim = jnp.sum(
        anchor_norm * positive_norm, axis=-1, keepdims=True
    )  # [B, PackedSamples, 1]

    # Apply sample_valid_mask: invalid samples have zero similarity
    # This ensures they don't contribute to loss or gradients
    pos_sim = pos_sim * sample_valid_mask_f32[..., None]

    # 4. Compute negative similarities
    # 4.1. Own negatives: [B, PackedSamples, Num_Negs]
    # [B, PackedSamples, 1, H] * [B, PackedSamples, Num_Negs, H] -> [B, PackedSamples, Num_Negs]
    own_neg_sim = jnp.sum(anchor_norm[..., None, :] * negatives_norm, axis=-1)
    # Apply masks: both sample_valid_mask and negs_mask
    # Only valid samples and valid negatives contribute
    own_neg_sim = (
        own_neg_sim * sample_valid_mask_f32[..., None] * negs_mask.astype(jnp.float32)
    )

    # 4.2. In-batch negatives: use all other samples' positives as negatives
    # Reshape for batch-wise computation: [B, PackedSamples, H] -> [B * PackedSamples, H]
    B, S, H = anchor_norm.shape
    anchor_flat = anchor_norm.reshape(B * S, H)  # [B*S, H]
    positive_flat = positive_norm.reshape(B * S, H)  # [B*S, H]
    sample_valid_flat = sample_valid_mask_f32.reshape(B * S)  # [B*S]
    pos_mask_flat = pos_mask.reshape(B * S).astype(jnp.float32)  # [B*S]

    # Compute similarity matrix: [B*S, H] @ [H, B*S] -> [B*S, B*S]
    # Each row i: similarity between anchor[i] and all positives
    in_batch_neg_sim = jnp.dot(anchor_flat, positive_flat.T)  # [B*S, B*S]

    # Create mask to exclude self (diagonal) and invalid samples
    # Self-exclusion mask: [B*S, B*S], diagonal is 0 (exclude self)
    self_mask = 1.0 - jnp.eye(B * S, dtype=jnp.float32)
    # Valid negative mask: [B*S, B*S]
    # Only valid anchors (sample_valid_flat) can have negatives
    # Only valid positives (pos_mask_flat) can be used as negatives
    valid_neg_mask = sample_valid_flat[:, None] * pos_mask_flat[None, :]
    # Combined mask: exclude self and invalid samples
    in_batch_mask = self_mask * valid_neg_mask
    # Apply mask: invalid positions become zero (no gradient)
    in_batch_neg_sim = in_batch_neg_sim * in_batch_mask

    # Reshape back: [B*S, B*S] -> [B, S, B*S]
    in_batch_neg_sim = in_batch_neg_sim.reshape(B, S, B * S)
    in_batch_mask_reshaped = in_batch_mask.reshape(B, S, B * S)

    # 5. Concatenate all negatives: own negatives + in-batch negatives
    # own_neg_sim: [B, S, Num_Negs], in_batch_neg_sim: [B, S, B*S]
    all_neg_sim = jnp.concatenate(
        [own_neg_sim, in_batch_neg_sim], axis=-1
    )  # [B, S, Num_Negs + B*S]

    # Create combined mask for all negatives
    # own_neg_mask: [B, S, Num_Negs], in_batch_mask: [B, S, B*S]
    all_neg_mask = jnp.concatenate(
        [negs_mask.astype(jnp.float32), in_batch_mask_reshaped], axis=-1
    )  # [B, S, Num_Negs + B*S]
    # Also apply sample_valid_mask: invalid samples don't have valid negatives
    all_neg_mask = all_neg_mask * sample_valid_mask_f32[..., None]

    # 6. Build logits: [B, PackedSamples, 1 + Num_Negs + B*S]
    # Index 0 is positive, indices 1..(Num_Negs+B*S) are negatives
    logits = jnp.concatenate(
        [pos_sim, all_neg_sim], axis=-1
    )  # [B, S, 1 + Num_Negs + B*S]
    logits = logits / temperature

    # 7. Apply mask to logits (set invalid positions to -inf)
    # This ensures masked positions don't contribute to softmax
    combined_mask = jnp.concatenate(
        [sample_valid_mask_f32[..., None], all_neg_mask], axis=-1
    )  # [B, S, 1 + Num_Negs + B*S]
    large_neg = -1e9  # Large negative value, safe for softmax
    logits = jnp.where(combined_mask > 0, logits, large_neg)

    # 8. Compute Cross Entropy Loss
    # Target is always 0 (positive is at index 0)
    log_probs = jax.nn.log_softmax(logits, axis=-1)  # [B, S, 1 + Num_Negs + B*S]
    pos_log_prob = log_probs[..., 0]  # [B, S]

    # Loss = -log(P(Positive))
    per_sample_loss = -pos_log_prob

    # Apply sample_valid_mask: invalid samples have zero loss and no gradient
    # This is critical: masked positions don't contribute to loss or gradients
    masked_loss = per_sample_loss * sample_valid_mask_f32
    loss = jnp.sum(masked_loss) / valid_count

    if return_metrics:
        # 9. Compute metrics
        # Accuracy: whether the max logit is at index 0 (positive)
        pred_ids = jnp.argmax(logits, axis=-1)  # [B, S]
        correct = (pred_ids == 0).astype(jnp.float32) * sample_valid_mask_f32
        accuracy = jnp.sum(correct) / valid_count

        # Positive similarity statistics (only valid samples)
        pos_sim_flat = pos_sim.flatten()  # [B*S]
        sample_valid_flat_metrics = sample_valid_mask_f32.flatten()  # [B*S]
        pos_sim_valid = pos_sim_flat * sample_valid_flat_metrics
        pos_sim_mean = jnp.sum(pos_sim_valid) / valid_count

        pos_sim_squared_diff = (
            pos_sim_valid - pos_sim_mean
        ) ** 2 * sample_valid_flat_metrics
        pos_sim_variance = jnp.sum(pos_sim_squared_diff) / valid_count
        pos_sim_std = jnp.sqrt(pos_sim_variance + 1e-8)

        # Negative similarity statistics (own negatives + in-batch negatives)
        # Only count valid negatives
        neg_valid_mask_combined = all_neg_mask  # [B, S, Num_Negs + B*S]
        neg_valid_count = jnp.sum(neg_valid_mask_combined) + 1e-8

        neg_sim_valid = all_neg_sim * neg_valid_mask_combined
        neg_sim_mean = jnp.sum(neg_sim_valid) / neg_valid_count

        neg_sim_squared_diff = (
            neg_sim_valid - neg_sim_mean
        ) ** 2 * neg_valid_mask_combined
        neg_sim_variance = jnp.sum(neg_sim_squared_diff) / neg_valid_count
        neg_sim_std = jnp.sqrt(neg_sim_variance + 1e-8)

        metrics = {
            "accuracy": accuracy,
            "valid_samples": valid_count,
            "pos_sim_mean": pos_sim_mean,
            "neg_sim_mean": neg_sim_mean,
            "pos_sim_std": pos_sim_std,
            "neg_sim_std": neg_sim_std,
        }

        return loss, metrics

    return loss, {}
