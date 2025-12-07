"""
InfoNCE loss function for contrastive learning
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Dict


def info_nce_loss(
    anchor_emb: jnp.ndarray,  # [B, S, H]
    positive_emb: jnp.ndarray,  # [B, S, H]
    negatives_emb: jnp.ndarray,  # [B, S, Num_Negs, H]
    anchor_mask: jnp.ndarray,  # [B, S]
    pos_mask: jnp.ndarray,  # [B, S]
    negs_mask: jnp.ndarray,  # [B, S, Num_Negs]
    temperature: float = 0.07,
    return_metrics: bool = False,
) -> Tuple[jnp.ndarray, Dict]:
    """
    Pure InfoNCE Loss supporting multiple negatives.
    No slicing inside. Expects pre-split embeddings.
    """
    # 1. 确定有效样本 (Anchor 和 Positive 必须同时存在)
    # [B, S]
    sample_valid_mask = anchor_mask & pos_mask
    valid_count = jnp.sum(sample_valid_mask) + 1e-8

    # 2. L2 Normalization (Zero-copy safe)
    def normalize(x):
        return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)

    anchor_norm = normalize(anchor_emb)  # [B, S, H]
    positive_norm = normalize(positive_emb)  # [B, S, H]
    negatives_norm = normalize(negatives_emb)  # [B, S, N, H]

    # 3. 计算相似度
    # Positive: (B, S, H) * (B, S, H) -> (B, S) -> unsqueeze -> (B, S, 1)
    pos_sim = jnp.sum(anchor_norm * positive_norm, axis=-1, keepdims=True)

    # Negatives: (B, S, 1, H) * (B, S, N, H) -> (B, S, N)
    # Broadcast anchor against all negatives
    neg_sim = jnp.sum(anchor_norm[..., None, :] * negatives_norm, axis=-1)

    # 4. 构建 Logits [B, S, 1 + N]
    # index 0 is Positive, indices 1..N are Negatives
    logits = jnp.concatenate([pos_sim, neg_sim], axis=-1)
    logits = logits / temperature

    # 5. Masking Logits (关键步骤)
    # 我们不仅要屏蔽掉 Sample 级无效的，还要屏蔽掉 Negatives 中具体的空槽位
    # Logit Mask shape: [B, S, 1 + N]
    # 拼接: [Pos_Mask(B,S,1), Negs_Mask(B,S,N)]
    combined_mask = jnp.concatenate([pos_mask[..., None], negs_mask], axis=-1)

    # 将无效槽位的 Logit 设为极小值 (-inf)，使其在 Softmax 中概率为 0
    large_neg = -1e9  # jnp.finfo(logits.dtype).min 可能会溢出，-1e9 足够安全
    logits = jnp.where(combined_mask > 0, logits, large_neg)

    # 6. 计算 Cross Entropy
    # 目标 Target 永远是 0 (Positive 永远在第 0 位)
    # LogSoftmax: [B, S, 1 + N]
    log_probs = jax.nn.log_softmax(logits, axis=-1)

    # 取出 Positive 对应的 log_prob
    pos_log_prob = log_probs[..., 0]  # [B, S]

    # Loss = -log(P(Positive))
    per_sample_loss = -pos_log_prob

    # 7. 应用 Sample 级 Mask
    masked_loss = per_sample_loss * sample_valid_mask
    loss = jnp.sum(masked_loss) / valid_count

    if return_metrics:
        # 8. Metrics
        # 计算准确率: Logits 最大的位置是否是 0 (Positive)
        pred_ids = jnp.argmax(logits, axis=-1)
        correct = (pred_ids == 0).astype(jnp.float32) * sample_valid_mask
        accuracy = jnp.sum(correct) / valid_count

        # 计算正样本相似度统计 (只考虑有效样本)
        pos_sim_flat = pos_sim.flatten()  # [B*S]
        sample_valid_mask_flat = sample_valid_mask.flatten()  # [B*S]
        pos_sim_valid = pos_sim_flat * sample_valid_mask_flat
        pos_sim_mean = jnp.sum(pos_sim_valid) / valid_count

        # 计算正样本相似度标准差
        pos_sim_squared_diff = (
            pos_sim_valid - pos_sim_mean
        ) ** 2 * sample_valid_mask_flat
        pos_sim_variance = jnp.sum(pos_sim_squared_diff) / valid_count
        pos_sim_std = jnp.sqrt(pos_sim_variance + 1e-8)

        # 计算负样本相似度统计 (考虑有效样本和负样本mask)
        # neg_sim shape: [B, S, N], negs_mask shape: [B, S, N]
        # 只计算有效的负样本 (sample_valid_mask 和 negs_mask 同时有效)
        neg_valid_mask = sample_valid_mask[..., None] * negs_mask  # [B, S, N]
        neg_valid_count = jnp.sum(neg_valid_mask) + 1e-8

        neg_sim_valid = neg_sim * neg_valid_mask
        neg_sim_mean = jnp.sum(neg_sim_valid) / neg_valid_count

        # 计算负样本相似度标准差
        neg_sim_squared_diff = (neg_sim_valid - neg_sim_mean) ** 2 * neg_valid_mask
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

    if return_metrics:
        return loss, metrics
    else:
        return loss
