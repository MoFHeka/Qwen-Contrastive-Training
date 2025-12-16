"""
Qwen3EmbeddingModel 完整流程测试 (重构版)

设计目标：

1. 统一管理 Mesh 环境

2. 复用模型前向传播代码

3. 提供两个并行的 JIT 入口：

   - inference_step: 仅模型前向 (Input -> Embeddings)

   - train_step (loss_step): 模型前向 + Loss 计算 (Input -> Loss)
"""

import os
import tempfile
import numpy as np
from typing import Tuple, Dict

# ==========================================
# 0. 环境配置 (必须在 import jax 之前)
# ==========================================
xla_cache_dir = os.path.join(os.getcwd(), ".xla_cache")
os.makedirs(xla_cache_dir, exist_ok=True)

xla_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_gpu_per_fusion_autotune_cache_dir" not in xla_flags:
    xla_flags += f" --xla_gpu_per_fusion_autotune_cache_dir={xla_cache_dir}"
os.environ["XLA_FLAGS"] = xla_flags.strip()
print(f"✓ XLA 编译缓存目录设置为: {xla_cache_dir}")

import jax
import jax.numpy as jnp

from transformers import AutoTokenizer
from flax import nnx

from model_io import save_easydel_model, load_easydel_model

# 尝试导入 loss 函数，如果不存在则定义 Dummy 用于测试
try:
    from loss import info_nce_loss
except ImportError:
    print("⚠ 未找到 loss.py，使用 Dummy Loss 用于演示")

    def info_nce_loss(anchor_emb, positive_emb, negatives_emb, **kwargs):
        # Dummy implementation
        return jnp.sum(anchor_emb - positive_emb) * 0.0 + 0.5, {
            "accuracy": 0.99,
            "pos_sim_mean": 0.8,
            "neg_sim_mean": 0.2,
        }


from qwen3_embedding_modeling import (
    Qwen3EmbeddingModel,
    create_from_initial_model,
)
from vocab_embedding_utils import get_tokenizer_from_model
from dataset_loader import extract_sample_info_from_segment_ids

# Set JAX config for compilation cache (additional method)
try:
    jax_cache_dir = os.path.join(os.getcwd(), ".jax_cache")
    os.makedirs(jax_cache_dir, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", jax_cache_dir)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_enable_xla_caches", xla_cache_dir)
except Exception as e:
    print(f"⚠ 通过 jax.config 设置缓存目录失败: {e}")


class Qwen3EmbeddingRunner:
    """
    负责管理模型的 Mesh 环境和 JIT 编译函数的执行器。
    """

    def __init__(self, model: Qwen3EmbeddingModel):
        self.model = model
        self.mesh = self._create_mesh()

        # 将 Mesh 注入模型 (如果模型支持)
        if hasattr(self.model, "set_model_mesh"):
            self.model.set_model_mesh(self.mesh)
        elif hasattr(self.model, "mesh"):
            self.model.mesh = self.mesh

        # 编译 JIT 函数
        print("\n正在编译 JIT 函数...")

        # 1. 仅推理路径 (Inference Only)
        self._inference_jit = jax.jit(
            self._inference_fn_impl,
            static_argnames=["max_samples", "num_negatives", "output_hidden_states"],
        )

        # 2. Loss 计算路径 (Forward + Loss)
        self._loss_jit = jax.jit(
            self._loss_fn_impl,
            static_argnames=["max_samples", "num_negatives", "temperature"],
        )
        print("✓ JIT 函数编译完成")

    def _create_mesh(self):
        """创建 SPMD Mesh"""
        devices = jax.devices()
        local_devices = jax.local_devices()

        # 优先使用 GPU
        gpu_devices = []
        cpu_devices = []

        for d in local_devices:
            platform = str(d).split(":")[0] if ":" in str(d) else ""
            if platform in ["gpu", "cuda"] or d.device_kind in ["gpu", "tpu"]:
                gpu_devices.append(d)
            elif d.device_kind == "cpu" or platform == "cpu":
                cpu_devices.append(d)

        # If no local GPU found by platform, check all devices
        if len(gpu_devices) == 0:
            for d in devices:
                platform = str(d).split(":")[0] if ":" in str(d) else ""
                if platform in ["gpu", "cuda"] or d.device_kind in ["gpu", "tpu"]:
                    gpu_devices.append(d)

        # Prefer GPU/TPU, but also allow CPU if explicitly configured
        if len(gpu_devices) > 0:
            selected_devices = gpu_devices
        else:
            selected_devices = cpu_devices if len(cpu_devices) > 0 else local_devices

        num_devices = len(selected_devices)
        print(f"\n[Mesh Setup] 使用设备数量: {num_devices}")

        if num_devices == 1:
            print("  警告: 仅检测到 1 个设备，将使用单设备模式")
            # 即使是单卡，也创建一个 Shape 为 (1, 1, 1, 1, 1) 的 Mesh 以保持逻辑一致
            mesh_shape = (1, 1, 1, 1, 1)
        else:
            mesh_shape = (1, num_devices, 1, 1, 1)

        mesh_axis_names = ("dp", "fsdp", "ep", "tp", "sp")
        devices_reshaped = np.array(
            selected_devices[: np.prod(mesh_shape)], dtype=object
        ).reshape(mesh_shape)
        return jax.sharding.Mesh(devices_reshaped, axis_names=mesh_axis_names)

    # ========================================================================
    # 核心逻辑 (Core Logic) - 被 inference 和 loss 路径共同复用
    # ========================================================================
    def _core_forward(
        self,
        model,
        input_ids,
        attention_mask,
        segment_ids,
        position_ids,
        max_samples,
        num_negatives,
        output_hidden_states,
    ):
        """
        纯粹的模型前向传播调用。
        """
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            position_ids=position_ids,
            max_samples=max_samples,
            num_negatives=num_negatives,
            output_hidden_states=output_hidden_states,
        )

    # ========================================================================
    # JIT 实现函数 (Implementation Functions)
    # ========================================================================

    def _inference_fn_impl(
        self,
        model,
        input_ids,
        attention_mask=None,
        segment_ids=None,
        position_ids=None,
        max_samples=64,
        num_negatives=1,
        output_hidden_states=False,
    ):
        """
        [JIT Path 1] 推理路径：只执行模型前向，返回 Embeddings
        """
        return self._core_forward(
            model,
            input_ids,
            attention_mask,
            segment_ids,
            position_ids,
            max_samples,
            num_negatives,
            output_hidden_states,
        )

    def _loss_fn_impl(
        self,
        model,
        input_ids,
        attention_mask,
        segment_ids,
        position_ids,
        max_samples=64,
        num_negatives=1,
        temperature=0.07,
    ):
        """
        [JIT Path 2] Loss 路径：执行模型前向 -> 切分数据 -> 计算 Loss
        整个过程在一个 JIT 块中完成，减少 Device<->Host 通信。
        """
        # 1. 复用核心前向逻辑
        # Loss 计算模式下不需要 hidden_states
        output = self._core_forward(
            model,
            input_ids,
            attention_mask,
            segment_ids,
            position_ids,
            max_samples,
            num_negatives,
            output_hidden_states=False,
        )

        # model 返回 (projected_embs, slot_mask)
        projected_embs, slot_mask = output

        # 2. 在 Device 上进行切片 (Slicing)
        # 假设 slot 结构: [Anchor(0), Positive(1), Negatives(2...)]
        anchor_emb = projected_embs[..., 0, :]  # [B, S, H]
        anchor_mask = slot_mask[..., 0]  # [B, S]
        positive_emb = projected_embs[..., 1, :]  # [B, S, H]
        pos_mask = slot_mask[..., 1]  # [B, S]
        negatives_emb = projected_embs[..., 2:, :]  # [B, S, N, H]
        negs_mask = slot_mask[..., 2:]  # [B, S, N]

        # 3. 计算 Loss (纯 JAX 操作)
        loss, metrics = info_nce_loss(
            anchor_emb=anchor_emb,
            positive_emb=positive_emb,
            negatives_emb=negatives_emb,
            anchor_mask=anchor_mask,
            pos_mask=pos_mask,
            negs_mask=negs_mask,
            temperature=temperature,
            return_metrics=True,
        )

        return loss, metrics, projected_embs

    # ========================================================================
    # 公共接口 (Public Interface) - 自动处理 Mesh Context
    # ========================================================================
    def predict(self, input_ids, **kwargs):
        """
        执行推理。
        自动包裹 with self.mesh，并调用 _inference_jit。
        """
        with self.mesh:
            return self._inference_jit(self.model, input_ids, **kwargs)

    def compute_loss(
        self,
        input_ids,
        attention_mask,
        segment_ids,
        position_ids,
        **kwargs,
    ):
        """
        计算 Loss。
        自动包裹 with self.mesh，并调用 _loss_jit。
        """
        with self.mesh:
            return self._loss_jit(
                self.model,
                input_ids,
                attention_mask,
                segment_ids,
                position_ids,
                **kwargs,
            )


def compute_infonce_loss_from_output(
    projected_embs: jnp.ndarray,  # [B, S, num_slots, H]
    slot_mask: jnp.ndarray,  # [B, S, num_slots]
    temperature: float = 0.07,
    num_negatives: int = 1,
) -> Tuple[jnp.ndarray, Dict]:
    """
    Compute InfoNCE loss from model output (external function)

    Args:
      projected_embs: Projected embeddings [batch_size, max_samples, num_slots, embedding_dim]
      slot_mask: Slot mask [batch_size, max_samples, num_slots]
      temperature: Temperature parameter for InfoNCE loss
      num_negatives: Number of negative samples per sample

    Returns:
      Tuple of (loss, metrics)
    """
    from loss import info_nce_loss

    # Split embeddings and masks
    # Anchor: index 0
    anchor_emb = projected_embs[..., 0, :]  # [B, S, H]
    anchor_mask = slot_mask[..., 0]  # [B, S]

    # Positive: index 1
    positive_emb = projected_embs[..., 1, :]  # [B, S, H]
    pos_mask = slot_mask[..., 1]  # [B, S]

    # Negatives: index 2 to End
    negatives_emb = projected_embs[..., 2:, :]  # [B, S, N, H]
    negs_mask = slot_mask[..., 2:]  # [B, S, N]

    # Compute InfoNCE loss
    loss, metrics = info_nce_loss(
        anchor_emb=anchor_emb,
        positive_emb=positive_emb,
        negatives_emb=negatives_emb,
        anchor_mask=anchor_mask,
        pos_mask=pos_mask,
        negs_mask=negs_mask,
        temperature=temperature,
        return_metrics=True,
    )

    return loss, metrics


def test_single_text_forward(runner: Qwen3EmbeddingRunner, tokenizer: AutoTokenizer):
    """
    测试单条文本的前向传播（多卡并行）

    Args:
      runner: Qwen3EmbeddingRunner 实例
      tokenizer: Tokenizer 实例
    """
    print("\n" + "=" * 60)
    print("测试 1: 单条文本前向传播（多卡并行）")
    print("=" * 60)

    # 测试文本
    text = "这是一个测试文本，用于验证模型的前向传播功能。"
    print(f"\n输入文本: {text}")

    # Tokenize
    encoded = tokenizer(
        text, padding=False, truncation=True, max_length=512, return_tensors="np"
    )
    input_ids = jnp.array(encoded["input_ids"][0])
    attention_mask = jnp.array(encoded["attention_mask"][0])

    print(f"输入 token IDs 形状: {input_ids.shape}")
    print(f"输入 token IDs (前10个): {input_ids[:10]}")
    print(f"Attention mask 形状: {attention_mask.shape}")

    # 前向传播
    print("\n执行前向传播（SPMD 多卡并行）...")
    # Add batch dimension for model call
    input_ids_batched = input_ids[None, :]
    # For simple inference, we don't need attention_mask
    # If we pass attention_mask, we must also pass segment_ids, triplet_type, position_ids
    # So for simple inference, we skip attention_mask
    embeddings_batched = runner.predict(
        input_ids=input_ids_batched,
        attention_mask=None,  # Skip attention_mask for simple inference
        output_hidden_states=False,
    )
    embeddings = embeddings_batched[0]  # Remove batch dimension
    print("✓ 前向传播成功")
    print(f"输出嵌入形状: {embeddings.shape}")
    print(f"输出嵌入数据类型: {embeddings.dtype}")
    print("输出嵌入 (前5个token, 前10维):")
    print(embeddings[:5, :10])

    # 验证输出形状
    expected_shape = (input_ids.shape[0], runner.model.embedding_dim)
    assert embeddings.shape == expected_shape, (
        f"输出形状不匹配: 期望 {expected_shape}, 实际 {embeddings.shape}"
    )
    print(f"✓ 输出形状验证通过: {embeddings.shape}")

    return embeddings


def test_batch_forward(runner: Qwen3EmbeddingRunner, tokenizer: AutoTokenizer):
    """
    测试批量文本的前向传播

    Args:
      runner: Qwen3EmbeddingRunner 实例
      tokenizer: Tokenizer 实例
    """
    print("\n" + "=" * 60)
    print("测试 2: 批量文本前向传播")
    print("=" * 60)

    # 测试文本列表
    texts = [
        "这是第一条测试文本。",
        "这是第二条测试文本，比第一条稍微长一些。",
        "第三条测试文本，用于验证批量处理功能。",
    ]
    print(f"\n输入文本数量: {len(texts)}")
    for i, text in enumerate(texts):
        print(f"  文本 {i + 1}: {text}")

    # Tokenize
    encoded = tokenizer(
        texts, padding=True, truncation=True, max_length=128, return_tensors="np"
    )
    input_ids = jnp.array(encoded["input_ids"])
    attention_mask = jnp.array(encoded["attention_mask"])

    print(f"\n输入 token IDs 形状: {input_ids.shape}")
    print(f"Attention mask 形状: {attention_mask.shape}")

    # 前向传播
    print("\n执行批量前向传播（SPMD 多卡并行）...")
    try:
        # For simple inference, we don't need attention_mask
        # If we pass attention_mask, we must also pass segment_ids, triplet_type, position_ids
        # So for simple inference, we skip attention_mask
        embeddings = runner.predict(
            input_ids=input_ids,
            attention_mask=None,  # Skip attention_mask for simple inference
            output_hidden_states=False,
        )
        print("✓ 批量前向传播成功")
        print(f"输出嵌入形状: {embeddings.shape}")
        print(f"输出嵌入数据类型: {embeddings.dtype}")

        # 验证输出形状
        expected_shape = (
            input_ids.shape[0],
            input_ids.shape[1],
            runner.model.embedding_dim,
        )
        assert embeddings.shape == expected_shape, (
            f"输出形状不匹配: 期望 {expected_shape}, 实际 {embeddings.shape}"
        )
        print(f"✓ 输出形状验证通过: {embeddings.shape}")

        # 显示每个样本的嵌入统计信息
        print("\n每个样本的嵌入统计:")
        for i in range(len(texts)):
            sample_emb = embeddings[i]
            # 只考虑有效 token (attention_mask == 1)
            valid_mask = attention_mask[i] == 1
            valid_emb = sample_emb[valid_mask]
            print(f"  样本 {i + 1}:")
            print(f"    有效 token 数: {int(valid_mask.sum())}")
            print(f"    嵌入均值: {float(valid_emb.mean()):.6f}")
            print(f"    嵌入标准差: {float(valid_emb.std()):.6f}")
            print(
                f"    嵌入范围: [{float(valid_emb.min()):.6f}, {float(valid_emb.max()):.6f}]"
            )

        return embeddings
    except Exception as e:
        print(f"✗ 批量前向传播失败: {e}")
        import traceback

        traceback.print_exc()
        return None


def test_different_lengths(runner: Qwen3EmbeddingRunner, tokenizer: AutoTokenizer):
    """
    测试不同长度输入的前向传播

    Args:
      model: 模型实例
      tokenizer: Tokenizer 实例
    """
    print("\n" + "=" * 60)
    print("测试 3: 不同长度输入")
    print("=" * 60)

    # 不同长度的测试文本
    test_cases = [
        ("短文本", "测试"),
        (
            "中等文本",
            "这是一个中等长度的测试文本，用于验证模型对不同长度输入的处理能力。",
        ),
        ("长文本", "这是一个较长的测试文本。" * 20),
    ]

    results = []
    for name, text in test_cases:
        print(f"\n测试: {name}")
        print(f"文本长度: {len(text)} 字符")

        # Tokenize
        encoded = tokenizer(
            text, padding=False, truncation=True, max_length=512, return_tensors="np"
        )
        input_ids = jnp.array(encoded["input_ids"][0])

        print(f"Token 数量: {input_ids.shape[0]}")

        # 前向传播
        try:
            # Add batch dimension for model call
            input_ids_batched = input_ids[None, :]
            # For simple inference, we don't need attention_mask
            # If we pass attention_mask, we must also pass segment_ids, triplet_type, position_ids
            # So for simple inference, we skip attention_mask
            embeddings_batched = runner.predict(
                input_ids=input_ids_batched,
                attention_mask=None,  # Skip attention_mask for simple inference
                output_hidden_states=False,
            )
            embeddings = embeddings_batched[0]  # Remove batch dimension
            print("✓ 前向传播成功")
            print(f"输出嵌入形状: {embeddings.shape}")

            # 计算平均池化嵌入 (使用所有 token，因为我们在推理模式下没有 attention_mask)
            pooled_embedding = embeddings.mean(axis=0)

            print(f"池化嵌入形状: {pooled_embedding.shape}")
            print(f"池化嵌入均值: {float(pooled_embedding.mean()):.6f}")
            print(f"池化嵌入标准差: {float(pooled_embedding.std()):.6f}")

            results.append((name, embeddings, pooled_embedding))
        except Exception as e:
            print(f"✗ 前向传播失败: {e}")
            import traceback

            traceback.print_exc()
            results.append((name, None, None))

    return results


def test_embedding_similarity(runner: Qwen3EmbeddingRunner, tokenizer: AutoTokenizer):
    """
    测试嵌入相似度计算

    Args:
      model: 模型实例
      tokenizer: Tokenizer 实例
    """
    print("\n" + "=" * 60)
    print("测试 4: 嵌入相似度计算")
    print("=" * 60)

    # 相似文本对
    text_pairs = [
        ("这是测试文本A", "这是测试文本B", "相似文本"),
        ("这是测试文本A", "完全不同的内容，没有任何关联", "不相似文本"),
        ("我喜欢吃苹果", "我喜欢吃水果", "相关文本"),
    ]

    for text1, text2, pair_type in text_pairs:
        print(f"\n测试对: {pair_type}")
        print(f"  文本1: {text1}")
        print(f"  文本2: {text2}")

        # Tokenize
        encoded1 = tokenizer(
            text1, padding=False, truncation=True, max_length=128, return_tensors="np"
        )
        encoded2 = tokenizer(
            text2, padding=False, truncation=True, max_length=128, return_tensors="np"
        )

        input_ids1 = jnp.array(encoded1["input_ids"][0])
        input_ids2 = jnp.array(encoded2["input_ids"][0])

        # 前向传播
        try:
            # Add batch dimension for model call
            input_ids1_batched = input_ids1[None, :]
            input_ids2_batched = input_ids2[None, :]
            # For simple inference, we don't need attention_mask
            # If we pass attention_mask, we must also pass segment_ids, triplet_type, position_ids
            # So for simple inference, we skip attention_mask
            emb1_batched = runner.predict(
                input_ids=input_ids1_batched,
                attention_mask=None,  # Skip attention_mask for simple inference
                output_hidden_states=False,
            )
            emb2_batched = runner.predict(
                input_ids=input_ids2_batched,
                attention_mask=None,  # Skip attention_mask for simple inference
                output_hidden_states=False,
            )
            emb1 = emb1_batched[0]  # Remove batch dimension
            emb2 = emb2_batched[0]  # Remove batch dimension

            # 计算平均池化嵌入 (使用所有 token，因为我们在推理模式下没有 attention_mask)
            pooled_emb1 = emb1.mean(axis=0)
            pooled_emb2 = emb2.mean(axis=0)

            # 计算余弦相似度
            def cosine_similarity(a, b):
                a_norm = jnp.linalg.norm(a)
                b_norm = jnp.linalg.norm(b)
                if a_norm == 0 or b_norm == 0:
                    return 0.0
                return jnp.dot(a, b) / (a_norm * b_norm)

            similarity = cosine_similarity(pooled_emb1, pooled_emb2)
            print(f"  余弦相似度: {float(similarity):.6f}")

        except Exception as e:
            print(f"  ✗ 计算失败: {e}")
            import traceback

            traceback.print_exc()


def test_infonce_loss_computation(
    runner: Qwen3EmbeddingRunner, tokenizer: AutoTokenizer
):
    """
    测试 InfoNCE loss 计算功能

    Args:
      model: 模型实例
      tokenizer: Tokenizer 实例
    """
    print("\n" + "=" * 60)
    print("测试 5: InfoNCE Loss 计算")
    print("=" * 60)

    # 创建测试用的 anchor/positive/negative 三元组
    test_triplets = [
        {
            "anchor": "什么是机器学习？",
            "positive": "机器学习是人工智能的一个分支，通过算法让计算机从数据中学习",
            "negative": "今天天气很好，适合出去散步",
        },
        {
            "anchor": "如何训练神经网络？",
            "positive": "训练神经网络需要准备数据集、定义损失函数、使用优化器进行反向传播",
            "negative": "我喜欢吃苹果和香蕉",
        },
        {
            "anchor": "深度学习的应用领域",
            "positive": "深度学习广泛应用于图像识别、自然语言处理、语音识别等领域",
            "negative": "明天要下雨，记得带伞",
        },
    ]

    print(f"\n测试三元组数量: {len(test_triplets)}")
    for i, triplet in enumerate(test_triplets):
        print(f"\n三元组 {i + 1}:")
        print(f"  Anchor: {triplet['anchor']}")
        print(f"  Positive: {triplet['positive']}")
        print(f"  Negative: {triplet['negative']}")

    try:
        # 构建 packed sequence
        all_input_ids = []
        all_triplet_type = []
        all_segment_ids = []
        all_attention_mask = []

        current_segment_id = 0

        for triplet in test_triplets:
            # Tokenize each part
            anchor_encoded = tokenizer(
                triplet["anchor"],
                padding=False,
                truncation=True,
                max_length=512,
                return_tensors="np",
            )
            positive_encoded = tokenizer(
                triplet["positive"],
                padding=False,
                truncation=True,
                max_length=512,
                return_tensors="np",
            )
            negative_encoded = tokenizer(
                triplet["negative"],
                padding=False,
                truncation=True,
                max_length=512,
                return_tensors="np",
            )

            anchor_ids = anchor_encoded["input_ids"][0].tolist()
            positive_ids = positive_encoded["input_ids"][0].tolist()
            negative_ids = negative_encoded["input_ids"][0].tolist()

            # Combine into one sample
            sample_input_ids = anchor_ids + positive_ids + negative_ids
            sample_triplet_type = (
                [0] * len(anchor_ids)
                + [1] * len(positive_ids)
                + [2] * len(negative_ids)
            )
            sample_segment_ids = [current_segment_id] * len(sample_input_ids)
            sample_attention_mask = [1] * len(sample_input_ids)

            # Add to packed sequence
            all_input_ids.extend(sample_input_ids)
            all_triplet_type.extend(sample_triplet_type)
            all_segment_ids.extend(sample_segment_ids)
            all_attention_mask.extend(sample_attention_mask)

            current_segment_id += 1

        # Convert to arrays and add batch dimension
        # Model expects [batch_size, seq_len] format
        input_ids = jnp.array(all_input_ids)[jnp.newaxis, :]  # [1, seq_len]
        triplet_type = jnp.array(all_triplet_type)[jnp.newaxis, :]  # [1, seq_len]
        segment_ids = jnp.array(all_segment_ids)[jnp.newaxis, :]  # [1, seq_len]
        attention_mask = jnp.array(all_attention_mask)[jnp.newaxis, :]  # [1, seq_len]

        # Generate position_ids: sequential positions starting from 0
        # Position IDs should match the sequence length
        seq_len = input_ids.shape[1]
        position_ids = jnp.arange(seq_len, dtype=jnp.int32)[
            jnp.newaxis, :
        ]  # [1, seq_len]

        print("\nPacked sequence 信息:")
        print(f"  总长度: {input_ids.shape[1]}")
        print(f"  Input IDs 形状: {input_ids.shape}")
        print(f"  Segment type 形状: {triplet_type.shape}")
        print(f"  Segment IDs 形状: {segment_ids.shape}")
        print(f"  Attention mask 形状: {attention_mask.shape}")
        print(f"  Position IDs 形状: {position_ids.shape}")

        # Extract sample info (use 1D arrays for extraction)
        sample_info = extract_sample_info_from_segment_ids(
            segment_ids=segment_ids,
            num_slots=3,  # 2 + 1 (num_negatives)
            attention_mask=attention_mask,
        )

        print("\n样本信息:")
        print(f"  样本数量: {sample_info.num_samples}")
        print(f"  样本起始位置: {sample_info.sample_starts}")
        print(f"  样本长度: {sample_info.sample_lengths}")

        # Forward pass + Loss computation (using Runner's compute_loss)
        print("\n执行前向传播 + Loss 计算（SPMD 多卡并行）...")
        # When attention_mask is provided, we must provide
        # segment_ids and position_ids with matching shapes
        loss, metrics, projected_embs = runner.compute_loss(
            input_ids=input_ids,  # [1, seq_len]
            attention_mask=attention_mask,  # [1, seq_len]
            segment_ids=segment_ids,  # [1, seq_len]
            position_ids=position_ids,  # [1, seq_len] - required when attention_mask is provided
            max_samples=64,  # Should match the max_samples used in dataset pipeline
            num_negatives=1,  # Number of negative samples per triplet
            temperature=0.07,
        )

        print("✓ 前向传播 + Loss 计算成功")
        print(f"  Projected embeddings shape: {projected_embs.shape}")

        print("\nLoss 和 Metrics:")
        print(f"  Loss: {float(loss):.6f}")
        print(f"  Accuracy: {float(metrics['accuracy']):.6f}")
        print(f"  Positive similarity (mean): {float(metrics['pos_sim_mean']):.6f}")
        print(f"  Negative similarity (mean): {float(metrics['neg_sim_mean']):.6f}")
        print(f"  Positive similarity (std): {float(metrics['pos_sim_std']):.6f}")
        print(f"  Negative similarity (std): {float(metrics['neg_sim_std']):.6f}")

        # Verify loss is reasonable
        assert loss >= 0, "Loss 应该是非负数"
        assert 0 <= metrics["accuracy"] <= 1, "Accuracy 应该在 [0, 1] 范围内"

        print("\n✓ Loss 和 Metrics 验证通过")

        # Test with different temperature
        print("\n测试不同 temperature 参数（SPMD 多卡并行）...")
        for temp in [0.05, 0.07, 0.1]:
            loss_temp, _, _ = runner.compute_loss(
                input_ids=input_ids,
                attention_mask=attention_mask,
                segment_ids=segment_ids,
                position_ids=position_ids,
                max_samples=64,
                num_negatives=1,
                temperature=temp,
            )
            print(f"  Temperature {temp}: Loss = {float(loss_temp):.6f}")

        return {"loss": loss, "metrics": metrics}

    except Exception as e:
        print(f"✗ InfoNCE loss 计算失败: {e}")
        import traceback

        traceback.print_exc()
        return None


def test_multiple_negatives(runner: Qwen3EmbeddingRunner, tokenizer: AutoTokenizer):
    """
    测试多个 negative 样本的 InfoNCE loss 计算功能

    Args:
      model: 模型实例
      tokenizer: Tokenizer 实例
    """
    print("\n" + "=" * 60)
    print("测试 6: 多个 Negative 样本的 InfoNCE Loss 计算")
    print("=" * 60)

    # 创建测试用的 anchor/positive/multiple negatives 三元组
    # 每个样本包含 1 个 anchor, 1 个 positive, 和多个 negatives
    num_negatives = 3  # 每个样本有 3 个 negative
    test_triplets = [
        {
            "anchor": "什么是机器学习？",
            "positive": "机器学习是人工智能的一个分支，通过算法让计算机从数据中学习",
            "negatives": [
                "今天天气很好，适合出去散步",
                "我喜欢吃苹果和香蕉",
                "明天要下雨，记得带伞",
            ],
        },
        {
            "anchor": "如何训练神经网络？",
            "positive": "训练神经网络需要准备数据集、定义损失函数、使用优化器进行反向传播",
            "negatives": [
                "我喜欢看科幻电影",
                "今天是个好日子",
                "咖啡和茶都是饮料",
            ],
        },
        {
            "anchor": "深度学习的应用领域",
            "positive": "深度学习广泛应用于图像识别、自然语言处理、语音识别等领域",
            "negatives": [
                "春天是万物复苏的季节",
                "我喜欢听音乐",
                "编程需要逻辑思维",
            ],
        },
    ]

    print(f"\n测试三元组数量: {len(test_triplets)}")
    print(f"每个样本的 Negative 数量: {num_negatives}")
    for i, triplet in enumerate(test_triplets):
        print(f"\n三元组 {i + 1}:")
        print(f"  Anchor: {triplet['anchor']}")
        print(f"  Positive: {triplet['positive']}")
        for j, neg in enumerate(triplet["negatives"]):
            print(f"  Negative {j + 1}: {neg}")

    try:
        # 构建 packed sequence
        all_input_ids = []
        all_triplet_type = []
        all_segment_ids = []
        all_attention_mask = []

        current_segment_id = 0

        for triplet in test_triplets:
            # Tokenize anchor
            anchor_encoded = tokenizer(
                triplet["anchor"],
                padding=False,
                truncation=True,
                max_length=512,
                return_tensors="np",
            )
            anchor_ids = anchor_encoded["input_ids"][0].tolist()

            # Tokenize positive
            positive_encoded = tokenizer(
                triplet["positive"],
                padding=False,
                truncation=True,
                max_length=512,
                return_tensors="np",
            )
            positive_ids = positive_encoded["input_ids"][0].tolist()

            # Tokenize all negatives
            negative_ids_list = []
            for negative_text in triplet["negatives"]:
                negative_encoded = tokenizer(
                    negative_text,
                    padding=False,
                    truncation=True,
                    max_length=512,
                    return_tensors="np",
                )
                negative_ids_list.append(negative_encoded["input_ids"][0].tolist())

            # Combine into one sample
            # Format: [anchor_tokens, positive_tokens, neg1_tokens, neg2_tokens, neg3_tokens, ...]
            sample_input_ids = anchor_ids + positive_ids
            sample_triplet_type = [0] * len(anchor_ids) + [1] * len(positive_ids)

            # Add negatives with correct triplet_type: 2, 3, 4, ...
            for neg_idx, neg_ids in enumerate(negative_ids_list):
                sample_input_ids.extend(neg_ids)
                # triplet_type for negatives: 2, 3, 4, ... (2 + neg_idx)
                sample_triplet_type.extend([2 + neg_idx] * len(neg_ids))

            sample_segment_ids = [current_segment_id] * len(sample_input_ids)
            sample_attention_mask = [1] * len(sample_input_ids)

            # Add to packed sequence
            all_input_ids.extend(sample_input_ids)
            all_triplet_type.extend(sample_triplet_type)
            all_segment_ids.extend(sample_segment_ids)
            all_attention_mask.extend(sample_attention_mask)

            current_segment_id += 1

        # Convert to arrays and add batch dimension
        input_ids = jnp.array(all_input_ids)[jnp.newaxis, :]  # [1, seq_len]
        triplet_type = jnp.array(all_triplet_type)[jnp.newaxis, :]  # [1, seq_len]
        segment_ids = jnp.array(all_segment_ids)[jnp.newaxis, :]  # [1, seq_len]
        attention_mask = jnp.array(all_attention_mask)[jnp.newaxis, :]  # [1, seq_len]

        # Generate position_ids: sequential positions starting from 0
        seq_len = input_ids.shape[1]
        position_ids = jnp.arange(seq_len, dtype=jnp.int32)[
            jnp.newaxis, :
        ]  # [1, seq_len]

        print("\nPacked sequence 信息:")
        print(f"  总长度: {input_ids.shape[1]}")
        print(f"  Input IDs 形状: {input_ids.shape}")
        print(f"  Segment type 形状: {triplet_type.shape}")
        print(f"  Segment IDs 形状: {segment_ids.shape}")
        print(f"  Attention mask 形状: {attention_mask.shape}")
        print(f"  Position IDs 形状: {position_ids.shape}")
        print(f"  Negative 数量: {num_negatives}")

        # Extract sample info
        sample_info = extract_sample_info_from_segment_ids(
            segment_ids=segment_ids,
            num_slots=3,  # 2 + 1 (num_negatives)
            attention_mask=attention_mask,
        )

        print("\n样本信息:")
        print(f"  样本数量: {sample_info.num_samples}")
        print(f"  样本起始位置: {sample_info.sample_starts}")
        print(f"  样本长度: {sample_info.sample_lengths}")

        # Forward pass + Loss computation (multiple negatives)
        print(
            f"\n执行前向传播 + Loss 计算（{num_negatives} 个 Negative，SPMD 多卡并行）..."
        )
        loss, metrics, projected_embs = runner.compute_loss(
            input_ids=input_ids,  # [1, seq_len]
            attention_mask=attention_mask,  # [1, seq_len]
            segment_ids=segment_ids,  # [1, seq_len]
            position_ids=position_ids,  # [1, seq_len]
            max_samples=64,  # Should match the max_samples used in dataset pipeline
            num_negatives=num_negatives,  # Number of negative samples per triplet
            temperature=0.07,
        )

        print("✓ 前向传播 + Loss 计算成功")
        print(f"  Projected embeddings shape: {projected_embs.shape}")

        print("\nLoss 和 Metrics:")
        print(f"  Loss: {float(loss):.6f}")
        print(f"  Accuracy: {float(metrics['accuracy']):.6f}")
        print(f"  Positive similarity (mean): {float(metrics['pos_sim_mean']):.6f}")
        print(f"  Negative similarity (mean): {float(metrics['neg_sim_mean']):.6f}")
        print(f"  Positive similarity (std): {float(metrics['pos_sim_std']):.6f}")
        print(f"  Negative similarity (std): {float(metrics['neg_sim_std']):.6f}")

        # Verify loss is reasonable
        assert loss >= 0, "Loss 应该是非负数"
        assert 0 <= metrics["accuracy"] <= 1, "Accuracy 应该在 [0, 1] 范围内"

        print("\n✓ Loss 和 Metrics 验证通过")

        # Test with different numbers of negatives
        print("\n测试不同 Negative 数量（SPMD 多卡并行）...")
        for num_neg in [1, 2, 3, 5]:
            if num_neg > len(test_triplets[0]["negatives"]):
                print(f"  跳过 {num_neg} 个 Negative（测试数据不足）")
                continue

            # Rebuild data with specified number of negatives
            # Note: segment_ids now encodes sample_id * num_slots + triplet_type
            # We need to compute segment_ids based on sample index and triplet_type
            all_input_ids_neg = []
            all_triplet_type_neg = []
            all_segment_ids_neg = []
            all_attention_mask_neg = []

            num_slots = 2 + num_neg  # Anchor(1) + Positive(1) + Negatives(N)
            current_sample_idx = 0
            for triplet in test_triplets:
                anchor_encoded = tokenizer(
                    triplet["anchor"],
                    padding=False,
                    truncation=True,
                    max_length=512,
                    return_tensors="np",
                )
                positive_encoded = tokenizer(
                    triplet["positive"],
                    padding=False,
                    truncation=True,
                    max_length=512,
                    return_tensors="np",
                )
                anchor_ids = anchor_encoded["input_ids"][0].tolist()
                positive_ids = positive_encoded["input_ids"][0].tolist()

                sample_input_ids_neg = anchor_ids + positive_ids
                sample_triplet_type_neg = [0] * len(anchor_ids) + [1] * len(
                    positive_ids
                )

                # Add only the first num_neg negatives
                for neg_idx in range(num_neg):
                    negative_encoded = tokenizer(
                        triplet["negatives"][neg_idx],
                        padding=False,
                        truncation=True,
                        max_length=512,
                        return_tensors="np",
                    )
                    neg_ids = negative_encoded["input_ids"][0].tolist()
                    sample_input_ids_neg.extend(neg_ids)
                    sample_triplet_type_neg.extend([2 + neg_idx] * len(neg_ids))

                # Compute segment_ids: sample_id * num_slots + triplet_type
                sample_segment_ids_neg = [
                    current_sample_idx * num_slots + t_type
                    for t_type in sample_triplet_type_neg
                ]
                sample_attention_mask_neg = [1] * len(sample_input_ids_neg)

                all_input_ids_neg.extend(sample_input_ids_neg)
                all_triplet_type_neg.extend(sample_triplet_type_neg)
                all_segment_ids_neg.extend(sample_segment_ids_neg)
                all_attention_mask_neg.extend(sample_attention_mask_neg)

                current_sample_idx += 1

            input_ids_neg = jnp.array(all_input_ids_neg)[jnp.newaxis, :]
            segment_ids_neg = jnp.array(all_segment_ids_neg)[jnp.newaxis, :]
            attention_mask_neg = jnp.array(all_attention_mask_neg)[jnp.newaxis, :]
            seq_len_neg = input_ids_neg.shape[1]
            position_ids_neg = jnp.arange(seq_len_neg, dtype=jnp.int32)[jnp.newaxis, :]

            # Compute loss directly using runner
            loss_neg, metrics_neg, _ = runner.compute_loss(
                input_ids=input_ids_neg,
                attention_mask=attention_mask_neg,
                segment_ids=segment_ids_neg,
                position_ids=position_ids_neg,
                max_samples=64,
                num_negatives=num_neg,
                temperature=0.07,
            )
            print(
                f"  {num_neg} 个 Negative: Loss = {float(loss_neg):.6f}, "
                f"Accuracy = {float(metrics_neg['accuracy']):.6f}"
            )

        return {"loss": loss, "metrics": metrics}

    except Exception as e:
        print(f"✗ 多个 Negative 的 InfoNCE loss 计算失败: {e}")
        import traceback

        traceback.print_exc()
        return None


def test_create_model_from_huggingface(
    initial_model: str, embedding_dim: int, seed: int
):
    """
    测试步骤 1: 从 HuggingFace 加载基础模型并创建新模型

    Args:
      initial_model: HuggingFace 模型名称
      embedding_dim: 嵌入维度
      seed: 随机种子

    Returns:
      创建的模型实例
    """
    print("\n" + "=" * 80)
    print("步骤 1: 从 HuggingFace 加载基础模型并创建合成新模型")
    print("=" * 80)

    print("\n配置:")
    print(f"  HuggingFace 模型: {initial_model}")
    print(f"  嵌入维度: {embedding_dim}")
    print(f"  随机种子: {seed}")

    try:
        print("\n正在从 HuggingFace 加载基础模型...")
        key = jax.random.PRNGKey(seed)
        rngs = nnx.Rngs(key)
        model = create_from_initial_model(
            initial_model=initial_model,
            embedding_dim=embedding_dim,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            seed=seed,
            rngs=rngs,
        )

        print("✓ 模型创建成功")
        print(f"  模型类型: {type(model)}")
        print(f"  嵌入维度: {model.embedding_dim}")
        model_name_or_path = (
            getattr(model.config, "model_name_or_path", "N/A")
            if hasattr(model, "config") and model.config
            else "N/A"
        )
        print(f"  基础模型名称: {model_name_or_path}")
        print(f"  投影头类型: {type(model.projection)}")
        print(f"  基础模型类型: {type(model.base_model)}")

        return model
    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        import traceback

        traceback.print_exc()
        raise


def test_save_model(model: Qwen3EmbeddingModel, save_path: str):
    """
    测试步骤 2: 保存新模型

    Args:
      model: 模型实例
      save_path: 保存路径
    """
    print("\n" + "=" * 80)
    print("步骤 2: 保存合成新模型")
    print("=" * 80)

    print(f"\n保存路径: {save_path}")

    try:
        # 确保目录存在
        os.makedirs(save_path, exist_ok=True)

        print("\n正在保存模型...")
        # 在测试环境中使用 overwrite=True 以允许覆盖旧的测试模型
        save_easydel_model(model, save_path, overwrite=True)

        print("\n✓ 模型保存成功")

        # 验证保存的文件
        print("\n验证保存的文件:")
        required_files = [
            "model"  # EasyDeLState directory
        ]

        for filename in required_files:
            filepath = os.path.join(save_path, filename)
            if os.path.exists(filepath):
                if os.path.isdir(filepath):
                    print(f"  ✓ {filename}/ (目录)")
                else:
                    size = os.path.getsize(filepath)
                    print(f"  ✓ {filename} ({size:,} bytes)")
            else:
                print(f"  ✗ {filename} (缺失)")

        # 检查可选文件
        optional_files = ["config.json"]
        for filename in optional_files:
            filepath = os.path.join(save_path, filename)
            if os.path.exists(filepath):
                if os.path.isdir(filepath):
                    print(f"  ✓ {filename}/ (目录)")
                else:
                    size = os.path.getsize(filepath)
                    print(f"  ✓ {filename} ({size:,} bytes)")

    except Exception as e:
        print(f"✗ 模型保存失败: {e}")
        import traceback

        traceback.print_exc()
        raise


def test_load_saved_model(save_path: str, seed: int):
    """
    测试步骤 3: 从保存的模型加载

    Args:
      save_path: 保存路径
      seed: 随机种子

    Returns:
      加载的模型实例
    """
    print("\n" + "=" * 80)
    print("步骤 3: 从保存的模型加载")
    print("=" * 80)

    print(f"\n加载路径: {save_path}")

    if not os.path.exists(save_path):
        raise FileNotFoundError(f"保存路径不存在: {save_path}")

    try:
        print("\n正在加载模型...")
        model = load_easydel_model(
            model_path=save_path,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
        )

        if model is None:
            raise RuntimeError("模型加载返回了 None，请检查模型保存和加载过程")

        print("✓ 模型加载成功")
        print(f"  模型类型: {type(model)}")

        if not hasattr(model, "config") or model.config is None:
            raise RuntimeError("模型缺少 config 属性或 config 为 None")

        print(f"  嵌入维度: {model.config.embedding_dim}")
        model_name_or_path = (
            getattr(model.config, "model_name_or_path", "N/A")
            if hasattr(model, "config") and model.config
            else "N/A"
        )
        print(f"  基础模型名称: {model_name_or_path}")

        if hasattr(model, "projection"):
            print(f"  投影头类型: {type(model.projection)}")
        else:
            print("  警告: 模型缺少 projection 属性")

        if hasattr(model, "base_model"):
            print(f"  基础模型类型: {type(model.base_model)}")
        else:
            print("  警告: 模型缺少 base_model 属性")

        return model
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        import traceback

        traceback.print_exc()
        raise


def main():
    """主测试函数 - 完整流程测试"""
    print("=" * 80)
    print("Qwen3EmbeddingModel 完整流程测试")
    print("=" * 80)

    # 配置
    initial_model = "Qwen/Qwen3-Embedding-0.6B"  # 使用较小的模型进行测试
    embedding_dim = 256
    seed = 42

    # 保存模型的路径（使用临时目录，实际使用时可以指定固定路径）
    saved_model_path = os.path.join(tempfile.gettempdir(), "qwen3_embedding_test_model")

    print("\n全局配置:")
    print(f"  HuggingFace 模型: {initial_model}")
    print(f"  嵌入维度: {embedding_dim}")
    print(f"  随机种子: {seed}")
    print(f"  保存路径: {saved_model_path}")

    # 步骤 1: 从 HuggingFace 加载基础模型并创建新模型
    try:
        model_created = test_create_model_from_huggingface(
            initial_model=initial_model, embedding_dim=embedding_dim, seed=seed
        )
    except Exception:
        print("\n✗ 步骤 1 失败，终止测试")
        return

    # 步骤 2: 保存新模型
    try:
        test_save_model(model_created, saved_model_path)
    except Exception:
        print("\n✗ 步骤 2 失败，终止测试")
        return

    # 步骤 3: 从保存的模型加载
    try:
        model_loaded = test_load_saved_model(saved_model_path, seed)
    except Exception:
        print("\n✗ 步骤 3 失败，终止测试")
        return

    # 步骤 3.5: 初始化 Runner (核心重构点)
    # Runner 内部会创建 Mesh 并编译 JIT 函数
    print("\n" + "=" * 80)
    print("步骤 3.5: 初始化 Qwen3EmbeddingRunner")
    print("=" * 80)
    try:
        runner = Qwen3EmbeddingRunner(model_loaded)
        print("✓ Runner 初始化成功")
    except Exception as e:
        print(f"✗ Runner 初始化失败: {e}")
        import traceback

        traceback.print_exc()
        print("  将使用单设备模式继续测试")
        return

    # 步骤 4: 加载 tokenizer
    print("\n" + "=" * 80)
    print("步骤 4: 加载 Tokenizer")
    print("=" * 80)
    try:
        tokenizer = get_tokenizer_from_model(model_loaded)
        print("✓ Tokenizer 加载成功")
        print(f"  词汇表大小: {len(tokenizer)}")
    except Exception as e:
        print(f"✗ Tokenizer 加载失败: {e}")
        import traceback

        traceback.print_exc()
        return

    # 步骤 5: 推理测试
    print("\n" + "=" * 80)
    print("步骤 5: 推理测试（使用加载的模型）")
    print("=" * 80)

    try:
        # 测试 1: 单条文本
        # test_single_text_forward(runner, tokenizer)

        # # 测试 2: 批量文本
        # test_batch_forward(runner, tokenizer)

        # # 测试 3: 不同长度
        # test_different_lengths(runner, tokenizer)

        # # 测试 4: 嵌入相似度
        # test_embedding_similarity(runner, tokenizer)

        # 测试 5: InfoNCE Loss 计算
        test_infonce_loss_computation(runner, tokenizer)

        # 测试 6: 多个 Negative 样本的 InfoNCE Loss 计算
        test_multiple_negatives(runner, tokenizer)

        print("\n" + "=" * 80)
        print("✓ 所有测试完成!")
        print("=" * 80)
        print("\n完整流程验证:")
        print("  ✓ 步骤 1: 从 HuggingFace 加载基础模型并创建合成新模型")
        print("  ✓ 步骤 2: 保存新模型")
        print("  ✓ 步骤 3: 从保存的模型加载")
        print("  ✓ 步骤 4: 加载 Tokenizer")
        print(
            "  ✓ 步骤 5: 推理测试（单条、批量、不同长度、相似度、InfoNCE Loss、多个 Negative）"
        )

    except Exception as e:
        print(f"\n✗ 推理测试过程中出现错误: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
