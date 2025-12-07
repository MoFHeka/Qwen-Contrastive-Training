"""
加载 Qwen/Qwen3-Embedding-0.6B 模型并转换为 EasyDeL 的 JAX 模型

这个脚本演示了如何从 HuggingFace 加载 Qwen3-Embedding-0.6B 模型
并将其转换为 EasyDeL 的 JAX 格式。
"""

import jax
import jax.numpy as jnp

import easydel as ed


def load_qwen3_embedding_model():
    """
    加载 Qwen3-Embedding-0.6B 模型并转换为 EasyDeL JAX 模型

    Returns:
        EasyDeLBaseModule: 加载的 EasyDeL JAX 模型
    """
    # 模型路径
    model_name = "Qwen/Qwen3-Embedding-0.6B"

    print(f"正在加载模型: {model_name}")
    print("从 HuggingFace 加载 PyTorch 模型并转换为 EasyDeL JAX 格式...")

    mesh_backend = ed.EasyDeLBackends.GPU
    # mesh_backend = ed.EasyDeLBackends.CPU
    if mesh_backend == ed.EasyDeLBackends.CPU:
        sharding_axis_dims = (1, 1, 1, 1, 1)
    else:
        sharding_axis_dims = (4, 1, 1, 1, 1)

    # 使用 AutoEasyDeLModel 加载基础模型(embedding 模型没有 LM head)
    # 设置 from_torch=True 以从 PyTorch 模型转换
    model = ed.AutoEasyDeLModel.from_pretrained(
        model_name,
        # 数据类型设置
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        precision=jax.lax.Precision.DEFAULT,
        backend=mesh_backend,
        # 从 PyTorch 模型转换
        from_torch=True,
        # 自动分片模型参数
        auto_shard_model=True,
        # 分片配置: 系统有 4 个设备,使用 (4, 1, 1, 1, 1) 将 4 个设备用于数据并行
        # 或者使用 (1, -1, 1, 1, 1) 让系统自动分配所有设备到 FSDP
        # 注意: 所有维度的乘积必须等于总设备数
        sharding_axis_dims=sharding_axis_dims,
        sharding_axis_names=("dp", "fsdp", "ep", "tp", "sp"),
        # 配置参数
        config_kwargs=ed.EasyDeLBaseConfigDict(
            # attn_mechanism=ed.AttentionMechanisms.FLASH_ATTN2,
            # attn_mechanism=ed.AttentionMechanisms.CUDNN,
            # attn_mechanism=ed.AttentionMechanisms.SDPA,
            attn_mechanism=ed.AttentionMechanisms.VANILLA,
            # attn_mechanism=ed.AttentionMechanisms.AUTO,
            # 梯度检查点
            gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NONE,
            # 不支持，自己写
            # pooling_mode=None
        ),
        # 详细输出
        verbose=True,
    )

    print("模型加载完成!")
    print(f"模型类型: {type(model)}")
    print(f"配置类型: {type(model.config)}")
    print(f"模型参数量: {model.config.num_hidden_layers} 层")
    print(f"隐藏层大小: {model.config.hidden_size}")
    print(f"词汇表大小: {model.config.vocab_size}")

    return model


def test_model_forward(model):
    """
    测试模型的前向传播

    Args:
        model: EasyDeL 模型实例
    """
    print("\n测试模型前向传播...")

    # 创建测试输入
    # 注意: 如果使用数据并行(DP=4), batch size 应该能被 4 整除以获得最佳性能
    # 这里使用 batch_size=4 来匹配数据并行配置
    input_ids = jnp.array(
        [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
            [31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
        ]
    )

    print(f"输入形状: {input_ids.shape}")

    # 前向传播需要在 model.mesh 上下文管理器中运行
    with model.mesh:
        outputs = model(
            input_ids=input_ids,
            output_hidden_states=True,
        )

    print(f"输出隐藏状态形状: {outputs.last_hidden_state.shape}")
    print(f"输出隐藏状态数据类型: {outputs.last_hidden_state.dtype}")

    # 如果是 embedding 模型,可以提取 pooled embeddings
    # 这里我们使用最后一个隐藏状态的平均池化作为示例
    if hasattr(outputs, "last_hidden_state"):
        pooled_embeddings = jnp.mean(outputs.last_hidden_state, axis=1)
        print(f"池化后的嵌入形状: {pooled_embeddings.shape}")
        print(f"嵌入向量示例 (前10个维度): {pooled_embeddings[0, :10]}")

    print("前向传播测试完成!")


def main():
    """主函数"""
    print("=" * 60)
    print("Qwen3-Embedding-0.6B 模型加载脚本")
    print("=" * 60)

    # 加载模型
    model = load_qwen3_embedding_model()

    # 测试模型
    test_model_forward(model)

    print("\n" + "=" * 60)
    print("脚本执行完成!")
    print("=" * 60)

    # 返回模型以便进一步使用
    return model


if __name__ == "__main__":
    model = main()
