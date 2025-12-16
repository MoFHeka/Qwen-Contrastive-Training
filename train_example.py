#!/usr/bin/env python3
"""
使用 data 目录下的 JSON 文件进行训练的脚本

这个脚本演示了如何使用"data"目录下的JSON文件进行对比学习训练。
它使用了项目中定义的所有组件：
1. 数据加载器 (dataset_loader.py)
2. 模型 (qwen3_embedding_modeling.py)
3. 训练器 (contrastive_trainer.py)
4. 模型IO工具 (model_io.py)
"""

import os
import json
import argparse
import tempfile

# os.environ["JAX_DISABLE_JIT"] = "true"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".80"
os.environ["JAX_TRACEBACK_FILTERING"] = "off"
# os.environ['XLA_FLAGS'] = ''

import jax
import jax.numpy as jnp

from transformers import AutoTokenizer

import easydel as ed
from easydel.trainers import TrainingArguments

from dataset_loader import create_dataset_source
from qwen3_embedding_modeling import create_from_initial_model
from model_io import save_easydel_model, load_easydel_state
from contrastive_trainer import ContrastiveTrainer

jax.clear_caches()
import jax.extend

jax.extend.backend.get_backend().defragment()


def find_json_files(data_dir: str = "data") -> list[str]:
    """查找data目录下的所有JSON文件"""
    json_files = []
    if os.path.exists(data_dir):
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".json") or file.endswith(".jsonl"):
                    json_files.append(os.path.join(root, file))
    return json_files


def initialize_model(
    initial_model: str = "Qwen/Qwen3-Embedding-0.6B",
    save_path: str = "initial_saved_model",
    embedding_dim: int = 256,
    dtype: jnp.dtype = jnp.bfloat16,
    param_dtype: jnp.dtype = jnp.bfloat16,
    seed: int = 42,
    overwrite: bool = False,
):
    """
    初始化模型并保存

    Args:
        initial_model: 初始模型名称或路径
        save_path: 保存路径
        embedding_dim: 嵌入维度
        dtype: 计算数据类型
        param_dtype: 参数数据类型
        seed: 随机种子
        overwrite: 是否覆盖已存在的模型

    Returns:
        保存路径
    """
    print("=" * 80)
    print("初始化并保存模型")
    print("=" * 80)

    # 检查模型是否已存在
    if os.path.exists(save_path) and not overwrite:
        print(f"✓ 模型已存在: {save_path}")
        return save_path

    # 1. 加载 tokenizer
    print(f"\n[1/3] 加载 tokenizer 从: {initial_model}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(initial_model, trust_remote_code=True)
        print("✓ Tokenizer 加载成功")
    except Exception as e:
        print(f"✗ Tokenizer 加载失败: {e}")
        import traceback

        traceback.print_exc()
        raise

    # 2. 创建模型
    print(f"\n[2/3] 创建模型从: {initial_model}")
    try:
        model = create_from_initial_model(
            initial_model=initial_model,
            embedding_dim=embedding_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            seed=seed,
        )
        print("✓ 模型创建成功")
    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        import traceback

        traceback.print_exc()
        raise

    # 3. 保存模型和 tokenizer
    print(f"\n[3/3] 保存模型和 tokenizer 到: {save_path}")
    try:
        # 保存模型
        save_easydel_model(
            model=model,
            save_path=save_path,
            optimizer=None,
            step=0,
            save_method="easydel",
            overwrite=overwrite,
            dtype=dtype,
        )
        print("✓ 模型保存成功")

        # 保存 tokenizer 到单独的子文件夹
        tokenizer_path = os.path.join(save_path, "tokenizer")
        tokenizer.save_pretrained(tokenizer_path)
        print(f"✓ Tokenizer 保存成功: {tokenizer_path}")
    except Exception as e:
        print(f"✗ 保存失败: {e}")
        import traceback

        traceback.print_exc()
        raise

    print("\n" + "=" * 80)
    print("模型和 Tokenizer 初始化完成!")
    print("=" * 80)
    return save_path


def train_model(
    data_files: str | list[str],
    saved_model_path: str = "initial_saved_model",
    output_dir: str = "training_output",
    batch_size: int = 1,
    num_gradient_accumulation_steps: int = 1,
    max_steps: int = 100000,
    seq_length: int = 32768,
    max_samples: int = 9,
    max_sample_length: int = 5120,
    num_negatives: int = 1,
    temperature: float = 0.07,
    learning_rate: float = 1e-4,
):
    """
    使用指定的JSON文件进行训练

    Args:
        data_files: 数据文件路径（可以是字符串或列表）
        saved_model_path: 已保存的模型路径
        output_dir: 输出目录
        max_steps: 最大训练步数
        seq_length: 序列长度
        max_samples: 每个packed sequence的最大样本数
        num_negatives: 每个样本的负样本数量
        temperature: InfoNCE loss的温度参数
        learning_rate: 学习率
    """
    print("=" * 80)
    print("开始训练")
    num_devices = jax.device_count()
    print("=" * 80)

    # 1. 加载 tokenizer
    print("\n[1/5] 加载 tokenizer...")
    try:
        # 尝试从保存的模型路径读取配置来获取模型名称
        model_name = "Qwen/Qwen3-Embedding-0.6B"
        if os.path.exists(saved_model_path):
            config_path = os.path.join(saved_model_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    if "model_name_or_path" in config:
                        model_name = config["model_name_or_path"]

        tokenizer_path = os.path.join(saved_model_path, "tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        print(f"✓ Tokenizer 加载成功: {model_name}")
    except Exception as e:
        print(f"✗ Tokenizer 加载失败: {e}")
        raise

    # 2. 从保存的模型加载
    print(f"\n[2/5] 从保存的模型加载: {saved_model_path}")
    try:
        if not os.path.exists(saved_model_path):
            raise FileNotFoundError(
                f"保存的模型路径不存在: {saved_model_path}\n请先运行模型初始化"
            )
        # AttentionMechanisms:
        # AUTO: Automatically selects best mechanism for hardware.
        # FLASH_ATTN2: FlashAttention-2 for efficient GPU computation.
        # RING: RingAttention for sequence parallelism.
        # VANILLA: Standard dot-product attention.
        # SPLASH: SplashAttention optimized for TPUs.
        # CUDNN: cuDNN implementation for NVIDIA GPUs.
        # BLOCKWISE: Blockwise computation for memory efficiency.
        # SDPA: Scaled Dot Product Attention (JAX native).
        # CUDA_FLASH_ATTN2: CUDA-specific FlashAttention-2.
        # RAGGED_PAGE_ATTENTION_V3: Paged attention for efficient inference.
        # RAGGED_PAGE_ATTENTION_V2: Paged attention for efficient inference.
        # REGRESSIVE_DECODE: Optimized autoregressive decoding.
        # EasyDeLGradientCheckPointers:
        # EVERYTHING_SAVEABLE = "everything_saveable"
        # NOTHING_SAVEABLE = "nothing_saveable"
        # CHECKPOINT_DOTS = "checkpoint_dots"
        # CHECKPOINT_DOTS_WITH_NO_BATCH_DMIS = "checkpoint_dots_with_no_batch_dims"
        # NONE = ""
        # DOTS_SAVEABLE = "dots_saveable"
        # DOTS_WITH_NO_BATCH_DIMS_AVAILABLE = "dots_with_no_batch_dims_saveable"
        # SAVE_ANYTHING_EXCEPT_THESE_NAMES = "save_anything_except_these_names"
        # SAVE_ANY_NAMES_BUT_THESE = "save_any_names_but_these"
        # SAVE_ONLY_THESE_NAMES = "save_only_these_names"
        # SAVE_FROM_BOTH_POLICIES = "save_from_both_policies"
        model_state = load_easydel_state(
            load_directory=saved_model_path,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            sharding_axis_dims=[1, -1, 1, 1, 1],
            sharding_axis_names=["dp", "fsdp", "ep", "tp", "sp"],
            config_kwargs=ed.EasyDeLBaseConfigDict(
                attn_mechanism=ed.AttentionMechanisms.AUTO,
                # gradient_checkpointing=ed.EasyDeLGradientCheckPointers.CHECKPOINT_DOTS_WITH_NO_BATCH_DMIS,
            ),
            verbose=True,
        )
        print("✓ 模型加载成功")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        import traceback

        traceback.print_exc()
        raise

    # 3. 创建数据 source
    print("\n[3/5] 创建数据 source...")
    try:
        train_data = create_dataset_source(
            data_files=data_files,
            tokenizer=tokenizer,
            batch_size=num_devices * batch_size,
            seq_length=seq_length,
            max_samples=max_samples,
            # max_sample_length=max_sample_length,
            shuffle=True,
            seed=42,
        )
        print("✓ 数据 source 创建成功")
        print(f"  - data_files: {data_files}")
        print(f"  - seq_length: {seq_length}")
        print(f"  - max_samples: {max_samples}")
    except Exception as e:
        print(f"✗ 数据 source 创建失败: {e}")
        import traceback

        traceback.print_exc()
        raise

    # 4. 配置训练参数
    print("\n[4/5] 配置训练参数...")
    try:
        training_args = TrainingArguments(
            model_name="contrastive_embedding_model",
            learning_rate=learning_rate,
            num_train_epochs=1,
            total_batch_size=num_devices * batch_size,
            max_training_steps=max_steps,
            max_sequence_length=seq_length,
            save_steps=1000,
            log_steps=10,
            report_steps=10,
            use_wandb=False,
            report_metrics=True,
            optimizer="muon",
            scheduler="cosine",
            warmup_steps=5,
            weight_decay=0.01,
            clip_grad=1.0,
            save_directory=output_dir,
            resume_if_possible=False,
            gradient_accumulation_steps=num_gradient_accumulation_steps,
            performance_mode=False,
            verbose=True,
        )
        print("✓ 训练参数配置成功")
        print(f"  - learning_rate: {training_args.learning_rate}")
        print(f"  - max_training_steps: {training_args.max_training_steps}")
        print(f"  - batch_size: {batch_size}")
        print(f"  - total_batch_size: {training_args.total_batch_size}")
        print(
            f"  - gradient_accumulation_steps: {training_args.gradient_accumulation_steps}"
        )
        print(f"  - save_directory: {training_args.save_directory}")
    except Exception as e:
        print(f"✗ 训练参数配置失败: {e}")
        import traceback

        traceback.print_exc()
        raise

    # 5. 创建 trainer 并开始训练
    print("\n[5/5] 创建 trainer 并开始训练...")
    try:
        trainer = ContrastiveTrainer(
            arguments=training_args,
            model_state=model_state,
            dataset_train=train_data,
            processing_class=tokenizer,
            max_samples=max_samples,
            num_negatives=num_negatives,
            temperature=temperature,
            track_memory=True,  # 开启内存跟踪
        )
        print("✓ Trainer 创建成功")
        print(f"  - max_samples: {trainer.max_samples}")
        print(f"  - num_negatives: {trainer.num_negatives}")
        print(f"  - temperature: {trainer.temperature}")

        print("\n开始训练...")
        print("-" * 80)
        try:
            output = trainer.train()
            print("-" * 80)
            print("✓ 训练完成!")
            if output and hasattr(output, "state"):
                print(f"  - 最终 step: {int(jax.device_get(output.state.step))}")
            if output and hasattr(output, "checkpoint_path"):
                print(f"  - 检查点路径: {output.checkpoint_path}")
        except KeyboardInterrupt:
            print("\n训练被用户中断")
        except Exception as e:
            print(f"\n训练过程中出现错误: {e}")
            import traceback

            traceback.print_exc()
            raise

    except Exception as e:
        print(f"✗ 训练失败: {e}")
        import traceback

        traceback.print_exc()
        raise

    print("\n" + "=" * 80)
    print("训练完成!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="使用 data 目录下的 JSON 文件进行训练")
    parser.add_argument("--data_dir", type=str, default="data", help="数据目录路径")
    parser.add_argument(
        "--data_files",
        type=str,
        nargs="+",
        help="数据文件路径（如果指定，将忽略 --data_dir）",
    )
    parser.add_argument("--init_model", action="store_true", help="初始化并保存模型")
    parser.add_argument(
        "--initial_model",
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
        help="初始模型名称或路径",
    )
    parser.add_argument(
        "--saved_model_path",
        type=str,
        default="initial_saved_model",
        help="已保存的模型路径",
    )
    parser.add_argument(
        "--output_dir", type=str, default="training_output", help="训练输出目录"
    )
    parser.add_argument(
        "--overwrite_model", action="store_true", help="覆盖已存在的模型"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument(
        "--num_gradient_accumulation_steps", type=int, default=1, help="batch size"
    )
    parser.add_argument("--max_steps", type=int, default=100000, help="最大训练步数")
    parser.add_argument("--seq_length", type=int, default=32768, help="序列长度")
    parser.add_argument(
        "--max_samples", type=int, default=5, help="每个packed sequence的最大样本数"
    )
    parser.add_argument(
        "--num_negatives", type=int, default=1, help="每个样本的负样本数量"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.07, help="InfoNCE loss的温度参数"
    )
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")

    arguments = parser.parse_args()

    # 确定数据文件
    if arguments.data_files:
        data_files = arguments.data_files
    else:
        # 查找data目录下的所有JSON文件
        json_files = find_json_files(arguments.data_dir)
        if not json_files:
            print(f"在目录 '{arguments.data_dir}' 中未找到任何JSON文件")
            return
        data_files = json_files
        print(f"找到以下数据文件: {data_files}")

    # 如果指定了 --init_model，先初始化并保存模型
    if arguments.init_model:
        initialize_model(
            initial_model=arguments.initial_model,
            save_path=arguments.saved_model_path,
            embedding_dim=256,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            seed=42,
            overwrite=arguments.overwrite_model,
        )
        print("\n模型初始化完成，可以运行训练了")
    else:
        # 运行训练
        train_model(
            data_files=data_files,
            saved_model_path=arguments.saved_model_path,
            output_dir=arguments.output_dir,
            batch_size=arguments.batch_size,
            num_gradient_accumulation_steps=arguments.num_gradient_accumulation_steps,
            max_steps=arguments.max_steps,
            seq_length=arguments.seq_length,
            max_samples=arguments.max_samples,
            num_negatives=arguments.num_negatives,
            temperature=arguments.temperature,
            learning_rate=arguments.learning_rate,
        )


if __name__ == "__main__":
    main()
