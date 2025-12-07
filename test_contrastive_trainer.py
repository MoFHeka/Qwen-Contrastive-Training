"""
测试 ContrastiveTrainer 的基本训练功能

这个脚本用于测试 ContrastiveTrainer 是否能正常进行训练。
包括：
1. 创建测试数据
2. 加载模型
3. 创建数据 pipeline
4. 配置训练参数
5. 运行训练
6. 测试从检查点恢复训练
"""

import os
import tempfile
import json

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from transformers import AutoTokenizer  # noqa: E402

import easydel as ed  # noqa: E402
from easydel.trainers import TrainingArguments  # noqa: E402
from easydel.infra.base_state import EasyDeLState  # noqa: E402
from easydel.data import ResumeState  # noqa: E402

from contrastive_trainer import ContrastiveTrainer  # noqa: E402
from dataset_loader import (
    create_dataset_pipeline,
    create_dataset_source,
    extract_sample_info,
)  # noqa: E402
from qwen3_embedding_modeling import create_from_initial_model  # noqa: E402
from model_io import save_easydel_model, load_easydel_state  # noqa: E402

# jax.config.update("jax_log_compiles", True)

num_devices = jax.device_count()


def create_test_data(num_samples: int = 50, output_file: str = "test_train_data.jsonl"):
    """创建测试训练数据"""
    samples = []
    for i in range(num_samples):
        sample = {
            "anchor": f"ANCHOR_{i:04d}: What is the concept of sample {i}?",
            "positive": f"POSITIVE_{i:04d}: This is the positive explanation for sample {i}.",
            "negative": f"NEGATIVE_{i:04d}: This is unrelated content for sample {i}.",
        }
        samples.append(sample)

    os.makedirs(
        os.path.dirname(output_file) if os.path.dirname(output_file) else ".",
        exist_ok=True,
    )
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

            print(f"✓ 创建测试数据: {output_file} ({num_samples} 个样本)")  # noqa: F541
    return output_file


def initialize_and_save_model(
    initial_model: str = "Qwen/Qwen3-Embedding-0.6B",
    save_path: str = "saved_test_model",
    embedding_dim: int = 256,
    dtype: jnp.dtype = jnp.bfloat16,
    param_dtype: jnp.dtype = jnp.bfloat16,
    seed: int = 42,
    overwrite: bool = False,
):
    """
    初始化模型并保存，用于测试

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
        print("  使用 --overwrite_model 参数可以重新创建模型")
        return save_path

    print(f"\n[1/2] 创建模型从: {initial_model}")
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

    print(f"\n[2/2] 保存模型到: {save_path}")
    try:
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
    except Exception as e:
        print(f"✗ 模型保存失败: {e}")
        import traceback

        traceback.print_exc()
        raise

    print("\n" + "=" * 80)
    print("模型初始化完成!")
    print("=" * 80)
    return save_path


def test_contrastive_trainer(
    saved_model_path: str = "saved_test_model",
    num_samples: int = 50,
    max_steps: int = 5,
    seq_length: int = 1024,
    max_samples: int = 8,
    num_negatives: int = 1,
    temperature: float = 0.07,
):
    """
    测试 ContrastiveTrainer 的基本训练功能

    Args:
        saved_model_path: 已保存的模型路径
        num_samples: 测试样本数量
        max_steps: 最大训练步数（用于快速测试）
        seq_length: 序列长度
        max_samples: 每个packed sequence的最大样本数
        num_negatives: 每个样本的负样本数量
        temperature: InfoNCE loss的温度参数
    """
    print("=" * 80)
    print("ContrastiveTrainer 基本训练测试")
    print("=" * 80)

    # 1. 创建测试数据
    print("\n[1/6] 创建测试数据...")
    with tempfile.TemporaryDirectory() as temp_dir:
        test_data_file = os.path.join(temp_dir, "test_data.jsonl")
        create_test_data(num_samples=num_samples, output_file=test_data_file)

        # 2. 加载 tokenizer（从保存的模型路径获取模型名称）
        print("\n[2/6] 加载 tokenizer...")
        try:
            # 尝试从保存的模型路径读取配置来获取模型名称
            # 如果失败，使用默认值
            model_name = "Qwen/Qwen3-Embedding-0.6B"
            if os.path.exists(saved_model_path):
                config_path = os.path.join(saved_model_path, "config.json")
                if os.path.exists(config_path):
                    with open(config_path, "r", encoding="utf-8") as f:
                        config = json.load(f)
                        if "model_name_or_path" in config:
                            model_name = config["model_name_or_path"]

            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            print(f"✓ Tokenizer 加载成功: {model_name}")  # noqa: F541
        except Exception as e:
            print(f"✗ Tokenizer 加载失败: {e}")
            print("  尝试使用本地模型路径...")
            return

        # 3. 从保存的模型加载
        print(f"\n[3/6] 从保存的模型加载: {saved_model_path}")
        try:
            if not os.path.exists(saved_model_path):
                raise FileNotFoundError(
                    f"保存的模型路径不存在: {saved_model_path}\n"
                    f"请先运行模型初始化: python {__file__} --init_model"
                )
            model_state = load_easydel_state(
                load_directory=saved_model_path,
                dtype=jnp.bfloat16,
                param_dtype=jnp.bfloat16,
                config_kwargs=ed.EasyDeLBaseConfigDict(
                    # attn_mechanism=ed.AttentionMechanisms.FLASH_ATTN2,
                    # attn_mechanism=ed.AttentionMechanisms.CUDNN,
                    # attn_mechanism=ed.AttentionMechanisms.SDPA,
                    attn_mechanism=ed.AttentionMechanisms.VANILLA,
                    # attn_mechanism=ed.AttentionMechanisms.AUTO,
                    gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NOTHING_SAVEABLE,
                ),
                verbose=True,
            )
            print("✓ 模型加载成功")
        except Exception as e:
            print(f"✗ 模型加载失败: {e}")
            import traceback

            traceback.print_exc()
            return

        # 4. 创建数据 pipeline
        print("\n[4/6] 创建数据 pipeline...")
        try:
            pipeline = create_dataset_pipeline(
                data_files=test_data_file,
                tokenizer=tokenizer,
                batch_size=1,
                seq_length=seq_length,
                max_samples=max_samples,
                shuffle=True,
                seed=42,
            )
            # 获取 ShardedDataSource (BaseTrainer 会自动处理数据加载)
            train_data = pipeline.pack().get_data()
            # 获取第一个数据集（通常是 "dataset_0"）
            if isinstance(train_data, dict):
                train_data = list(train_data.values())[0]
            print("✓ 数据 pipeline 创建成功")
            print(f"  - seq_length: {seq_length}")
            print(f"  - max_samples: {max_samples}")
        except Exception as e:
            print(f"✗ 数据 pipeline 创建失败: {e}")
            import traceback

            traceback.print_exc()
            return

        # 5. 配置训练参数
        print("\n[5/6] 配置训练参数...")
        try:
            training_args = TrainingArguments(
                model_name="test_contrastive_model",
                learning_rate=1e-4,
                num_train_epochs=1,
                total_batch_size=num_devices,
                max_training_steps=max_steps,  # 限制步数用于快速测试
                max_sequence_length=seq_length,
                save_steps=2,  # 设置较小的 save_steps 确保有检查点保存
                log_steps=1,
                report_steps=1,
                use_wandb=False,  # 测试时禁用 wandb
                report_metrics=True,
                optimizer="muon",
                scheduler="cosine",
                warmup_steps=2,
                weight_decay=0.01,
                clip_grad=1.0,
                save_directory=os.path.join(temp_dir, "checkpoints"),
                resume_if_possible=False,  # 第一次训练不恢复
                gradient_accumulation_steps=1,
                verbose=True,
            )
            print("✓ 训练参数配置成功")
            print(f"  - learning_rate: {training_args.learning_rate}")
            print(f"  - max_training_steps: {training_args.max_training_steps}")
            print(f"  - total_batch_size: {training_args.total_batch_size}")
        except Exception as e:
            print(f"✗ 训练参数配置失败: {e}")
            import traceback

            traceback.print_exc()
            return

        # 6. 创建 trainer 并开始训练
        print("\n[6/6] 创建 trainer 并开始训练...")
        try:
            trainer = ContrastiveTrainer(
                arguments=training_args,
                model_state=model_state,
                dataset_train=train_data,
                processing_class=tokenizer,
                max_samples=max_samples,
                num_negatives=num_negatives,
                temperature=temperature,
            )
            print("✓ Trainer 创建成功")
            print(f"  - max_samples: {trainer.max_samples}")
            print(f"  - num_negatives: {trainer.num_negatives}")
            print(f"  - temperature: {trainer.temperature}")

            # 检查TensorBoard初始化
            tb_writer = trainer.arguments.get_tensorboard()
            tb_path = trainer.arguments._get_save_directory(create=True)

            print(f"TensorBoard writer: {tb_writer}")
            print(f"TensorBoard log directory: {tb_path}")
            print(f"Can log metrics: {trainer.arguments.can_log_metrics}")
            print(f"Report metrics: {trainer.arguments.report_metrics}")

            print("\n开始训练...")
            print("-" * 80)
            try:
                output = trainer.train()
                print("-" * 80)
                print("✓ 训练完成!")  # noqa: F541
                if output and hasattr(output, "state"):
                    print(f"  - 最终 step: {int(jax.device_get(output.state.step))}")
                if output and hasattr(output, "checkpoint_path"):
                    print(f"  - 检查点路径: {output.checkpoint_path}")

                # 验证检查点是否存在
                checkpoint_dir = training_args.save_directory
                if os.path.exists(checkpoint_dir):
                    checkpoint_files = []
                    for root, dirs, files in os.walk(checkpoint_dir):
                        for file in files:
                            if file.endswith(".ckpt") or "checkpoint" in file.lower():
                                checkpoint_files.append(os.path.join(root, file))
                    if checkpoint_files:
                        print(f"  - 找到 {len(checkpoint_files)} 个检查点文件")
                    else:
                        print("  - 警告: 未找到检查点文件")
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
            return

    print("\n" + "=" * 80)
    print("测试完成!")
    print("=" * 80)


def test_resume_training(
    saved_model_path: str = "saved_test_model",
    num_samples: int = 50,
    initial_steps: int = 5,  # 第一次训练的步数
    resume_steps: int = 3,  # 恢复后训练的步数
    seq_length: int = 1024,
    max_samples: int = 8,
    num_negatives: int = 1,
    temperature: float = 0.07,
):
    """
    测试从检查点恢复训练

    Args:
        saved_model_path: 已保存的模型路径
        num_samples: 测试样本数量
        initial_steps: 第一次训练的步数
        resume_steps: 恢复后训练的步数
        seq_length: 序列长度
        max_samples: 每个packed sequence的最大样本数
        num_negatives: 每个样本的负样本数量
        temperature: InfoNCE loss的温度参数
    """
    print("=" * 80)
    print("ContrastiveTrainer 恢复训练测试")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as temp_dir:
        test_data_file = os.path.join(temp_dir, "test_data.jsonl")
        checkpoint_dir = os.path.join(temp_dir, "checkpoints")

        # 1. 创建测试数据
        print("\n[步骤 1/7] 创建测试数据...")
        create_test_data(num_samples=num_samples, output_file=test_data_file)

        # 2. 加载 tokenizer（从保存的模型路径获取模型名称）
        print("\n[步骤 2/7] 加载 tokenizer...")
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

            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            print(f"✓ Tokenizer 加载成功: {model_name}")
        except Exception as e:
            print(f"✗ Tokenizer 加载失败: {e}")
            return

        # 3. 第一次训练 - 保存检查点
        print("\n[步骤 3/7] 第一次训练（保存检查点）...")
        print("-" * 80)
        try:
            # 从保存的模型加载
            print(f"  从保存的模型加载: {saved_model_path}")
            if not os.path.exists(saved_model_path):
                raise FileNotFoundError(
                    f"保存的模型路径不存在: {saved_model_path}\n"
                    f"请先运行模型初始化: python {__file__} --init_model"
                )
            model_state = load_easydel_state(
                load_directory=saved_model_path,
                dtype=jnp.bfloat16,
                param_dtype=jnp.bfloat16,
                config_kwargs=ed.EasyDeLBaseConfigDict(
                    # attn_mechanism=ed.AttentionMechanisms.FLASH_ATTN2,
                    # attn_mechanism=ed.AttentionMechanisms.CUDNN,
                    # attn_mechanism=ed.AttentionMechanisms.SDPA,
                    attn_mechanism=ed.AttentionMechanisms.VANILLA,
                    # attn_mechanism=ed.AttentionMechanisms.AUTO,
                    gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NOTHING_SAVEABLE,
                ),
                verbose=True,
            )
            print("✓ 模型加载成功")

            # 创建数据 source（关闭 shuffle 以便验证数据恢复）
            dataset_source = create_dataset_source(
                data_files=test_data_file,
                tokenizer=tokenizer,
                batch_size=1,
                seq_length=seq_length,
                max_samples=max_samples,
                shuffle=False,  # 关闭 shuffle 以便验证数据恢复
                seed=42,
            )
            train_data = dataset_source
            if isinstance(train_data, dict):
                train_data = list(train_data.values())[0]

            # 创建一个单独的数据 source 用于记录数据特征（不消耗训练数据）
            print("  记录第一次训练将处理的数据特征...")
            dataset_source_for_recording = create_dataset_source(
                data_files=test_data_file,
                tokenizer=tokenizer,
                batch_size=1,
                seq_length=seq_length,
                max_samples=max_samples,
                shuffle=False,  # 关闭 shuffle 以便验证数据恢复
                seed=42,
            )
            train_data_for_recording = dataset_source_for_recording
            if isinstance(train_data_for_recording, dict):
                train_data_for_recording = list(train_data_for_recording.values())[0]

            # 记录第一次训练将处理的数据特征（用于验证恢复训练时数据是否跳过）
            initial_batches_data = []
            initial_batch_iter = iter(train_data_for_recording.iter_shards())
            for step_idx in range(initial_steps):
                try:
                    batch = next(initial_batch_iter)
                    # 记录每个 batch 的第一个样本的第一个 token（作为特征）
                    ndim = batch["input_ids"].ndim
                    if ndim == 1:
                        batch_size = 1
                        first_token = int(jax.device_get(batch["input_ids"][0]))
                    elif ndim == 2:
                        batch_size = batch["input_ids"].shape[0]
                        first_token = int(jax.device_get(batch["input_ids"][0, 0]))
                    else:
                        raise ValueError(f"Unsupported batch dimension: {ndim}")
                    sample_info = extract_sample_info(batch)
                    initial_batches_data.append(
                        {
                            "step": step_idx,
                            "first_token": first_token,
                            "num_samples": sample_info.num_samples,
                            "batch_size": batch_size,
                        }
                    )
                except StopIteration:
                    break
            print(f"  记录了 {len(initial_batches_data)} 个 batch 的数据特征")

            # 配置训练参数
            model_name = "test_contrastive_model"
            training_args = TrainingArguments(
                model_name=model_name,
                learning_rate=1e-4,
                num_train_epochs=1,
                total_batch_size=num_devices,
                max_training_steps=initial_steps,
                max_sequence_length=seq_length,
                warmup_steps=1,
                save_steps=2,  # 确保有检查点保存
                log_steps=1,
                report_steps=1,
                use_wandb=False,
                report_metrics=True,
                optimizer="muon",
                scheduler="cosine",
                weight_decay=0.01,
                clip_grad=1.0,
                save_directory=checkpoint_dir,
                resume_if_possible=False,  # 第一次训练不恢复
                gradient_accumulation_steps=1,
                verbose=True,
            )

            # 创建 trainer 并训练
            trainer = ContrastiveTrainer(
                arguments=training_args,
                model_state=model_state,
                dataset_train=train_data,
                processing_class=tokenizer,
                max_samples=max_samples,
                num_negatives=num_negatives,
                temperature=temperature,
            )

            output = trainer.train()
            initial_step = int(jax.device_get(output.state.step))
            print(f"✓ 第一次训练完成，最终 step: {initial_step}")

            # 尝试从检查点获取 ResumeState（如果存在）
            dataset_resume_state = None
            if hasattr(output.state, "dataset_states") and output.state.dataset_states:
                # EasyDeLState 可能包含 dataset_states
                dataset_states = output.state.dataset_states
                if isinstance(dataset_states, dict) and "train" in dataset_states:
                    dataset_resume_state = dataset_states["train"]
                elif isinstance(dataset_states, ResumeState):
                    dataset_resume_state = dataset_states
            elif hasattr(output, "dataset_states") and output.dataset_states:
                dataset_states = output.dataset_states
                if isinstance(dataset_states, dict) and "train" in dataset_states:
                    dataset_resume_state = dataset_states["train"]
                elif isinstance(dataset_states, ResumeState):
                    dataset_resume_state = dataset_states

            if dataset_resume_state:
                print(
                    f"✓ 找到数据集恢复状态: shard_index={dataset_resume_state.shard_index}, "
                    f"row_index={dataset_resume_state.row_index}"
                )
            else:
                print("⚠ 未找到数据集恢复状态，将使用检查点中的 step 信息")

            # 验证检查点是否存在
            if not os.path.exists(checkpoint_dir):
                print(f"✗ 检查点目录不存在: {checkpoint_dir}")
                return

            # 查找检查点
            model_checkpoint_dir = os.path.join(checkpoint_dir, model_name)
            if not os.path.exists(model_checkpoint_dir):
                print(f"✗ 模型检查点目录不存在: {model_checkpoint_dir}")
                return

            checkpoint_paths = []
            for entry in os.listdir(model_checkpoint_dir):
                path = os.path.join(model_checkpoint_dir, entry)
                if os.path.isdir(path):
                    checkpoint_paths.append(path)

            if not checkpoint_paths:
                print("✗ 未找到检查点，无法测试恢复训练")
                print(
                    f"  检查点目录内容: {os.listdir(checkpoint_dir) if os.path.exists(checkpoint_dir) else '不存在'}"
                )
                return

            # 选择最新的检查点（通常是数字最大的目录）
            import re

            def extract_number_from_name(name):
                match = re.search(r"\d+", name)
                return int(match.group()) if match else -1

            checkpoint_paths.sort(
                key=lambda x: extract_number_from_name(os.path.basename(x)),
                reverse=True,
            )
            latest_checkpoint = checkpoint_paths[0]
            print(f"✓ 找到检查点: {latest_checkpoint}")

        except Exception as e:
            print(f"✗ 第一次训练失败: {e}")
            import traceback

            traceback.print_exc()
            return

        # 4. 从检查点恢复训练
        print("\n[步骤 4/7] 从检查点恢复训练...")
        print("-" * 80)
        try:
            # 创建数据 source（使用相同配置，trainer 会自动从检查点恢复 ResumeState）
            dataset_source_resume = create_dataset_source(
                data_files=test_data_file,
                tokenizer=tokenizer,
                batch_size=1,
                seq_length=seq_length,
                max_samples=max_samples,
                shuffle=False,  # 关闭 shuffle 以便验证数据恢复
                seed=42,
                # 不手动传入 resume_state，让 trainer 自动从检查点恢复
            )
            train_data_resume = dataset_source_resume
            if isinstance(train_data_resume, dict):
                train_data_resume = list(train_data_resume.values())[0]

            # 配置训练参数（继续训练）
            # 使用相同的 checkpoint_dir，这样 trainer 才能找到检查点
            # Ensure warmup_steps < max_training_steps for cosine scheduler
            max_training_steps_resume = initial_step + resume_steps
            warmup_steps_resume = 0  # No warmup when resuming
            if max_training_steps_resume <= warmup_steps_resume:
                # If max_training_steps is too small, adjust it
                max_training_steps_resume = warmup_steps_resume + 1
            training_args_resume = TrainingArguments(
                model_name=model_name,
                learning_rate=1e-4,
                num_train_epochs=1,
                total_batch_size=num_devices,
                max_training_steps=max_training_steps_resume,  # 继续训练 resume_steps 步
                warmup_steps=warmup_steps_resume,
                max_sequence_length=seq_length,
                save_steps=10,
                log_steps=1,
                report_steps=1,
                use_wandb=False,
                report_metrics=True,
                optimizer="muon",
                scheduler="cosine",
                weight_decay=0.01,
                clip_grad=1.0,
                save_directory=checkpoint_dir,  # 使用相同的检查点目录
                resume_if_possible=True,  # 自动从检查点恢复
                gradient_accumulation_steps=1,
                verbose=True,
            )

            # 从config创建模型
            config = ed.EasyDeLBaseConfig.from_pretrained(
                os.path.join(saved_model_path, "config.json"),
                local_files_only=True,
            )
            # 修复bug: 修改config的partition_axis
            try:
                config.partition_axis = ed.PartitionAxis(**config.partition_axis)
            except Exception as e:
                print(f"✗ 修复config的partition_axis失败: {e}")
                config.partition_axis = ed.PartitionAxis()
            config_model = ed.AutoEasyDeLModel.from_config(
                config=config,  # 实际使用的时候可以从文件中加载config.json
                dtype=jnp.bfloat16,
                param_dtype=jnp.bfloat16,
                precision=jax.lax.Precision.DEFAULT,
            )

            # 创建新的 trainer（使用原始模型，trainer 会自动从检查点加载状态）
            trainer_resume = ContrastiveTrainer(
                arguments=training_args_resume,
                model=config_model,  # 传入空模型，trainer 会自动从检查点恢复状态
                dataset_train=train_data_resume,
                processing_class=tokenizer,
                max_samples=max_samples,
                num_negatives=num_negatives,
                temperature=temperature,
            )

            print(f"  继续训练 {resume_steps} 步...")
            output_resume = trainer_resume.train()
            final_step = int(jax.device_get(output_resume.state.step))
            print(f"✓ 恢复训练完成，最终 step: {final_step}")

            # 验证训练步数（trainer 会自动从检查点恢复，所以恢复的 step 应该是 initial_step）
            expected_final_step = initial_step + resume_steps
            if final_step == expected_final_step:
                print(f"✓ 训练步数正确: {initial_step} + {resume_steps} = {final_step}")
            else:
                print("⚠ 警告: 训练步数不符合预期")
                print(f"  预期: {expected_final_step}, 实际: {final_step}")

            # 验证数据恢复：检查恢复训练处理的数据是否与第一次训练不同
            # 注意：trainer 会自动从检查点恢复 ResumeState 并应用到数据源
            print("\n  验证数据恢复: 检查恢复训练处理的数据...")
            # 创建一个新的数据源迭代器来验证数据位置（trainer 已自动应用 ResumeState）
            resume_batches_data = []
            resume_batch_iter = iter(train_data_resume.iter_shards())
            for step_idx in range(resume_steps):
                try:
                    batch = next(resume_batch_iter)
                    # 记录每个 batch 的第一个样本的第一个 token（作为特征）
                    ndim = batch["input_ids"].ndim
                    if ndim == 1:
                        batch_size = 1
                        first_token = int(jax.device_get(batch["input_ids"][0]))
                    elif ndim == 2:
                        batch_size = batch["input_ids"].shape[0]
                        first_token = int(jax.device_get(batch["input_ids"][0, 0]))
                    else:
                        raise ValueError(f"Unsupported batch dimension: {ndim}")
                    sample_info = extract_sample_info(batch)
                    initial_batches_data.append(
                        {
                            "step": step_idx,
                            "first_token": first_token,
                            "num_samples": sample_info.num_samples,
                            "batch_size": batch_size,
                        }
                    )
                except StopIteration:
                    break

            print(f"  恢复训练处理了 {len(resume_batches_data)} 个 batch")
            if initial_batches_data and resume_batches_data:
                # 检查是否有重复的数据
                initial_tokens = {b["first_token"] for b in initial_batches_data}
                resume_tokens = {b["first_token"] for b in resume_batches_data}
                overlap = initial_tokens & resume_tokens
                if overlap:
                    print(
                        f"  ⚠ 警告: 发现 {len(overlap)} 个重复的 batch（基于第一个 token）"
                    )
                    print(f"    重复的 tokens: {sorted(list(overlap))[:5]}...")
                else:
                    print("  ✓ 数据恢复正确: 恢复训练的数据与第一次训练的数据没有重复")
                    print(
                        f"    第一次训练处理了 {len(initial_batches_data)} 个不同的 batch"
                    )
                    print(
                        f"    恢复训练处理了 {len(resume_batches_data)} 个不同的 batch"
                    )

        except Exception as e:
            print(f"✗ 恢复训练失败: {e}")
            import traceback

            traceback.print_exc()
            return

    print("\n" + "=" * 80)
    print("恢复训练测试完成!")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="测试 ContrastiveTrainer")
    parser.add_argument(
        "--saved_model_path",
        type=str,
        default="saved_test_model",
        help="已保存的模型路径（用于测试）",
    )
    parser.add_argument(
        "--init_model",
        action="store_true",
        help="初始化并保存模型（使用 --initial_model 指定初始模型）",
    )
    parser.add_argument(
        "--initial_model",
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
        help="初始模型名称或路径（用于初始化模型）",
    )
    parser.add_argument(
        "--overwrite_model",
        action="store_true",
        help="覆盖已存在的模型（用于 --init_model）",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="测试样本数量",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=5,
        help="最大训练步数（用于快速测试）",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=1024,
        help="序列长度",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=8,
        help="每个packed sequence的最大样本数",
    )
    parser.add_argument(
        "--num_negatives",
        type=int,
        default=1,
        help="每个样本的负样本数量",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="InfoNCE loss的温度参数",
    )
    parser.add_argument(
        "--test_resume",
        action="store_true",
        help="测试恢复训练功能",
    )
    parser.add_argument(
        "--initial_steps",
        type=int,
        default=3,
        help="第一次训练的步数（用于恢复测试）",
    )
    parser.add_argument(
        "--resume_steps",
        type=int,
        default=3,
        help="恢复后训练的步数（用于恢复测试）",
    )

    args = parser.parse_args()

    # 如果指定了 --init_model，先初始化并保存模型
    if args.init_model:
        initialize_and_save_model(
            initial_model=args.initial_model,
            save_path=args.saved_model_path,
            embedding_dim=256,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            seed=42,
            overwrite=args.overwrite_model,
        )
        print("\n模型初始化完成，可以运行测试了")
    else:
        # 运行测试
        if args.test_resume:
            # 测试恢复训练
            test_resume_training(
                saved_model_path=args.saved_model_path,
                num_samples=args.num_samples,
                initial_steps=args.initial_steps,
                resume_steps=args.resume_steps,
                seq_length=args.seq_length,
                max_samples=args.max_samples,
                num_negatives=args.num_negatives,
                temperature=args.temperature,
            )
        else:
            # 基本训练测试
            test_contrastive_trainer(
                saved_model_path=args.saved_model_path,
                num_samples=args.num_samples,
                max_steps=args.max_steps,
                seq_length=args.seq_length,
                max_samples=args.max_samples,
                num_negatives=args.num_negatives,
                temperature=args.temperature,
            )
