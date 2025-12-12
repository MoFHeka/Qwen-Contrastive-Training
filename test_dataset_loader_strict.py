"""
严格的数据加载器测试，包含详细的诊断信息
"""

import json
import os
import sys
import traceback

import numpy as np
from transformers import AutoTokenizer

from dataset_loader import (
    create_dataset_pipeline,
    extract_sample_info,
    extract_sample_info_from_segment_ids,
)


def verify_position_ids(
    position_ids: np.ndarray,
    segment_ids: np.ndarray,
    attention_mask: np.ndarray | None = None,
    verbose: bool = False,
) -> bool:
    """
    Verify position_ids are correctly generated.

    Rules:
    1. position_ids must exist and have the same shape as segment_ids
    2. Each sample (segment_id) should reset position_id starting from 0
    3. position_ids should increment within each sample
    4. Padding positions should have position_id = 0
    """
    if position_ids is None:
        if verbose:
            print("  ✗ position_ids is None")
        return False

    position_ids = np.array(position_ids)
    segment_ids = np.array(segment_ids)

    # Check shape consistency
    if position_ids.shape != segment_ids.shape:
        if verbose:
            print(
                f"  ✗ Shape mismatch: position_ids {position_ids.shape} vs segment_ids {segment_ids.shape}"
            )
        return False

    # Handle 2D batch case
    if position_ids.ndim == 2:
        for i in range(position_ids.shape[0]):
            if not verify_position_ids(
                position_ids[i],
                segment_ids[i],
                attention_mask[i] if attention_mask is not None else None,
                verbose=False,
            ):
                if verbose:
                    print(f"  ✗ position_ids validation failed for batch index {i}")
                return False
        if verbose:
            print("  ✓ position_ids validation passed for all sequences in batch")
        return True

    # Handle 1D case
    if position_ids.ndim != 1:
        if verbose:
            print(f"  ✗ position_ids must be 1D or 2D, got shape {position_ids.shape}")
        return False

    # Use attention_mask to identify valid positions
    if attention_mask is not None:
        attention_mask = np.array(attention_mask)
        valid_mask = attention_mask == 1
    else:
        # If no attention_mask, identify padding by segment_id
        # Padding positions have segment_id = max(valid_segment_ids) + 1
        # So we find the maximum valid segment_id and mark positions with higher IDs as padding
        if len(segment_ids) > 0:
            # Find all unique segment_ids, excluding negative ones
            unique_seg_ids = np.unique(segment_ids)
            valid_seg_ids = unique_seg_ids[unique_seg_ids >= 0]

            if len(valid_seg_ids) > 0:
                # The maximum valid segment_id
                max_valid_seg_id = np.max(valid_seg_ids)
                # Padding positions have segment_id > max_valid_seg_id
                valid_mask = segment_ids <= max_valid_seg_id
            else:
                # No valid segment_ids found
                valid_mask = np.ones(len(position_ids), dtype=bool)
        else:
            valid_mask = np.ones(len(position_ids), dtype=bool)

    # Group positions by segment_id
    unique_segments = np.unique(segment_ids[valid_mask])

    for seg_id in unique_segments:
        if seg_id < 0:  # Skip invalid segment IDs
            continue

        # Get positions for this segment
        seg_mask = (segment_ids == seg_id) & valid_mask
        seg_positions = position_ids[seg_mask]

        if len(seg_positions) == 0:
            continue

        # Check that positions start from 0 and are consecutive
        expected_positions = np.arange(len(seg_positions))
        if not np.array_equal(seg_positions, expected_positions):
            if verbose:
                print(
                    f"  ✗ Segment {seg_id}: position_ids {seg_positions} != expected {expected_positions}"
                )
            return False

        if verbose and len(seg_positions) > 0:
            print(
                f"  ✓ Segment {seg_id}: position_ids correctly start from 0 and increment (length={len(seg_positions)})"
            )

    # Check padding positions have position_id = 0
    padding_mask = ~valid_mask
    if np.any(padding_mask):
        padding_positions = position_ids[padding_mask]
        if not np.all(padding_positions == 0):
            if verbose:
                print(
                    f"  ✗ Padding positions should have position_id=0, got {np.unique(padding_positions)}"
                )
            return False
        if verbose:
            print(
                f"  ✓ Padding positions correctly have position_id=0 (count={np.sum(padding_mask)})"
            )

    if verbose:
        print("  ✓ position_ids validation passed")
    return True


def create_test_data(num_samples: int = 10, output_file: str = "test_data.jsonl"):
    """创建测试数据"""
    samples = []
    for i in range(num_samples):
        samples.append(
            {
                "anchor": f"ANCHOR_{i:06d}: What is the concept of sample {i}?",
                "positive": f"POSITIVE_{i:06d}: This is the positive explanation for sample {i}.",
                "negative": f"NEGATIVE_{i:06d}: This is unrelated content for sample {i}.",
                "sample_id": i,
            }
        )

    os.makedirs(
        os.path.dirname(output_file) if os.path.dirname(output_file) else ".",
        exist_ok=True,
    )
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    return output_file


def test_basic_flow():
    """测试基本流程"""
    print("=" * 80)
    print("测试1: 基本数据流")
    print("=" * 80)

    # 1. 创建测试数据
    print("\n[1] 创建测试数据...")
    test_file = create_test_data(
        num_samples=50, output_file="test_strict.jsonl"
    )  # 增加样本数量
    print(f"  ✓ 创建了测试文件: {test_file}")

    # 验证文件内容
    with open(test_file, "r") as f:
        lines = f.readlines()
        print(f"  ✓ 文件包含 {len(lines)} 行")
        if len(lines) > 0:
            sample = json.loads(lines[0])
            print(f"  ✓ 第一行示例: anchor={sample.get('anchor', '')[:50]}...")

    # 2. 加载tokenizer
    print("\n[2] 加载tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True
        )
        print("  ✓ Tokenizer加载成功")
    except Exception as e:
        print(f"  ✗ Tokenizer加载失败: {e}")
        return False

    # 3. 创建pipeline
    print("\n[3] 创建pipeline...")
    try:
        pipeline = create_dataset_pipeline(
            data_files=test_file,
            tokenizer=tokenizer,
            seq_length=256,  # 使用更小的序列长度，以便产生更多batch
            shuffle=False,  # 禁用shuffle以便调试
            seed=42,
            max_samples=None,  # 不限制max_samples
        )
        print("  ✓ Pipeline创建成功")
    except Exception as e:
        print(f"  ✗ Pipeline创建失败: {e}")
        traceback.print_exc()
        return False

    # 4. 检查source阶段（跳过，因为create_dataset_pipeline已经调用了source()）
    print("\n[4] 跳过source阶段检查（已在create_dataset_pipeline中调用）...")

    # 5. 检查pack阶段
    print("\n[5] 检查pack阶段...")
    try:
        pack_pipeline = pipeline.pack()
        packed_data = pack_pipeline.get_data()
        print(f"  ✓ Pack阶段数据: {list(packed_data.keys())}")

        for name, packed_source in packed_data.items():
            print(f"    - Packed数据集 '{name}':")
            print(f"      shard_names: {list(packed_source.shard_names)}")
            print(f"      num_shards: {packed_source.num_shards()}")
            print(f"      type: {type(packed_source).__name__}")

            # 尝试读取第一个shard的一些数据
            if len(packed_source.shard_names) > 0:
                shard_name = packed_source.shard_names[0]
                print(f"      尝试打开shard: '{shard_name}'")
                count = 0
                try:
                    for batch in packed_source.open_shard(shard_name):
                        count += 1
                        if count == 1:
                            print(f"      ✓ 第一个batch: {list(batch.keys())}")
                            if "input_ids" in batch:
                                ids = batch["input_ids"]
                                shape = (
                                    ids.shape if hasattr(ids, "shape") else (len(ids),)
                                )
                                print(f"        input_ids形状: {shape}")
                            if "segment_ids" in batch:
                                seg_ids = batch["segment_ids"]
                                shape = (
                                    seg_ids.shape
                                    if hasattr(seg_ids, "shape")
                                    else (len(seg_ids),)
                                )
                                print(f"        segment_ids形状: {shape}")
                            if "triplet_type" in batch:
                                seg_type = batch["triplet_type"]
                                shape = (
                                    seg_type.shape
                                    if hasattr(seg_type, "shape")
                                    else (len(seg_type),)
                                )
                                print(f"        triplet_type形状: {shape}")
                            if "position_ids" in batch:
                                pos_ids = batch["position_ids"]
                                shape = (
                                    pos_ids.shape
                                    if hasattr(pos_ids, "shape")
                                    else (len(pos_ids),)
                                )
                                print(f"        position_ids形状: {shape}")
                        if count >= 3:
                            break
                    print(f"      ✓ 从shard '{shard_name}'读取了 {count} 个batch")
                    if count == 0:
                        print("      ✗ 警告: 没有读取到任何batch!")
                        # 尝试检查底层packed_source
                        if hasattr(packed_source, "_packed_source"):
                            print("      检查底层packed_source...")
                            base_count = 0
                            for b in packed_source._packed_source.open_shard(
                                shard_name
                            ):
                                base_count += 1
                                if base_count >= 3:
                                    break
                            print(f"      底层packed_source产生了 {base_count} 个batch")
                except Exception as e:
                    print(f"      ✗ 读取shard时出错: {e}")
                    traceback.print_exc()
    except Exception as e:
        print(f"  ✗ Pack阶段检查失败: {e}")
        traceback.print_exc()
        return False

    # 6. 检查load和build阶段
    print("\n[6] 检查load和build阶段...")
    try:
        # 先检查load阶段
        load_pipeline = pipeline.pack().load()
        load_data = load_pipeline.get_data()
        print(f"  ✓ Load阶段数据: {list(load_data.keys())}")

        for name, loader in load_data.items():
            print(f"    - Loader '{name}':")
            print(f"      type: {type(loader).__name__}")
            if hasattr(loader, "batch_size"):
                print(f"      batch_size: {loader.batch_size}")

        # 然后检查build阶段
        dataset = load_pipeline.build()
        print("  ✓ Dataset创建成功")
        print(f"    type: {type(dataset).__name__}")

        # 尝试转换为列表看看有多少数据
        try:
            dataset_list = list(dataset)
            print(f"  ✓ Dataset转换为列表: {len(dataset_list)} 个batch")

            if len(dataset_list) > 0:
                batch = dataset_list[0]
                print(f"  ✓ 第一个batch: {list(batch.keys())}")
                input_ids = np.array(batch["input_ids"])
                print(f"    input_ids形状: {input_ids.shape}")
        except Exception as e:
            print(f"  ⚠ 无法转换为列表: {e}")

        batch_count = 0
        total_samples = 0

        for batch in dataset:
            batch_count += 1

            input_ids = np.array(batch["input_ids"])
            segment_ids = (
                np.array(batch.get("segment_ids"))
                if batch.get("segment_ids") is not None
                else None
            )
            triplet_type = (
                np.array(batch.get("triplet_type"))
                if batch.get("triplet_type") is not None
                else None
            )
            position_ids = (
                np.array(batch.get("position_ids"))
                if batch.get("position_ids") is not None
                else None
            )
            attention_mask = (
                np.array(batch.get("attention_mask"))
                if batch.get("attention_mask") is not None
                else None
            )

            if batch_count == 1:
                print("  ✓ 第一个batch:")
                print(f"    input_ids形状: {input_ids.shape}")
                if segment_ids is not None:
                    print(f"    segment_ids形状: {segment_ids.shape}")
                else:
                    print("    segment_ids: None")
                if triplet_type is not None:
                    print(f"    triplet_type形状: {triplet_type.shape}")
                else:
                    print("    triplet_type: None")
                if position_ids is not None:
                    print(f"    position_ids形状: {position_ids.shape}")
                else:
                    print("    position_ids: None")

            # Verify position_ids
            if position_ids is None:
                print("  ✗ 错误: batch中缺少position_ids字段")
                return False

            if segment_ids is None:
                print("  ✗ 错误: batch中缺少segment_ids字段，无法验证position_ids")
                return False

            if not verify_position_ids(
                position_ids, segment_ids, attention_mask, verbose=(batch_count == 1)
            ):
                print(f"  ✗ 错误: batch {batch_count} 的position_ids验证失败")
                return False

            sample_info = extract_sample_info(batch)
            total_samples += sample_info.num_samples

            if batch_count >= 5:
                break

        print(f"  ✓ 处理了 {batch_count} 个batch")
        print(f"  ✓ 处理了 {total_samples} 个样本")

        if batch_count == 0:
            print("  ✗ 错误: 没有处理任何batch!")
            print("    可能原因: batch_size太大或数据不足")
            return False

    except Exception as e:
        print(f"  ✗ Load/Build阶段检查失败: {e}")
        traceback.print_exc()
        return False

    # 清理
    if os.path.exists(test_file):
        os.remove(test_file)
        print(f"\n清理测试文件: {test_file}")

    print("\n" + "=" * 80)
    print("✓ 基本流程测试通过!")
    print("=" * 80)
    return True


def test_max_samples():
    """测试max_samples功能"""
    print("\n" + "=" * 80)
    print("测试2: max_samples功能")
    print("=" * 80)

    # 1. 创建测试数据
    print("\n[1] 创建测试数据...")
    test_file = create_test_data(num_samples=50, output_file="test_max_samples.jsonl")
    print(f"  ✓ 创建了测试文件: {test_file}")

    # 2. 加载tokenizer
    print("\n[2] 加载tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True
        )
        print("  ✓ Tokenizer加载成功")
    except Exception as e:
        print(f"  ✗ Tokenizer加载失败: {e}")
        return False

    # 3. 测试不同的max_samples值
    for max_samples in [None, 5, 10, 20]:
        print(f"\n[3] 测试max_samples={max_samples}...")
        try:
            # 使用较小的seq_length以确保产生足够的packed序列来组成batch
            # batch_size默认为8，所以需要至少8个packed序列
            pipeline = create_dataset_pipeline(
                data_files=test_file,
                tokenizer=tokenizer,
                seq_length=256,  # 使用较小的seq_length以产生更多packed序列
                shuffle=False,
                seed=42,
                max_samples=max_samples,
            )

            dataset = pipeline.pack().load().build()

            batch_count = 0
            total_samples = 0
            max_samples_per_sequence = 0  # 每个packed sequence中的最大样本数

            for batch in dataset:
                batch_count += 1

                # 检查每个packed sequence（batch中的每一行）的样本数
                segment_ids = np.array(batch.get("segment_ids"))
                position_ids = (
                    np.array(batch.get("position_ids"))
                    if batch.get("position_ids") is not None
                    else None
                )
                attention_mask = (
                    np.array(batch.get("attention_mask"))
                    if batch.get("attention_mask") is not None
                    else None
                )

                # Verify position_ids
                if position_ids is None:
                    print(f"  ✗ 错误: batch {batch_count} 中缺少position_ids字段")
                    return False

                if not verify_position_ids(
                    position_ids, segment_ids, attention_mask, verbose=False
                ):
                    print(f"  ✗ 错误: batch {batch_count} 的position_ids验证失败")
                    return False

                if segment_ids is not None:
                    if segment_ids.ndim == 2:
                        # Batch维度: 检查每个packed sequence
                        batch_size = segment_ids.shape[0]
                        for i in range(batch_size):
                            seq_segment_ids = segment_ids[i]
                            seq_info = extract_sample_info_from_segment_ids(
                                seq_segment_ids,
                                attention_mask[i]
                                if attention_mask is not None
                                else None,
                            )
                            max_samples_per_sequence = max(
                                max_samples_per_sequence, seq_info.num_samples
                            )
                            total_samples += seq_info.num_samples

                            # 检查每个packed sequence是否超过max_samples限制
                            if (
                                max_samples is not None
                                and seq_info.num_samples > max_samples
                            ):
                                print(
                                    f"  ✗ 错误: batch {batch_count} 中的序列 {i} 包含 {seq_info.num_samples} 个样本，超过max_samples={max_samples}"
                                )
                                return False
                    else:
                        # 1D: 单个序列
                        seq_info = extract_sample_info_from_segment_ids(segment_ids)
                        max_samples_per_sequence = max(
                            max_samples_per_sequence, seq_info.num_samples
                        )
                        total_samples += seq_info.num_samples

                        if (
                            max_samples is not None
                            and seq_info.num_samples > max_samples
                        ):
                            print(
                                f"  ✗ 错误: batch {batch_count} 包含 {seq_info.num_samples} 个样本，超过max_samples={max_samples}"
                            )
                            return False

            print(f"  ✓ 处理了 {batch_count} 个batch")
            print(f"  ✓ 处理了 {total_samples} 个样本")
            print(f"  ✓ 每个packed sequence最多包含 {max_samples_per_sequence} 个样本")

            if max_samples is not None:
                if max_samples_per_sequence > max_samples:
                    print(
                        f"  ✗ 错误: 最大样本数 {max_samples_per_sequence} 超过max_samples={max_samples}"
                    )
                    return False
                else:
                    print(
                        f"  ✓ max_samples限制生效: {max_samples_per_sequence} <= {max_samples}"
                    )

            if batch_count == 0:
                print("  ✗ 错误: 没有处理任何batch!")
                return False

        except Exception as e:
            print(f"  ✗ 测试失败: {e}")
            traceback.print_exc()
            return False

    # 清理
    if os.path.exists(test_file):
        os.remove(test_file)
        print(f"\n清理测试文件: {test_file}")

    print("\n" + "=" * 80)
    print("✓ max_samples功能测试通过!")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success1 = test_basic_flow()
    success2 = test_max_samples()

    if success1 and success2:
        print("\n" + "=" * 80)
        print("✓ 所有测试通过!")
        print("=" * 80)
        sys.exit(0)
    else:
        print("\n" + "=" * 80)
        print("✗ 部分测试失败")
        print("=" * 80)
        sys.exit(1)
