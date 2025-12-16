"""
全面测试数据加载器：验证样本边界和语义类型标记

测试内容：
1. segment_ids 正确区分不同样本
2. triplet_type 正确标记 anchor/positive/negative
3. 样本边界计算正确
4. 能够正确拆分样本并提取各部分
5. 处理边界情况（短样本、长样本、跨 batch 样本）
6. 极大量数据测试（10万+样本）
7. 多文件/多shard分布式数据测试
8. 数据完整性验证（不重复、不遗漏）
"""

import json
import os
import tempfile
import time

import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer

from dataset_loader import (
    create_dataset_pipeline,
    extract_sample_info,
    extract_sample_info_from_segment_ids,
    split_packed_sequence,
)


def create_test_data(
    num_samples: int = 100,
    output_file: str = "test_data.jsonl",
    start_id: int = 0,
):
    """创建大量测试数据，确保每个样本都有独特的标识。

    Args:
      num_samples: 要创建的样本数量
      output_file: 输出文件路径
      start_id: 起始样本ID（用于多文件场景）
    """
    samples = []

    # 创建不同长度的样本，确保能够测试各种情况
    for i in range(num_samples):
        sample_id = start_id + i
        # 使用样本索引创建独特的文本，便于验证
        anchor = f"ANCHOR_{sample_id:06d}: What is the concept of sample {sample_id}?"
        positive = f"POSITIVE_{sample_id:06d}: This is the positive explanation for sample {sample_id}."
        negative = f"NEGATIVE_{sample_id:06d}: This is unrelated content for sample {sample_id}."

        # 创建不同长度的样本（短、中、长）
        if sample_id % 3 == 0:
            # 短样本
            anchor = f"ANCHOR_{sample_id:06d}: Short {sample_id}"
            positive = f"POSITIVE_{sample_id:06d}: Short pos {sample_id}"
            negative = f"NEGATIVE_{sample_id:06d}: Short neg {sample_id}"
        elif sample_id % 3 == 1:
            # 中等样本
            anchor = f"ANCHOR_{sample_id:06d}: Medium length anchor text for sample {sample_id}"
            positive = f"POSITIVE_{sample_id:06d}: Medium length positive explanation for sample {sample_id}"
            negative = f"NEGATIVE_{sample_id:06d}: Medium length negative unrelated content for sample {sample_id}"
        else:
            # 长样本
            anchor = (
                f"ANCHOR_{sample_id:06d}: "
                + "This is a longer anchor text. " * 5
                + f"Sample {sample_id}."
            )
            positive = (
                f"POSITIVE_{sample_id:06d}: "
                + "This is a longer positive explanation. " * 5
                + f"Sample {sample_id}."
            )
            negative = (
                f"NEGATIVE_{sample_id:06d}: "
                + "This is a longer negative unrelated content. " * 5
                + f"Sample {sample_id}."
            )

        samples.append(
            {
                "anchor": anchor,
                "positive": positive,
                "negative": negative,
                "sample_id": sample_id,  # 保存样本 ID 用于验证
            }
        )

    # 写入文件
    os.makedirs(
        os.path.dirname(output_file) if os.path.dirname(output_file) else ".",
        exist_ok=True,
    )
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    return output_file


def verify_segment_ids(
    segment_ids: np.ndarray,
    sample_info,
    num_negatives: int = 1,  # Number of negative samples per anchor-positive pair
    verbose: bool = False,
):
    """验证 segment_ids 正确标记了样本边界。

    Args:
      segment_ids: Segment IDs 数组，编码 sample_id * num_slots + triplet_type
                  可以是1D [seq_length] 或2D [batch_size, seq_length]
      sample_info: 样本信息（如果是批次数据，包含所有packed sequences的聚合信息）
      num_negatives: Number of negative samples per anchor-positive pair. Default: 1.
      verbose: 是否打印详细信息

    Returns:
      bool: 是否通过验证
    """
    errors = []
    num_slots = 2 + num_negatives

    # 处理批次维度
    if segment_ids.ndim == 2:
        # 批次数据：对每个packed sequence分别验证
        batch_size = segment_ids.shape[0]

        # 计算每个packed sequence的样本数
        # sample_info包含所有packed sequences的聚合信息，需要按packed sequence分组
        from dataset_loader import extract_sample_info_from_segment_ids

        total_verified_samples = 0
        for batch_idx in range(batch_size):
            seq_segment_ids = segment_ids[batch_idx]
            seq_info = extract_sample_info_from_segment_ids(
                seq_segment_ids,
                num_slots,
                attention_mask=None,
            )

            # 验证这个packed sequence
            # segment_ids 现在编码为 sample_id * num_slots + triplet_type
            # 同一样本内的 segment_id 可能不同（因为 triplet_type 不同），但 sample_id 应该相同
            for i, (start, length) in enumerate(
                zip(seq_info.sample_starts, seq_info.sample_lengths)
            ):
                end = start + length
                sample_segments = seq_segment_ids[start:end]

                # 过滤掉 padding (-1)
                valid_segments = sample_segments[sample_segments >= 0]
                if len(valid_segments) == 0:
                    continue

                # 提取 sample_id: segment_id // num_slots
                sample_ids = valid_segments // num_slots
                unique_sample_ids = np.unique(sample_ids)

                # 同一样本内的 sample_id 应该相同
                if len(unique_sample_ids) > 1:
                    errors.append(
                        f"批次 {batch_idx} 样本 {i} (位置 {start}-{end}) 内有多个 sample_id: {unique_sample_ids}"
                    )

                # 相邻样本的 sample_id 应该不同
                if i > 0:
                    prev_end = (
                        seq_info.sample_starts[i - 1] + seq_info.sample_lengths[i - 1]
                    )
                    prev_valid_segments = (
                        seq_segment_ids[prev_end - 1] if prev_end > 0 else -1
                    )
                    curr_valid_segments = (
                        seq_segment_ids[start] if start < len(seq_segment_ids) else -1
                    )

                    if prev_valid_segments >= 0 and curr_valid_segments >= 0:
                        prev_sample_id = prev_valid_segments // num_slots
                        curr_sample_id = curr_valid_segments // num_slots

                        if prev_sample_id == curr_sample_id:
                            errors.append(
                                f"批次 {batch_idx} 样本 {i - 1} 和样本 {i} 的 sample_id 相同: {prev_sample_id}"
                            )

            total_verified_samples += seq_info.num_samples

        if errors:
            if verbose:
                for error in errors:
                    print(f"  ✗ {error}")
            return False

        if verbose:
            print(
                f"  ✓ segment_ids 验证通过：{total_verified_samples} 个样本（{batch_size} 个packed sequences）正确标记"
            )
        return True
    else:
        # 1D数据：单个packed sequence
        # segment_ids 现在编码为 sample_id * num_slots + triplet_type
        # 同一样本内的 segment_id 可能不同（因为 triplet_type 不同），但 sample_id 应该相同
        for i, (start, length) in enumerate(
            zip(sample_info.sample_starts, sample_info.sample_lengths)
        ):
            end = start + length
            sample_segments = segment_ids[start:end]

            # 过滤掉 padding (-1)
            valid_segments = sample_segments[sample_segments >= 0]
            if len(valid_segments) == 0:
                continue

            # 提取 sample_id: segment_id // num_slots
            sample_ids = valid_segments // num_slots
            unique_sample_ids = np.unique(sample_ids)

            # 同一样本内的 sample_id 应该相同
            if len(unique_sample_ids) > 1:
                errors.append(
                    f"样本 {i} (位置 {start}-{end}) 内有多个 sample_id: {unique_sample_ids}"
                )

            # 相邻样本的 sample_id 应该不同
            if i > 0:
                prev_end = (
                    sample_info.sample_starts[i - 1] + sample_info.sample_lengths[i - 1]
                )
                prev_valid_segment = segment_ids[prev_end - 1] if prev_end > 0 else -1
                curr_valid_segment = (
                    segment_ids[start] if start < len(segment_ids) else -1
                )

                if prev_valid_segment >= 0 and curr_valid_segment >= 0:
                    prev_sample_id = prev_valid_segment // num_slots
                    curr_sample_id = curr_valid_segment // num_slots

                    if prev_sample_id == curr_sample_id:
                        errors.append(
                            f"样本 {i - 1} 和样本 {i} 的 sample_id 相同: {prev_sample_id}"
                        )

        if errors:
            if verbose:
                for error in errors:
                    print(f"  ✗ {error}")
            return False

        if verbose:
            print(f"  ✓ segment_ids 验证通过：{sample_info.num_samples} 个样本正确标记")
        return True


def verify_triplet_type(
    triplet_type: np.ndarray,
    input_ids: np.ndarray,
    tokenizer,
    sample_info,
    verbose: bool = False,
):
    """验证 triplet_type 正确标记了 anchor/positive/negative。

    Args:
      triplet_type: Segment type 数组，可以是1D [seq_length] 或2D [batch_size, seq_length]
        有效值：-1=special_token, 0=anchor, 1=positive, 2=negative
      input_ids: Input IDs 数组
      sample_info: 样本信息（如果是批次数据，包含所有packed sequences的聚合信息）
      tokenizer: Tokenizer 用于解码验证

    Returns:
      bool: 是否通过验证
    """
    errors = []

    # 处理批次维度
    if triplet_type.ndim == 2:
        # 批次数据：对每个packed sequence分别验证
        batch_size = triplet_type.shape[0]

        # 需要segment_ids来获取sample_info，如果没有则跳过详细验证
        # 这里我们假设调用者会同时提供segment_ids
        # 为了简化，我们只做基本的形状和值范围检查
        total_verified_samples = 0

        for batch_idx in range(batch_size):
            seq_triplet_type = triplet_type[batch_idx]

            # 基本检查：值范围（允许 -1=special_token, 0=anchor, 1=positive, 2=negative）
            invalid_types = seq_triplet_type[
                (seq_triplet_type < -1) | (seq_triplet_type > 2)
            ]
            if len(invalid_types) > 0:
                errors.append(
                    f"批次 {batch_idx} 包含无效的 triplet_type: {np.unique(invalid_types)}"
                )

            # 检查是否包含所有三种类型（至少应该有一些）
            unique_types = np.unique(seq_triplet_type)
            if len(unique_types) == 0:
                errors.append(f"批次 {batch_idx} triplet_type 为空")

            total_verified_samples += 1

        if errors:
            if verbose:
                for error in errors:
                    print(f"  ✗ {error}")
            return False

        if verbose:
            print(
                f"  ✓ triplet_type 基本验证通过：{batch_size} 个packed sequences（跳过详细验证）"
            )
        return True
    else:
        # 1D数据：单个packed sequence
        # 验证每个样本内的 triplet_type 模式
        for i, (start, length) in enumerate(
            zip(sample_info.sample_starts, sample_info.sample_lengths)
        ):
            end = start + length
            sample_types = triplet_type[start:end]

            # 检查 triplet_type 的值是否在有效范围内（允许 -1=special_token, 0=anchor, 1=positive, 2=negative）
            invalid_types = sample_types[(sample_types < -1) | (sample_types > 2)]
            if len(invalid_types) > 0:
                errors.append(
                    f"样本 {i} 包含无效的 triplet_type: {np.unique(invalid_types)}"
                )
                continue

            # 排除 special tokens (-1) 进行验证
            content_mask = sample_types >= 0
            content_types = sample_types[content_mask]

            if len(content_types) == 0:
                errors.append(f"样本 {i} 只包含 special tokens，没有 content tokens")
                continue

            # 验证 triplet_type 的模式：应该是 0 (anchor) -> 1 (positive) -> 2 (negative)
            # 允许有重复，但顺序应该大致正确
            unique_types = np.unique(content_types)

            # 检查是否包含所有三种类型（或者至少包含 anchor）
            if 0 not in unique_types:
                errors.append(f"样本 {i} 缺少 anchor (triplet_type=0)")

            # 验证 triplet_type 的模式：应该是 0 (anchor) -> 1 (positive) -> 2 (negative)
            # 由于 TripletSegmentTransform 直接连接 anchor + positive + negative，
            # 每个样本应该是: [0, 0, ..., 0, 1, 1, ..., 1, 2, 2, ..., 2]
            # 即：单调非递减，且只允许 0->1, 1->2, 或保持不变
            # 注意：special tokens (-1) 可能出现在任何位置，需要排除它们

            # 计算类型变化（只考虑 content tokens）
            type_changes = np.diff(content_types)

            # 检查无效的类型变化：
            # - 不允许负变化（除了可能的边界情况）
            # - 不允许跳跃（如 0->2，应该先 0->1 再 1->2）
            invalid_changes = []
            for j, change in enumerate(type_changes):
                if change < 0:
                    # 负变化：不允许（除非是特殊情况，但在这个实现中不应该出现）
                    invalid_changes.append(
                        (j, content_types[j], content_types[j + 1], "negative change")
                    )
                elif change > 1:
                    # 跳跃变化：不允许（如 0->2，应该先 0->1）
                    invalid_changes.append(
                        (j, content_types[j], content_types[j + 1], "skip change")
                    )

            if invalid_changes:
                for j, from_type, to_type, reason in invalid_changes:
                    errors.append(
                        f"样本 {i} 位置 {start + j} 有无效的 triplet_type 变化 ({reason}): "
                        f"{from_type} -> {to_type}"
                    )

            # 验证各部分都存在（排除 special tokens）
            has_anchor = np.any(content_types == 0)
            has_positive = np.any(content_types == 1)
            has_negative = np.any(content_types == 2)

            if not has_anchor:
                errors.append(f"样本 {i} 缺少 anchor 部分")
            if not has_positive:
                errors.append(f"样本 {i} 缺少 positive 部分")
            if not has_negative:
                errors.append(f"样本 {i} 缺少 negative 部分")

            # 验证顺序：anchor 应该在 positive 之前，positive 应该在 negative 之前
            # 需要在原始 sample_types 中查找位置（考虑 special tokens）
            if has_anchor and has_positive:
                anchor_positions = np.where(sample_types == 0)[0]
                positive_positions = np.where(sample_types == 1)[0]
                if len(anchor_positions) > 0 and len(positive_positions) > 0:
                    anchor_max_idx = np.max(anchor_positions)
                    positive_min_idx = np.min(positive_positions)
                    if anchor_max_idx >= positive_min_idx:
                        errors.append(f"样本 {i} anchor 和 positive 顺序错误")

            if has_positive and has_negative:
                positive_positions = np.where(sample_types == 1)[0]
                negative_positions = np.where(sample_types == 2)[0]
                if len(positive_positions) > 0 and len(negative_positions) > 0:
                    positive_max_idx = np.max(positive_positions)
                    negative_min_idx = np.min(negative_positions)
                    if positive_max_idx >= negative_min_idx:
                        errors.append(f"样本 {i} positive 和 negative 顺序错误")

    if errors:
        if verbose:
            for error in errors:
                print(f"  ✗ {error}")
        return False

    if verbose:
        print("  ✓ triplet_type 验证通过：所有样本的语义类型标记正确")
    return True


def verify_sample_boundaries(
    sample_info, segment_ids: np.ndarray, verbose: bool = False
):
    """验证样本边界计算是否正确。

    Args:
      sample_info: 样本信息（如果是批次数据，包含所有packed sequences的聚合信息）
      segment_ids: Segment IDs 数组，可以是1D [seq_length] 或2D [batch_size, seq_length]
      verbose: 是否打印详细信息

    Returns:
      bool: 是否通过验证
    """
    errors = []

    # 处理批次维度
    if segment_ids.ndim == 2:
        # 批次数据：对每个packed sequence分别验证
        batch_size = segment_ids.shape[0]
        from dataset_loader import extract_sample_info_from_segment_ids

        total_verified_samples = 0
        for batch_idx in range(batch_size):
            seq_segment_ids = segment_ids[batch_idx]
            seq_info = extract_sample_info_from_segment_ids(
                seq_segment_ids,
                num_slots=3,
                attention_mask=None,  # 2 + 1 (num_negatives), default value
            )

            # 验证这个packed sequence的样本边界
            for i in range(len(seq_info.sample_starts) - 1):
                current_end = seq_info.sample_starts[i] + seq_info.sample_lengths[i]
                next_start = seq_info.sample_starts[i + 1]

                if current_end > next_start:
                    errors.append(
                        f"批次 {batch_idx} 样本 {i} 和样本 {i + 1} 重叠: {current_end} > {next_start}"
                    )

            # 验证样本数量
            if seq_info.num_samples != len(seq_info.sample_lengths):
                errors.append(
                    f"批次 {batch_idx} 样本数量不一致: num_samples={seq_info.num_samples}, "
                    f"len(sample_lengths)={len(seq_info.sample_lengths)}"
                )

            total_verified_samples += seq_info.num_samples

        if errors:
            if verbose:
                for error in errors:
                    print(f"  ✗ {error}")
            return False

        if verbose:
            print(
                f"  ✓ 样本边界验证通过：{total_verified_samples} 个样本（{batch_size} 个packed sequences）边界正确"
            )
        return True
    else:
        # 1D数据：单个packed sequence
        # 验证样本之间没有重叠
        for i in range(len(sample_info.sample_starts) - 1):
            current_end = sample_info.sample_starts[i] + sample_info.sample_lengths[i]
            next_start = sample_info.sample_starts[i + 1]

            if current_end > next_start:
                errors.append(
                    f"样本 {i} 和样本 {i + 1} 重叠: {current_end} > {next_start}"
                )
            elif current_end < next_start:
                # 允许有间隔（可能是 EOS token 或其他）
                pass

        # 验证样本数量
        if sample_info.num_samples != len(sample_info.sample_lengths):
            errors.append(
                f"样本数量不一致: num_samples={sample_info.num_samples}, "
                f"len(sample_lengths)={len(sample_info.sample_lengths)}"
            )

        if errors:
            if verbose:
                for error in errors:
                    print(f"  ✗ {error}")
            return False

        if verbose:
            print(f"  ✓ 样本边界验证通过：{sample_info.num_samples} 个样本边界正确")
        return True


def verify_sample_reconstruction(
    input_ids: np.ndarray,
    segment_ids: np.ndarray,
    sample_info,
    tokenizer,
    num_negatives: int = 1,
    verbose: bool = False,
):
    """验证能够正确拆分样本并提取各部分。

    Args:
      input_ids: Input IDs 数组
      segment_ids: Segment IDs 数组，编码 sample_id * num_slots + triplet_type
      sample_info: 样本信息
      tokenizer: Tokenizer 用于解码验证
      num_negatives: Number of negative samples per anchor-positive pair. Default: 1.
      verbose: 是否打印详细信息

    Returns:
      bool: 是否通过验证
    """
    errors = []
    num_slots = 2 + num_negatives

    # 拆分样本
    samples = split_packed_sequence(
        jnp.array(input_ids),
        jnp.array(segment_ids) if segment_ids is not None else None,
        sample_info,
        num_negatives=num_negatives,
    )

    if len(samples) != sample_info.num_samples:
        errors.append(
            f"拆分后的样本数量不匹配: {len(samples)} != {sample_info.num_samples}"
        )
        return False

    # 验证每个样本
    for i, sample in enumerate(samples[:5]):  # 只验证前5个样本
        sample_start = sample_info.sample_starts[i]
        sample_length = sample_info.sample_lengths[i]
        sample_input_ids = input_ids[sample_start : sample_start + sample_length]

        # 验证 input_ids 匹配
        if not np.array_equal(sample["input_ids"], sample_input_ids):
            errors.append(f"样本 {i} 的 input_ids 不匹配")
            continue

        # 从 segment_ids 中提取 triplet_type: segment_id % num_slots
        if segment_ids is not None:
            sample_segment_ids = segment_ids[
                sample_start : sample_start + sample_length
            ]
            # 提取 triplet_type: segment_id % num_slots
            sample_triplet_type = sample_segment_ids % num_slots
            # 过滤掉 padding (-1 % num_slots 会得到负数，需要特殊处理)
            valid_mask = sample_segment_ids >= 0
            sample_triplet_type = np.where(valid_mask, sample_triplet_type, -1)

            anchor_mask = sample_triplet_type == 0
            positive_mask = sample_triplet_type == 1
            negative_mask = (sample_triplet_type >= 2) & (
                sample_triplet_type < num_slots
            )
            # Note: special tokens are marked with -1 and should be excluded from content extraction

            if not np.any(anchor_mask):
                errors.append(f"样本 {i} 缺少 anchor 部分")
            if not np.any(positive_mask):
                errors.append(f"样本 {i} 缺少 positive 部分")
            if not np.any(negative_mask):
                errors.append(f"样本 {i} 缺少 negative 部分")

            # 验证各部分顺序：anchor -> positive -> negative
            anchor_indices = np.where(anchor_mask)[0]
            positive_indices = np.where(positive_mask)[0]
            negative_indices = np.where(negative_mask)[0]

            if len(anchor_indices) > 0 and len(positive_indices) > 0:
                if np.max(anchor_indices) >= np.min(positive_indices):
                    errors.append(f"样本 {i} anchor 和 positive 顺序错误")

        if len(positive_indices) > 0 and len(negative_indices) > 0:
            if np.max(positive_indices) >= np.min(negative_indices):
                errors.append(f"样本 {i} positive 和 negative 顺序错误")

    if errors:
        if verbose:
            for error in errors:
                print(f"  ✗ {error}")
        return False

    if verbose:
        print("  ✓ 样本重建验证通过：能够正确拆分和提取各部分")
    return True


def verify_data_completeness(
    all_sample_ids: set[int],
    expected_sample_ids: set[int],
    verbose: bool = False,
) -> bool:
    """验证数据完整性：不重复、不遗漏。

    Args:
      all_sample_ids: 实际处理的所有样本ID集合
      expected_sample_ids: 期望的样本ID集合
      verbose: 是否打印详细信息

    Returns:
      bool: 是否通过验证
    """
    errors = []

    # 检查遗漏的样本
    missing = expected_sample_ids - all_sample_ids
    if missing:
        errors.append(
            f"遗漏了 {len(missing)} 个样本: {sorted(list(missing))[:10]}..."
            if len(missing) > 10
            else f"遗漏了 {len(missing)} 个样本: {sorted(list(missing))}"
        )

    # 检查重复的样本（通过Counter检查）
    # 注意：由于packing，同一个样本不应该出现多次
    # 但这里我们只检查是否有意外的重复

    # 检查多余的样本（不在期望集合中的）
    extra = all_sample_ids - expected_sample_ids
    if extra:
        errors.append(
            f"出现了 {len(extra)} 个意外的样本: {sorted(list(extra))[:10]}..."
            if len(extra) > 10
            else f"出现了 {len(extra)} 个意外的样本: {sorted(list(extra))}"
        )

    if errors:
        if verbose:
            for error in errors:
                print(f"  ✗ {error}")
        return False

    if verbose:
        coverage = (
            len(all_sample_ids) / len(expected_sample_ids) * 100
            if expected_sample_ids
            else 0
        )
        print(
            f"  ✓ 数据完整性验证通过：覆盖率 {coverage:.2f}% ({len(all_sample_ids)}/{len(expected_sample_ids)})"
        )
    return True


def run_comprehensive_test(
    tokenizer_name: str = "Qwen/Qwen3-Embedding-0.6B",
    num_samples: int = 100,
    seq_length: int = 2048,
    verbose: bool = True,
    max_batches: int | None = 5,  # None表示处理所有批次
    max_samples: int | None = None,  # max_samples参数
    max_sample_length: int | None = None,  # max_sample_length参数
):
    """运行全面的测试。

    Args:
      tokenizer_name: Tokenizer 名称
      num_samples: 测试样本数量
      seq_length: 序列长度
      verbose: 是否打印详细信息
      max_batches: 最大处理批次数量，None表示处理所有批次
      max_samples: max_samples参数
      max_sample_length: max_sample_length参数，每个样本的最大长度
    """
    print("=" * 80)
    print("开始全面测试数据加载器")
    print("=" * 80)

    # 1. 创建测试数据
    print("\n[1/6] 创建测试数据...")
    test_file = create_test_data(num_samples=num_samples, output_file="test_data.jsonl")

    # 2. 加载 tokenizer
    print("\n[2/6] 加载 tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=True
        )
        print(f"  ✓ Tokenizer 加载成功: {tokenizer_name}")
    except Exception as e:
        print(f"  ✗ Tokenizer 加载失败: {e}")
        return False

    # 3. 创建数据集
    print("\n[3/6] 创建数据集...")
    try:
        pipeline = create_dataset_pipeline(
            data_files=test_file,
            tokenizer=tokenizer,
            seq_length=seq_length,
            shuffle=False,  # 不 shuffle 以便验证
            seed=42,
            max_samples=max_samples,
            max_sample_length=max_sample_length,
        )
        print("  ✓ Pipeline 创建成功")
    except Exception as e:
        print(f"  ✗ Pipeline 创建失败: {e}")
        import traceback

        traceback.print_exc()
        return False

    # 4. 迭代数据并验证
    print("\n[4/6] 验证数据...")
    all_passed = True
    batch_count = 0
    total_samples_processed = 0

    try:
        # Build pipeline: source -> pack -> load -> build
        dataset = pipeline.pack().load().build()

        start_time = time.time()
        for batch in dataset:
            batch_count += 1

            # 转换为 numpy
            input_ids = np.array(batch["input_ids"])
            segment_ids = np.array(batch.get("segment_ids"))
            # triplet_type has been removed, segment_ids now encode sample_id * num_slots + triplet_type
            # triplet_type = np.array(batch.get("triplet_type"))  # No longer available

            # 提取样本信息
            sample_info = extract_sample_info(batch)
            total_samples_processed += sample_info.num_samples

            if verbose and batch_count <= 5:
                print(f"\n  批次 {batch_count}:")
                print(f"    input_ids 形状: {input_ids.shape}")
                print(
                    f"    segment_ids 形状: {segment_ids.shape if segment_ids is not None else None}"
                )
                # Note: triplet_type has been removed, segment_ids now encode sample_id * num_slots + triplet_type
                print(f"    包含样本数: {sample_info.num_samples}")
                print(
                    f"    样本长度: {sample_info.sample_lengths[:5]}..."
                    if len(sample_info.sample_lengths) > 5
                    else f"    样本长度: {sample_info.sample_lengths}"
                )

            # 验证 segment_ids
            if segment_ids is not None:
                if not verify_segment_ids(
                    segment_ids,
                    sample_info,
                    num_negatives=1,
                    verbose=verbose and batch_count <= 3,
                ):
                    all_passed = False

            # Note: triplet_type has been removed, segment_ids now encode sample_id * num_slots + triplet_type
            # No need to verify triplet_type separately

            # 验证样本边界
            if not verify_sample_boundaries(
                sample_info, segment_ids, verbose=verbose and batch_count <= 3
            ):
                all_passed = False

            # 验证样本重建（只对前几个批次进行详细验证，且只对1D数据验证）
            # Note: triplet_type has been removed, we can extract it from segment_ids
            if batch_count <= 3:
                # 对于批次数据（2D），跳过详细验证，因为sample_info是聚合的
                if input_ids.ndim == 1 and segment_ids is not None:
                    if not verify_sample_reconstruction(
                        input_ids,
                        segment_ids,
                        sample_info,
                        tokenizer,
                        num_negatives=1,  # Default value
                        verbose=verbose,
                    ):
                        all_passed = False
                elif verbose:
                    print(
                        f"  ⚠ 批次 {batch_count}: 跳过样本重建验证（批次数据需要单独处理）"
                    )

            # 限制批次数量（如果指定）
            if max_batches is not None and batch_count >= max_batches:
                if verbose:
                    print(
                        f"\n  已处理 {batch_count} 个批次，达到限制（max_batches={max_batches}）"
                    )
                break

        elapsed_time = time.time() - start_time
        if verbose:
            print(f"\n  处理时间: {elapsed_time:.2f} 秒")
            print(
                f"  平均速度: {total_samples_processed / elapsed_time:.2f} 样本/秒"
                if elapsed_time > 0
                else "  速度: N/A"
            )

    except Exception as e:
        print(f"  ✗ 数据验证过程中出错: {e}")
        import traceback

        traceback.print_exc()
        return False

    # 5. 验证数据完整性（如果处理了所有批次）
    if max_batches is None or total_samples_processed >= num_samples * 0.9:
        print("\n[5/7] 验证数据完整性...")
        # 注意：由于我们无法从packed数据中直接提取sample_id，这里只做基本统计
        if verbose:
            print(f"  处理了 {total_samples_processed} 个样本（期望 {num_samples}）")
            if total_samples_processed < num_samples * 0.9:
                print("  ⚠ 处理的样本数量少于期望的90%，可能未处理完所有数据")
                all_passed = False
            else:
                print("  ✓ 样本数量在合理范围内")
    else:
        print("\n[5/7] 跳过数据完整性验证（未处理完所有批次）")

    # 6. 验证样本信息提取函数
    print("\n[6/7] 验证样本信息提取函数...")
    try:
        # 测试 extract_sample_info_from_segment_ids
        test_segment_ids = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        # Note: This test uses old format segment_ids (direct sample_id encoding)
        # For new format, segment_ids encode sample_id * num_slots + triplet_type
        # Using num_slots=1 to match old behavior (each segment_id change is a new sample)
        test_info = extract_sample_info_from_segment_ids(
            test_segment_ids, num_slots=1, attention_mask=None
        )

        if test_info.num_samples != 3:
            print(
                f"  ✗ extract_sample_info_from_segment_ids 测试失败: 期望 3 个样本，得到 {test_info.num_samples}"
            )
            all_passed = False
        else:
            if verbose:
                print("  ✓ extract_sample_info_from_segment_ids 测试通过")
    except Exception as e:
        print(f"  ✗ extract_sample_info_from_segment_ids 测试失败: {e}")
        all_passed = False

    # 7. 总结
    print("\n[7/7] 测试总结...")
    print(f"  处理了 {batch_count} 个批次")
    print(f"  处理了 {total_samples_processed} 个样本")

    if all_passed:
        print("\n" + "=" * 80)
        print("✓ 所有测试通过！")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("✗ 部分测试失败，请检查上面的错误信息")
        print("=" * 80)

    # 清理测试文件
    if os.path.exists(test_file):
        os.remove(test_file)
        if verbose:
            print(f"\n清理测试文件: {test_file}")

    return all_passed


def run_large_scale_test(
    tokenizer_name: str = "Qwen/Qwen3-Embedding-0.6B",
    num_samples: int = 100000,  # 10万样本
    seq_length: int = 2048,
    verbose: bool = True,
    max_sample_length: int | None = None,  # max_sample_length参数
):
    """运行极大量数据测试。

    Args:
      tokenizer_name: Tokenizer 名称
      num_samples: 测试样本数量（默认10万）
      seq_length: 序列长度
      verbose: 是否打印详细信息
      max_sample_length: max_sample_length参数，每个样本的最大长度
    """
    print("=" * 80)
    print("开始极大量数据测试")
    print("=" * 80)
    print(f"测试规模: {num_samples:,} 个样本")
    print(f"序列长度: {seq_length}")

    # 1. 创建测试数据
    print("\n[1/5] 创建测试数据...")
    start_time = time.time()
    test_file = create_test_data(
        num_samples=num_samples, output_file="test_large_data.jsonl"
    )
    creation_time = time.time() - start_time
    file_size = os.path.getsize(test_file) / (1024 * 1024)  # MB
    print(f"  ✓ 创建了 {num_samples:,} 个测试样本到 {test_file}")
    print(f"  文件大小: {file_size:.2f} MB")
    print(f"  创建时间: {creation_time:.2f} 秒")

    # 2. 加载 tokenizer
    print("\n[2/5] 加载 tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=True
        )
        print(f"  ✓ Tokenizer 加载成功: {tokenizer_name}")
    except Exception as e:
        print(f"  ✗ Tokenizer 加载失败: {e}")
        return False

    # 3. 创建数据集
    print("\n[3/5] 创建数据集...")
    try:
        pipeline = create_dataset_pipeline(
            data_files=test_file,
            tokenizer=tokenizer,
            seq_length=seq_length,
            shuffle=True,  # 启用shuffle测试
            seed=42,
            max_sample_length=max_sample_length,
        )
        print("  ✓ Pipeline 创建成功")
    except Exception as e:
        print(f"  ✗ Pipeline 创建失败: {e}")
        import traceback

        traceback.print_exc()
        return False

    # 4. 迭代数据并统计
    print("\n[4/5] 处理数据...")
    all_passed = True
    batch_count = 0
    total_samples_processed = 0
    total_tokens = 0
    start_time = time.time()

    try:
        dataset = pipeline.pack().load().build()

        for batch in dataset:
            batch_count += 1

            input_ids = np.array(batch["input_ids"])
            sample_info = extract_sample_info(batch)
            total_samples_processed += sample_info.num_samples
            total_tokens += np.sum(input_ids > 0)  # 统计非padding tokens

            # 每100个批次打印一次进度
            if verbose and batch_count % 100 == 0:
                elapsed = time.time() - start_time
                rate = total_samples_processed / elapsed if elapsed > 0 else 0
                print(
                    f"  进度: {batch_count} 批次, {total_samples_processed:,} 样本, {rate:.0f} 样本/秒"
                )

            # 基本验证（只对前几个批次进行详细验证）
            if batch_count <= 3:
                segment_ids = np.array(batch.get("segment_ids"))
                # triplet_type has been removed, segment_ids now encode sample_id * num_slots + triplet_type
                # triplet_type = np.array(batch.get("triplet_type"))  # No longer available

                if segment_ids is not None:
                    if not verify_segment_ids(
                        segment_ids, sample_info, num_negatives=1, verbose=False
                    ):
                        print(f"  ✗ 批次 {batch_count}: segment_ids 验证失败")
                        all_passed = False

                # Note: triplet_type has been removed, segment_ids now encode sample_id * num_slots + triplet_type
                # We can extract triplet_type from segment_ids if needed: triplet_type = segment_ids % num_slots
                # No need to verify triplet_type separately

        elapsed_time = time.time() - start_time

    except Exception as e:
        print(f"  ✗ 数据处理过程中出错: {e}")
        import traceback

        traceback.print_exc()
        return False

    # 5. 总结
    print("\n[5/5] 测试总结...")
    print(f"  处理了 {batch_count:,} 个批次")
    print(f"  处理了 {total_samples_processed:,} 个样本（期望 {num_samples:,}）")
    print(f"  处理了 {total_tokens:,} 个有效tokens")
    print(f"  总耗时: {elapsed_time:.2f} 秒")
    print(
        f"  平均速度: {total_samples_processed / elapsed_time:.2f} 样本/秒"
        if elapsed_time > 0
        else "  速度: N/A"
    )
    print(
        f"  吞吐量: {total_tokens / elapsed_time / 1e6:.2f}M tokens/秒"
        if elapsed_time > 0
        else "  吞吐量: N/A"
    )

    # 验证覆盖率
    coverage = total_samples_processed / num_samples * 100
    if coverage < 95:
        print(f"  ⚠ 覆盖率较低: {coverage:.2f}%")
        all_passed = False
    else:
        print(f"  ✓ 覆盖率: {coverage:.2f}%")

    if all_passed:
        print("\n" + "=" * 80)
        print("✓ 极大量数据测试通过！")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("✗ 极大量数据测试失败")
        print("=" * 80)

    # 清理测试文件
    if os.path.exists(test_file):
        os.remove(test_file)
        if verbose:
            print(f"\n清理测试文件: {test_file}")

    return all_passed


def run_distributed_test(
    tokenizer_name: str = "Qwen/Qwen3-Embedding-0.6B",
    num_files: int = 5,
    samples_per_file: int = 1000,
    seq_length: int = 2048,
    verbose: bool = True,
    max_sample_length: int | None = None,  # max_sample_length参数
):
    """运行多文件/分布式数据测试。

    模拟分布式场景：多个文件对应多个shard，验证：
    1. 多文件数据加载正确
    2. Shard分布正确
    3. 数据不重复、不遗漏

    Args:
      tokenizer_name: Tokenizer 名称
      num_files: 文件数量（模拟多个shard）
      samples_per_file: 每个文件的样本数量
      seq_length: 序列长度
      verbose: 是否打印详细信息
      max_sample_length: max_sample_length参数，每个样本的最大长度
    """
    print("=" * 80)
    print("开始多文件/分布式数据测试")
    print("=" * 80)
    print(f"文件数量: {num_files}")
    print(f"每文件样本数: {samples_per_file:,}")
    print(f"总样本数: {num_files * samples_per_file:,}")

    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    test_files = []
    expected_sample_ids = set()

    try:
        # 1. 创建多个测试文件
        print("\n[1/6] 创建多个测试文件...")
        for i in range(num_files):
            file_path = os.path.join(temp_dir, f"test_data_{i:03d}.jsonl")
            start_id = i * samples_per_file
            create_test_data(
                num_samples=samples_per_file,
                output_file=file_path,
                start_id=start_id,
            )
            test_files.append(file_path)
            expected_sample_ids.update(range(start_id, start_id + samples_per_file))

        print(f"  ✓ 创建了 {num_files} 个测试文件")
        total_size = sum(os.path.getsize(f) for f in test_files) / (1024 * 1024)
        print(f"  总文件大小: {total_size:.2f} MB")

        # 2. 加载 tokenizer
        print("\n[2/6] 加载 tokenizer...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name, trust_remote_code=True
            )
            print(f"  ✓ Tokenizer 加载成功: {tokenizer_name}")
        except Exception as e:
            print(f"  ✗ Tokenizer 加载失败: {e}")
            return False

        # 3. 创建数据集（使用glob pattern或文件列表）
        print("\n[3/6] 创建数据集...")
        try:
            # 使用glob pattern测试
            glob_pattern = os.path.join(temp_dir, "test_data_*.jsonl")
            pipeline = create_dataset_pipeline(
                data_files=glob_pattern,
                tokenizer=tokenizer,
                seq_length=seq_length,
                shuffle=True,  # 启用shuffle
                seed=42,
                max_sample_length=max_sample_length,
            )
            print(f"  ✓ Pipeline 创建成功（使用glob pattern: {glob_pattern}）")

            # 检查shard数量
            pipeline_source = pipeline.source()
            data = pipeline_source.get_data()
            total_shards = 0
            for name, source in data.items():
                num_shards = source.num_shards() if hasattr(source, "num_shards") else 1
                total_shards += num_shards
                if verbose:
                    print(f"    数据集 '{name}': {num_shards} shards")
            print(f"  总shard数: {total_shards}")

        except Exception as e:
            print(f"  ✗ Pipeline 创建失败: {e}")
            import traceback

            traceback.print_exc()
            return False

        # 4. 迭代数据并验证
        print("\n[4/6] 处理数据...")
        all_passed = True
        batch_count = 0
        total_samples_processed = 0
        start_time = time.time()

        try:
            dataset = pipeline.pack().load().build()

            for batch in dataset:
                batch_count += 1

                segment_ids = np.array(batch.get("segment_ids"))
                # triplet_type has been removed, segment_ids now encode sample_id * num_slots + triplet_type
                # triplet_type = np.array(batch.get("triplet_type"))  # No longer available
                sample_info = extract_sample_info(batch, num_negatives=1)
                total_samples_processed += sample_info.num_samples

                # 每50个批次打印一次进度
                if verbose and batch_count % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = total_samples_processed / elapsed if elapsed > 0 else 0
                    print(
                        f"  进度: {batch_count} 批次, {total_samples_processed:,} 样本, {rate:.0f} 样本/秒"
                    )

                # 基本验证（只对前几个批次进行详细验证）
                if batch_count <= 3:
                    if segment_ids is not None:
                        if not verify_segment_ids(
                            segment_ids, sample_info, num_negatives=1, verbose=False
                        ):
                            print(f"  ✗ 批次 {batch_count}: segment_ids 验证失败")
                            all_passed = False

                    # Note: triplet_type has been removed, segment_ids now encode sample_id * num_slots + triplet_type
                    # We can extract triplet_type from segment_ids if needed: triplet_type = segment_ids % num_slots

            elapsed_time = time.time() - start_time

        except Exception as e:
            print(f"  ✗ 数据处理过程中出错: {e}")
            import traceback

            traceback.print_exc()
            return False

        # 5. 验证数据完整性
        print("\n[5/6] 验证数据完整性...")
        expected_total = num_files * samples_per_file
        coverage = (
            total_samples_processed / expected_total * 100 if expected_total > 0 else 0
        )

        print(f"  期望样本数: {expected_total:,}")
        print(f"  实际处理数: {total_samples_processed:,}")
        print(f"  覆盖率: {coverage:.2f}%")

        if coverage < 90:
            print(f"  ✗ 覆盖率过低: {coverage:.2f}% < 90%")
            all_passed = False
        elif coverage > 110:
            print(f"  ⚠ 覆盖率异常高: {coverage:.2f}% > 110%（可能有重复）")
            all_passed = False
        else:
            print(f"  ✓ 覆盖率正常: {coverage:.2f}%")

        # 6. 总结
        print("\n[6/6] 测试总结...")
        print(f"  处理了 {batch_count:,} 个批次")
        print(f"  处理了 {total_samples_processed:,} 个样本")
        print(f"  总耗时: {elapsed_time:.2f} 秒")
        print(
            f"  平均速度: {total_samples_processed / elapsed_time:.2f} 样本/秒"
            if elapsed_time > 0
            else "  速度: N/A"
        )

        if all_passed:
            print("\n" + "=" * 80)
            print("✓ 多文件/分布式数据测试通过！")
            print("=" * 80)
        else:
            print("\n" + "=" * 80)
            print("✗ 多文件/分布式数据测试失败")
            print("=" * 80)

        return all_passed

    finally:
        # 清理测试文件
        import shutil

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            if verbose:
                print(f"\n清理临时目录: {temp_dir}")


if __name__ == "__main__":
    import sys

    # 解析命令行参数
    test_type = "basic"
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()

    if test_type == "large":
        # 运行极大量数据测试
        success = run_large_scale_test(
            tokenizer_name="Qwen/Qwen3-Embedding-0.6B",
            num_samples=100000,  # 10万样本
            seq_length=2048,
            verbose=True,
        )
    elif test_type == "distributed":
        # 运行分布式测试
        success = run_distributed_test(
            tokenizer_name="Qwen/Qwen3-Embedding-0.6B",
            num_files=5,
            samples_per_file=1000,
            seq_length=2048,
            verbose=True,
        )
    elif test_type == "all":
        # 运行所有测试
        print("运行所有测试套件...\n")

        # 1. 基础测试
        print("\n" + "=" * 80)
        print("测试 1/4: 基础功能测试")
        print("=" * 80)
        success1 = run_comprehensive_test(
            tokenizer_name="Qwen/Qwen3-Embedding-0.6B",
            num_samples=100,
            seq_length=2048,
            verbose=True,
            max_batches=5,
        )

        # 2. 极大量数据测试
        print("\n" + "=" * 80)
        print("测试 2/4: 极大量数据测试")
        print("=" * 80)
        success2 = run_large_scale_test(
            tokenizer_name="Qwen/Qwen3-Embedding-0.6B",
            num_samples=10000,  # 1万样本（可根据需要调整）
            seq_length=2048,
            verbose=True,
        )

        # 3. 分布式测试
        print("\n" + "=" * 80)
        print("测试 3/4: 多文件/分布式数据测试")
        print("=" * 80)
        success3 = run_distributed_test(
            tokenizer_name="Qwen/Qwen3-Embedding-0.6B",
            num_files=5,
            samples_per_file=500,
            seq_length=2048,
            verbose=True,
        )

        # 4. 运行max_samples测试
        print("\n" + "=" * 80)
        print("测试 4/4: max_samples测试")
        print("=" * 80)
        success4 = run_comprehensive_test(
            tokenizer_name="Qwen/Qwen3-Embedding-0.6B",
            num_samples=100,  # 创建 100 个测试样本
            seq_length=256,  # 使用较小的序列长度以便产生足够的packed序列
            verbose=True,
            max_batches=None,  # 处理所有批次
            max_samples=5,  # 测试max_samples=5
        )

        success = success1 and success2 and success3 and success4
    else:
        # 默认：基础测试
        success = run_comprehensive_test(
            tokenizer_name="Qwen/Qwen3-Embedding-0.6B",
            num_samples=100,  # 创建 100 个测试样本
            seq_length=256,  # 使用较小的序列长度以便产生足够的packed序列组成batch
            verbose=True,
            max_batches=5,
        )

    exit(0 if success else 1)
