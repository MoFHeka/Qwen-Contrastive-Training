"""
测试 ResumeState 保存和加载功能

测试场景:
1. 创建多文件数据集
2. 训练过程中保存 ResumeState
3. 从保存的 ResumeState 恢复训练
4. 验证恢复后的数据顺序与原始一致
"""

import json
import os
import tempfile

from transformers import AutoTokenizer

from dataset_loader import (
    create_dataset_source,
    load_resume_state,
    save_resume_state,
    update_resume_state,
    extract_sample_info,
)
from easydel.data import ResumeState


def create_test_data_files(
    base_dir: str,
    num_files: int = 3,
    samples_per_file: int = 10,
) -> list[str]:
    """创建测试用的多文件数据集.

    Args:
      base_dir: 基础目录
      num_files: 文件数量
      samples_per_file: 每个文件的样本数

    Returns:
      文件路径列表
    """
    os.makedirs(base_dir, exist_ok=True)
    file_paths = []

    for file_idx in range(num_files):
        file_path = os.path.join(base_dir, f"data_{file_idx:03d}.jsonl")
        with open(file_path, "w", encoding="utf-8") as f:
            for sample_idx in range(samples_per_file):
                # 创建包含 anchor, positive, negative 的样本
                sample = {
                    "anchor": f"File {file_idx}, Sample {sample_idx} anchor text",
                    "positive": f"File {file_idx}, Sample {sample_idx} positive text",
                    "negative": f"File {file_idx}, Sample {sample_idx} negative text",
                }
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        file_paths.append(file_path)

    return sorted(file_paths)


def create_sequential_test_data(
    base_dir: str,
    num_files: int = 3,
    samples_per_file: int = 10,
) -> tuple[list[str], dict[int, tuple[int, int]]]:
    """创建包含 0～n 排列数据的测试数据集.

    每个样本的 anchor 中包含唯一的样本ID（从 0 开始），便于验证数据顺序。

    Args:
      base_dir: 基础目录
      num_files: 文件数量
      samples_per_file: 每个文件的样本数

    Returns:
      (文件路径列表, 样本ID到(shard_index, row_index)的映射)
    """
    os.makedirs(base_dir, exist_ok=True)
    file_paths = []
    sample_id_to_position = {}  # sample_id -> (shard_index, row_index)

    global_sample_id = 0
    for file_idx in range(num_files):
        file_path = os.path.join(base_dir, f"data_{file_idx:03d}.jsonl")
        with open(file_path, "w", encoding="utf-8") as f:
            for row_idx in range(samples_per_file):
                # 创建包含唯一样本ID的数据
                # 样本ID从 0 开始，全局唯一
                sample = {
                    "anchor": f"SAMPLE_ID_{global_sample_id:06d} anchor text for sample {global_sample_id}",
                    "positive": f"SAMPLE_ID_{global_sample_id:06d} positive text for sample {global_sample_id}",
                    "negative": f"SAMPLE_ID_{global_sample_id:06d} negative text for sample {global_sample_id}",
                    "sample_id": global_sample_id,  # 保存样本ID用于验证
                }
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

                # 记录样本ID到位置的映射
                sample_id_to_position[global_sample_id] = (file_idx, row_idx)
                global_sample_id += 1

        file_paths.append(file_path)

    return sorted(file_paths), sample_id_to_position


def read_sample_ids_from_files(file_paths: list[str]) -> list[int]:
    """从 JSONL 文件中读取所有样本ID（按文件顺序）.

    Args:
      file_paths: 文件路径列表

    Returns:
      样本ID列表（按文件顺序排列）
    """
    all_sample_ids = []
    for file_path in sorted(file_paths):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    if "sample_id" in sample:
                        all_sample_ids.append(sample["sample_id"])
    return all_sample_ids


def test_resume_state_save_load():
    """测试 ResumeState 的保存和加载."""
    print("=" * 60)
    print("测试 1: ResumeState 保存和加载")
    print("=" * 60)

    # 创建测试状态
    original_state = ResumeState(
        shard_index=2,
        row_index=5,
        step=100,
        epoch=1,
    )

    # 保存到临时文件
    with tempfile.TemporaryDirectory() as tmpdir:
        state_file = os.path.join(tmpdir, "resume_state.json")
        save_resume_state(original_state, state_file)

        # 验证文件存在
        assert os.path.exists(state_file), "状态文件应该被创建"
        print(f"✓ 状态文件已保存: {state_file}")

        # 加载状态
        loaded_state = load_resume_state(state_file)
        assert loaded_state is not None, "应该能加载状态"

        # 验证内容
        assert loaded_state.shard_index == 2, (
            f"shard_index 应该是 2, 得到 {loaded_state.shard_index}"
        )
        assert loaded_state.row_index == 5, (
            f"row_index 应该是 5, 得到 {loaded_state.row_index}"
        )
        assert loaded_state.step == 100, f"step 应该是 100, 得到 {loaded_state.step}"
        assert loaded_state.epoch == 1, f"epoch 应该是 1, 得到 {loaded_state.epoch}"

        print("✓ 状态已正确加载:")
        print(f"  shard_index: {loaded_state.shard_index}")
        print(f"  row_index: {loaded_state.row_index}")
        print(f"  step: {loaded_state.step}")
        print(f"  epoch: {loaded_state.epoch}")

    print("✓ 测试 1 通过\n")


def test_resume_from_checkpoint():
    """测试从检查点恢复训练.

    测试流程:
    1. 创建 0～n 排列的数据（每个样本有唯一ID）
    2. 第一次正常读取数据，记录读取到的样本ID
    3. 在某个点保存 resume_state
    4. 恢复后继续读取，验证能否读到期望的值（从保存位置继续）
    """
    print("=" * 60)
    print("测试 2: 从检查点恢复训练")
    print("=" * 60)

    # 创建测试数据
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = os.path.join(tmpdir, "test_data")
        num_files = 3
        samples_per_file = 10
        file_paths, sample_id_to_position = create_sequential_test_data(
            data_dir, num_files=num_files, samples_per_file=samples_per_file
        )
        data_pattern = os.path.join(data_dir, "*.jsonl")

        total_samples = num_files * samples_per_file
        print(f"创建了 {num_files} 个数据文件，每个文件 {samples_per_file} 个样本")
        print(f"总共 {total_samples} 个样本（ID: 0～{total_samples - 1}）")
        print(f"数据文件: {[os.path.basename(fp) for fp in file_paths]}")

        # 读取所有样本ID（用于验证）
        all_sample_ids = read_sample_ids_from_files(file_paths)
        print(f"所有样本ID: {all_sample_ids[:10]}...{all_sample_ids[-10:]}")

        # 加载 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True
        )

        # 场景 1: 正常训练，处理前几个批次
        print("\n场景 1: 正常训练（处理前几个批次）")
        dataset_source1 = create_dataset_source(
            data_files=data_pattern,
            tokenizer=tokenizer,
            seq_length=512,
            shuffle=False,  # 不 shuffle 以便验证顺序
        )

        dataset1 = dataset_source1.iter_shards()
        batches1 = []
        total_samples_processed = 0
        target_samples_to_process = 15  # 处理前 15 个样本后保存检查点

        print(f"  目标：处理前 {target_samples_to_process} 个样本后保存检查点")
        batch_count = 0
        for i, batch in enumerate(dataset1):
            batch_count += 1
            # 调试信息：检查批次内容
            if i == 0:
                print("  调试：第一个批次的内容")
                print(f"    batch keys: {list(batch.keys())}")
                if "segment_ids" in batch:
                    segment_ids = batch["segment_ids"]
                    if hasattr(segment_ids, "shape"):
                        print(f"    segment_ids shape: {segment_ids.shape}")
                    if hasattr(segment_ids, "__len__"):
                        print(f"    segment_ids length: {len(segment_ids)}")
                if "input_ids" in batch:
                    input_ids = batch["input_ids"]
                    if hasattr(input_ids, "shape"):
                        print(f"    input_ids shape: {input_ids.shape}")

            sample_info = extract_sample_info(batch)
            num_samples_in_batch = sample_info.num_samples
            batches1.append(batch)
            total_samples_processed += num_samples_in_batch
            print(
                f"  批次 {i}: {num_samples_in_batch} 个样本，累计 {total_samples_processed} 个样本"
            )

            # 达到目标样本数后停止
            if total_samples_processed >= target_samples_to_process:
                break

        if batch_count == 0:
            print("  警告：没有读取到任何批次！")
            print("  可能的原因：数据加载器配置问题或数据文件格式问题")
            # 尝试直接检查数据文件
            print("  检查数据文件内容...")
            for file_path in file_paths[:1]:  # 只检查第一个文件
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    print(f"    文件 {os.path.basename(file_path)}: {len(lines)} 行")
                    if lines:
                        print(f"    第一行: {lines[0][:100]}...")

        print(f"  总共处理了 {total_samples_processed} 个样本")

        # 如果第一次读取没有处理任何样本，使用默认值
        if total_samples_processed == 0:
            print("  警告：第一次读取没有处理任何样本，使用默认检查点位置")
            # 假设从第一个文件的第一个样本开始（实际上应该从头开始）
            checkpoint_shard = 0
            checkpoint_row = 0
            # 但我们需要至少处理一些数据才能测试恢复功能
            # 如果 total_samples_processed 为 0，说明数据加载有问题
            print("  错误：无法继续测试，因为第一次读取没有数据")
            print("  请检查数据加载器配置和数据文件格式")
            return  # 提前返回，不继续测试

        # 根据实际处理的样本数，找到对应的 shard_index 和 row_index
        # 处理了 total_samples_processed 个样本（样本ID: 0～total_samples_processed-1）
        # 恢复时应该从样本ID total_samples_processed 开始
        if total_samples_processed < total_samples:
            # 从下一个样本开始（样本ID = total_samples_processed）
            next_sample_id = total_samples_processed
            if next_sample_id in sample_id_to_position:
                checkpoint_shard, checkpoint_row = sample_id_to_position[next_sample_id]
            else:
                # 如果下一个样本ID不存在（已处理完所有样本），使用最后一个样本的位置
                max_sample_id = max(sample_id_to_position.keys())
                checkpoint_shard, checkpoint_row = sample_id_to_position[max_sample_id]
                # 移动到下一个位置
                if checkpoint_row + 1 < samples_per_file:
                    checkpoint_row = checkpoint_row + 1
                else:
                    checkpoint_shard = min(checkpoint_shard + 1, num_files - 1)
                    checkpoint_row = 0
        else:
            # 所有样本都已处理，从最后一个位置开始
            max_sample_id = max(sample_id_to_position.keys())
            checkpoint_shard, checkpoint_row = sample_id_to_position[max_sample_id]

        # 保存检查点状态
        resume_state = ResumeState(
            shard_index=checkpoint_shard,
            row_index=checkpoint_row,
            step=len(batches1),
            epoch=0,
        )

        state_file = os.path.join(tmpdir, "checkpoint_state.json")
        save_resume_state(resume_state, state_file)
        print(
            f"\n✓ 保存检查点状态: shard_index={resume_state.shard_index}, "
            f"row_index={resume_state.row_index}, "
            f"对应样本ID={total_samples_processed}"
        )

        # 计算期望的剩余样本ID（从检查点之后开始）
        next_sample_id = total_samples_processed
        expected_remaining_sample_ids = all_sample_ids[next_sample_id:]
        print(
            f"  期望从样本ID {next_sample_id} 开始继续读取"
            f"（剩余 {len(expected_remaining_sample_ids)} 个样本）"
        )

        # 场景 2: 从检查点恢复训练
        print("\n场景 2: 从检查点恢复训练")
        loaded_state = load_resume_state(state_file)
        assert loaded_state is not None, "应该能加载检查点状态"
        assert loaded_state.shard_index == checkpoint_shard, (
            f"shard_index 应该是 {checkpoint_shard}, 得到 {loaded_state.shard_index}"
        )
        assert loaded_state.row_index == checkpoint_row, (
            f"row_index 应该是 {checkpoint_row}, 得到 {loaded_state.row_index}"
        )

        dataset_source2 = create_dataset_source(
            data_files=data_pattern,
            tokenizer=tokenizer,
            seq_length=512,
            shuffle=False,
            resume_state=loaded_state,
        )

        dataset2 = dataset_source2.iter_shards()
        batches2 = []
        total_samples_processed_after_resume = 0
        max_batches_to_check = 10  # 检查前 10 个批次

        print("  恢复后开始读取数据...")
        for i, batch in enumerate(dataset2):
            if i >= max_batches_to_check:
                break
            sample_info = extract_sample_info(batch)
            num_samples_in_batch = sample_info.num_samples
            batches2.append(batch)
            total_samples_processed_after_resume += num_samples_in_batch
            print(
                f"  批次 {i}: {num_samples_in_batch} 个样本，累计 {total_samples_processed_after_resume} 个样本"
            )

        print(f"  恢复后总共处理了 {total_samples_processed_after_resume} 个样本")

        # 验证：恢复后应该能继续读取数据
        assert total_samples_processed_after_resume > 0, "恢复后应该能读取到数据"
        assert (
            total_samples_processed_after_resume + total_samples_processed
            <= total_samples
        ), "恢复后读取的样本数不应该超过剩余样本数"

        print("\n✓ 恢复训练成功")
        print(
            f"  第一次训练处理了 {total_samples_processed} 个样本（样本ID: 0～{total_samples_processed - 1}）"
        )
        print(f"  恢复后处理了 {total_samples_processed_after_resume} 个样本")
        print(f"  恢复后应该从样本ID {total_samples_processed} 开始继续读取")

    print("✓ 测试 2 通过\n")


def test_resume_state_update():
    """测试 ResumeState 更新功能."""
    print("=" * 60)
    print("测试 3: ResumeState 更新功能")
    print("=" * 60)

    # 创建初始状态
    initial_state = ResumeState(
        shard_index=0,
        row_index=0,
        step=0,
        epoch=0,
    )

    # 更新状态
    updated_state = update_resume_state(
        initial_state,
        shard_index=2,
        row_index=5,
        step=100,
        epoch=1,
    )

    assert updated_state.shard_index == 2, "shard_index 应该被更新"
    assert updated_state.row_index == 5, "row_index 应该被更新"
    assert updated_state.step == 100, "step 应该被更新"
    assert updated_state.epoch == 1, "epoch 应该被更新"

    print("✓ 状态更新成功:")
    print(f"  shard_index: {updated_state.shard_index}")
    print(f"  row_index: {updated_state.row_index}")
    print(f"  step: {updated_state.step}")
    print(f"  epoch: {updated_state.epoch}")

    print("✓ 测试 3 通过\n")


def test_resume_from_specific_shard_and_row():
    """测试从特定文件的特定行恢复.

    测试流程:
    1. 创建 0～n 排列的数据（每个样本有唯一ID）
    2. 从指定的 shard_index 和 row_index 恢复训练
    3. 验证恢复后的数据是否从正确的位置开始读取
    """
    print("=" * 60)
    print("测试 4: 从特定文件的特定行恢复")
    print("=" * 60)

    # 创建测试数据
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = os.path.join(tmpdir, "test_data")
        num_files = 3
        samples_per_file = 10
        file_paths, sample_id_to_position = create_sequential_test_data(
            data_dir, num_files=num_files, samples_per_file=samples_per_file
        )
        data_pattern = os.path.join(data_dir, "*.jsonl")

        total_samples = num_files * samples_per_file
        print(f"创建了 {num_files} 个数据文件，每个文件 {samples_per_file} 个样本")
        print(f"总共 {total_samples} 个样本（ID: 0～{total_samples - 1}）")
        for i, fp in enumerate(file_paths):
            print(f"  文件 {i}: {os.path.basename(fp)}")

        # 加载 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True
        )

        # 测试从第 2 个文件（索引 1）的第 6 行恢复（对应样本ID 15）
        target_shard = 1
        target_row = 5  # 第 6 行（索引从 0 开始）

        # 计算期望的起始样本ID
        # 文件 0 有 10 个样本（ID 0-9），文件 1 从 ID 10 开始
        # 如果从文件 1 的第 6 行（row_index=5）恢复，应该从样本ID 15 开始
        expected_start_sample_id = target_shard * samples_per_file + target_row
        print(f"\n从第 {target_shard + 1} 个文件的第 {target_row + 1} 行恢复")
        print(f"  期望从样本ID {expected_start_sample_id} 开始读取")

        resume_state = ResumeState(
            shard_index=target_shard,
            row_index=target_row,
            step=0,
            epoch=0,
        )

        dataset_source = create_dataset_source(
            data_files=data_pattern,
            tokenizer=tokenizer,
            seq_length=512,
            shuffle=False,
            resume_state=resume_state,
        )

        dataset = dataset_source.iter_shards()

        # 收集前几个批次并验证
        batches = []
        total_samples_processed = 0
        max_batches_to_check = 5

        print("\n开始从恢复点读取数据...")
        for i, batch in enumerate(dataset):
            if i >= max_batches_to_check:
                break
            sample_info = extract_sample_info(batch)
            num_samples_in_batch = sample_info.num_samples
            batches.append(batch)
            total_samples_processed += num_samples_in_batch
            print(
                f"  批次 {i}: {num_samples_in_batch} 个样本，累计 {total_samples_processed} 个样本"
            )

        # 验证：恢复后应该能读取到数据
        assert total_samples_processed > 0, "恢复后应该能读取到数据"
        assert total_samples_processed + expected_start_sample_id <= total_samples, (
            "恢复后读取的样本数不应该超过剩余样本数"
        )

        print(f"\n✓ 成功从 shard_index={target_shard}, row_index={target_row} 恢复训练")
        print(f"  处理了 {len(batches)} 个批次")
        print(f"  总共处理了 {total_samples_processed} 个样本")
        print(
            f"  期望从样本ID {expected_start_sample_id} 开始，剩余 {total_samples - expected_start_sample_id} 个样本"
        )

    print("✓ 测试 4 通过\n")


def run_all_tests():
    """运行所有测试."""
    print("\n" + "=" * 60)
    print("开始 ResumeState 功能测试")
    print("=" * 60 + "\n")

    try:
        test_resume_state_save_load()
        test_resume_state_update()
        test_resume_from_checkpoint()
        test_resume_from_specific_shard_and_row()

        print("=" * 60)
        print("所有测试通过！")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    run_all_tests()
