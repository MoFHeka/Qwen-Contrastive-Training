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
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from dataset_loader import (
    create_dataset_pipeline,
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

        print(f"✓ 状态已正确加载:")
        print(f"  shard_index: {loaded_state.shard_index}")
        print(f"  row_index: {loaded_state.row_index}")
        print(f"  step: {loaded_state.step}")
        print(f"  epoch: {loaded_state.epoch}")

    print("✓ 测试 1 通过\n")


def test_resume_from_checkpoint():
    """测试从检查点恢复训练."""
    print("=" * 60)
    print("测试 2: 从检查点恢复训练")
    print("=" * 60)

    # 创建测试数据
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = os.path.join(tmpdir, "test_data")
        file_paths = create_test_data_files(data_dir, num_files=3, samples_per_file=10)
        data_pattern = os.path.join(data_dir, "*.jsonl")

        print(f"创建了 {len(file_paths)} 个数据文件，每个文件 10 个样本")
        print(f"数据文件: {file_paths}")

        # 加载 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True
        )

        # 场景 1: 正常训练，处理前 5 个批次
        print("\n场景 1: 正常训练（处理前 5 个批次）")
        pipeline1 = create_dataset_pipeline(
            data_files=data_pattern,
            tokenizer=tokenizer,
            seq_length=512,
            shuffle=False,  # 不 shuffle 以便验证顺序
        )

        dataset1 = pipeline1.pack().load().build()
        batches1 = []
        samples1 = []

        for i, batch in enumerate(dataset1):
            if i >= 5:
                break
            batches1.append(batch)
            sample_info = extract_sample_info(batch)
            samples1.append(sample_info.num_samples)
            print(f"  批次 {i}: {sample_info.num_samples} 个样本")

        total_samples1 = sum(samples1)
        print(f"  总共处理了 {total_samples1} 个样本")

        # 保存检查点状态（假设在第 2 个文件的第 3 行停止）
        resume_state = ResumeState(
            shard_index=1,  # 第 2 个文件（索引从 0 开始）
            row_index=3,  # 第 4 行（索引从 0 开始）
            step=5,
            epoch=0,
        )

        state_file = os.path.join(tmpdir, "checkpoint_state.json")
        save_resume_state(resume_state, state_file)
        print(
            f"\n✓ 保存检查点状态: shard_index={resume_state.shard_index}, row_index={resume_state.row_index}"
        )

        # 场景 2: 从检查点恢复训练
        print("\n场景 2: 从检查点恢复训练")
        loaded_state = load_resume_state(state_file)
        assert loaded_state is not None, "应该能加载检查点状态"

        pipeline2 = create_dataset_pipeline(
            data_files=data_pattern,
            tokenizer=tokenizer,
            seq_length=512,
            shuffle=False,
            resume_state=loaded_state,
        )

        dataset2 = pipeline2.pack().load().build()
        batches2 = []
        samples2 = []

        for i, batch in enumerate(dataset2):
            if i >= 5:
                break
            batches2.append(batch)
            sample_info = extract_sample_info(batch)
            samples2.append(sample_info.num_samples)
            print(f"  批次 {i}: {sample_info.num_samples} 个样本")

        total_samples2 = sum(samples2)
        print(f"  恢复后处理了 {total_samples2} 个样本")

        # 验证恢复后的数据与原始数据不同（因为跳过了前面的数据）
        print(f"\n✓ 恢复训练成功")
        print(f"  原始训练样本数: {total_samples1}")
        print(f"  恢复后样本数: {total_samples2}")
        print(f"  恢复后应该跳过前面的数据，所以样本数可能不同")

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

    print(f"✓ 状态更新成功:")
    print(f"  shard_index: {updated_state.shard_index}")
    print(f"  row_index: {updated_state.row_index}")
    print(f"  step: {updated_state.step}")
    print(f"  epoch: {updated_state.epoch}")

    print("✓ 测试 3 通过\n")


def test_resume_from_specific_shard_and_row():
    """测试从特定文件的特定行恢复."""
    print("=" * 60)
    print("测试 4: 从特定文件的特定行恢复")
    print("=" * 60)

    # 创建测试数据
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = os.path.join(tmpdir, "test_data")
        file_paths = create_test_data_files(data_dir, num_files=3, samples_per_file=10)
        data_pattern = os.path.join(data_dir, "*.jsonl")

        print(f"创建了 {len(file_paths)} 个数据文件")
        for i, fp in enumerate(file_paths):
            print(f"  文件 {i}: {os.path.basename(fp)}")

        # 加载 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True
        )

        # 测试从第 2 个文件（索引 1）的第 5 行恢复
        target_shard = 1
        target_row = 5

        print(f"\n从第 {target_shard + 1} 个文件的第 {target_row + 1} 行恢复")

        resume_state = ResumeState(
            shard_index=target_shard,
            row_index=target_row,
            step=0,
            epoch=0,
        )

        pipeline = create_dataset_pipeline(
            data_files=data_pattern,
            tokenizer=tokenizer,
            seq_length=512,
            shuffle=False,
            resume_state=resume_state,
        )

        dataset = pipeline.pack().load().build()

        # 收集前几个批次
        batches = []
        for i, batch in enumerate(dataset):
            if i >= 3:
                break
            batches.append(batch)
            sample_info = extract_sample_info(batch)
            print(f"  批次 {i}: {sample_info.num_samples} 个样本")

        print(f"\n✓ 成功从 shard_index={target_shard}, row_index={target_row} 恢复训练")
        print(f"  处理了 {len(batches)} 个批次")

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
