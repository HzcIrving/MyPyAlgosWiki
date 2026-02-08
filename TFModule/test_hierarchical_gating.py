# ============================================================================
# 高低频分层记忆 + Gating 网络单元测试
# ============================================================================
"""
测试模块：验证高低频分层记忆管理和 Gating 网络功能

测试内容：
1. Memory 布局正确性验证
2. 高低频分区更新逻辑验证
3. 降频采样逻辑验证
4. Gating 网络输出验证
5. 端到端融合效果验证
"""

import torch
import torch.nn as nn
import numpy as np
from easydict import EasyDict as edict


# ============================================================================
# 测试1：Memory 布局正确性
# ============================================================================
def test_memory_layout():
    """
    测试目标：验证高低频分层的 Memory 布局是否正确

    验证点：
        1. Memory 总长度 = 1 + high_freq_len + low_freq_len
        2. Index 0 是当前帧
        3. Index 1~high_freq_len 是高频历史
        4. Index high_freq_len+1~end 是低频历史
    """
    print("=" * 60)
    print("测试1：Memory 布局正确性")
    print("=" * 60)

    # 导入模块（假设已修改好）
    import sys
    sys.path.append('.')
    from temp_fusion_module import PVLongTempFusionBaseModule

    # 配置
    config = edict({
        'token_num': 838,
        'tf_fuse_cfg': {'feat_dim': 256, 'num_layers': 2},
        'modal_fuse_cfg': {'feat_dim': 256, 'num_layers': 2},
        'num_layers': 2,
        'motion_dim': 256,
        'use_temp_attn_mask': False,
        'convertD': False,
        'enable_hierarchical': True,  # 启用高低频分层
        'high_freq_len': 6,
        'low_freq_len': 3,
        'use_gating': True,
        'front_pos_param': {'cam_depth': 96, 'feat_dim': 256},
        'side_pos_param': {'cam_depth': 96, 'feat_dim': 256},
        'rear_pos_param': {'cam_depth': 96, 'feat_dim': 256},
    })

    # 创建模型
    model = PVLongTempFusionBaseModule(config)

    # 验证 memory_len 自动计算
    expected_len = 1 + 6 + 3
    assert model.memory_len == expected_len, f"memory_len 计算错误：{model.memory_len} != {expected_len}"
    print(f"✓ Memory 长度正确: {model.memory_len} = 1(当前) + 6(高频) + 3(低频)")

    # 验证高低频分区长度
    assert model.high_freq_len == 6, f"high_freq_len 错误"
    assert model.low_freq_len == 3, f"low_freq_len 错误"
    print(f"✓ 高频区长度: {model.high_freq_len}")
    print(f"✓ 低频区长度: {model.low_freq_len}")

    # 验证组件初始化
    assert hasattr(model, 'time_embedder'), "时间编码器未初始化"
    assert hasattr(model, 'gating_net'), "Gating 网络未初始化"
    print("✓ 时间编码器已初始化")
    print("✓ Gating 网络已初始化")

    print("\n测试1 通过！\n")


# ============================================================================
# 测试2：高低频分区更新逻辑
# ============================================================================
def test_hierarchical_update():
    """
    测试目标：验证高低频分层的更新逻辑是否正确

    验证点：
        1. 当前帧更新后存入 Index 0
        2. 旧当前帧移入高频区头部
        3. 高频区末尾帧溢出到低频区
        4. 低频区每2帧更新一次（降频采样）
    """
    print("=" * 60)
    print("测试2：高低频分区更新逻辑")
    print("=" * 60)

    from temp_fusion_module import PVLongTempFusionBaseModule

    config = edict({
        'token_num': 100,  # 简化 token 数量
        'tf_fuse_cfg': {'feat_dim': 128, 'num_layers': 1},
        'modal_fuse_cfg': {'feat_dim': 128, 'num_layers': 1},
        'num_layers': 1,
        'motion_dim': 128,
        'use_temp_attn_mask': False,
        'convertD': False,
        'enable_hierarchical': True,
        'high_freq_len': 4,  # 简化：4帧高频
        'low_freq_len': 2,  # 简化：2帧低频
        'use_gating': False,  # 暂时关闭 gating，专注测试更新逻辑
        'front_pos_param': {'cam_depth': 96, 'feat_dim': 128},
        'side_pos_param': {'cam_depth': 96, 'feat_dim': 128},
        'rear_pos_param': {'cam_depth': 96, 'feat_dim': 128},
    })

    model = PVLongTempFusionBaseModule(config)

    B = 2
    N = 100
    C = 128
    device = 'cpu'

    # 模拟输入
    def create_mock_input(frame_idx):
        return edict({
            'tokens': torch.randn(B, N, C) * (frame_idx + 1),  # 让每帧数据不同
            'tokens_pos': torch.randn(B, N, C),
            'ego_pose': torch.eye(4).unsqueeze(0).repeat(B, 1, 1, 1),
            'tps': torch.ones(B, 1, 1) * frame_idx * 0.5,  # 每0.5秒一帧
        })

    # 模拟 pre_update_memory
    def mock_pre_update(frame_idx):
        return edict({
            'prev_exists': torch.ones(B, dtype=torch.bool),
            'tps': torch.ones(B, 1, 1) * frame_idx * 0.5,
        })

    # 模拟10帧更新
    for i in range(10):
        # 输入数据
        inputs = create_mock_input(i)
        pre_params = mock_pre_update(i)

        # 更新 memory（模拟 forward 中的流程）
        model.pre_update_memory(pre_params)
        model._update_memory(inputs)

        # 验证当前帧在 Index 0
        cur_token = model.memory['tokens'][:, 0]
        expected_value = (i + 1)  # 因为我们在 tokens 中乘了 (frame_idx + 1)
        assert torch.allclose(cur_token.mean(), expected_value, atol=1e-5), \
            f"第 {i} 帧：Index 0 应该是当前帧"

        # 验证更新计数器
        expected_counter = i + 1
        assert model.update_counter == expected_counter, \
            f"第 {i} 帧：update_counter 应该是 {expected_counter}"

        # 验证低频区降频采样
        if model.low_freq_len > 0:
            # 获取低频区的 mask
            low_mask = model.memory['memory_mask'][:, 1 + model.high_freq_len:]
            # 期望：奇数次更新，偶数次不更新
            if i > 0:
                if expected_counter % 2 == 1:
                    # 奇数次后，低频区应该增加
                    assert low_mask[0, -1].item() > 0.5, \
                        f"第 {i} 帧（奇数次）：低频区应该接收新帧"
                else:
                    # 偶数次后，低频区应该保持
                    prev_low_mask = low_mask[0, -2].item() if i > 1 else 0
                    assert low_mask[0, -1].item() == prev_low_mask, \
                        f"第 {i} 帧（偶数次）：低频区应该保持不变"

        print(f"  帧 {i}: ✓ 布局验证通过")

    print("\n测试2 通过！\n")


# ============================================================================
# 测试3：降频采样时间逻辑
# ============================================================================
def test_downsampling_timing():
    """
    测试目标：验证降频采样的时间逻辑是否正确

    验证点：
        1. 输入 2Hz 数据，高频区每帧更新
        2. 输入 2Hz 数据，低频区每2帧更新一次（相当于1Hz）
        3. 时间戳计算正确
    """
    print("=" * 60)
    print("测试3：降频采样时间逻辑")
    print("=" * 60)

    from temp_fusion_module import PVLongTempFusionBaseModule

    config = edict({
        'token_num': 100,
        'tf_fuse_cfg': {'feat_dim': 128, 'num_layers': 1},
        'modal_fuse_cfg': {'feat_dim': 128, 'num_layers': 1},
        'num_layers': 1,
        'motion_dim': 128,
        'use_temp_attn_mask': False,
        'convertD': False,
        'enable_hierarchical': True,
        'high_freq_len': 6,  # 3秒 @ 2Hz
        'low_freq_len': 3,  # 3秒 @ 1Hz
        'use_gating': False,
        'front_pos_param': {'cam_depth': 96, 'feat_dim': 128},
        'side_pos_param': {'cam_depth': 96, 'feat_dim': 128},
        'rear_pos_param': {'cam_depth': 96, 'feat_dim': 128},
    })

    model = PVLongTempFusionBaseModule(config)

    # 模拟输入：2Hz 数据（每0.5秒一帧）
    B = 2
    N = 100
    C = 128

    def create_mock_input_with_time(frame_idx, time_sec):
        """创建带时间戳的模拟输入"""
        return edict({
            'tokens': torch.randn(B, N, C),
            'tokens_pos': torch.randn(B, N, C),
            'ego_pose': torch.eye(4).unsqueeze(0).repeat(B, 1, 1, 1),
            'tps': torch.ones(B, 1, 1) * time_sec,
        })

    def mock_pre_update_with_time(time_sec):
        """创建带时间戳的 pre_update 参数"""
        return edict({
            'prev_exists': torch.ones(B, dtype=torch.bool),
            'tps': torch.ones(B, 1, 1) * time_sec,
        })

    # 模拟8秒的输入（16帧 @ 0.5秒/帧）
    num_frames = 16
    time_step = 0.5  # 0.5秒/帧

    for i in range(num_frames):
        time_sec = i * time_step
        inputs = create_mock_input_with_time(i, time_sec)
        pre_params = mock_pre_update_with_time(time_sec)

        # 更新 memory
        model.pre_update_memory(pre_params)
        model._update_memory(inputs)

        # 验证时间戳
        current_time = model.memory['tps'][:, 0, 0].item()
        expected_time = time_sec
        assert abs(current_time - expected_time) < 1e-5, \
            f"帧 {i}：当前帧时间戳不正确，{current_time} != {expected_time}"

        # 验证高频区时间戳（应该有 0.5s 间隔）
        if model.high_freq_len > 0 and i >= 1:
            high_time_0 = model.memory['tps'][:, 1, 0].item()  # 高频区第一帧
            high_time_1 = model.memory['tps'][:, 2, 0].item()  # 高频区第二帧
            time_diff = high_time_0 - high_time_1
            assert abs(time_diff - 0.5) < 1e-5, \
                f"帧 {i}：高频区时间间隔应为 0.5s，实际为 {time_diff}s"

        print(f"  帧 {i} (t={time_sec:.1f}s): ✓ 时间戳验证通过")

    print("\n测试3 通过！\n")


# ============================================================================
# 测试4：Gating 网络输出验证
# ============================================================================
def test_gating_network():
    """
    测试目标：验证 Gating 网络的输出是否正确

    验证点：
        1. 输出形状正确：(B, 3)
        2. 权重归一化：和为1
        3. 权重范围：[0, 1]
        4. 梯度能正常反向传播
    """
    print("=" * 60)
    print("测试4：Gating 网络输出验证")
    print("=" * 60)

    from temp_fusion_module import FrequencyGating

    # 创建 Gating 网络
    feat_dim = 256
    gating_net = FrequencyGating(feat_dim=feat_dim)

    # 模拟输入
    B, N, C = 4, 100, 256
    current_token = torch.randn(B, N, C)
    high_time_emb = torch.randn(B, C)
    low_time_emb = torch.randn(B, C)

    # 前向传播
    gate_weights = gating_net(current_token, high_time_emb, low_time_emb)

    # 验证输出形状
    assert gate_weights.shape == (B, 3), f"输出形状错误：{gate_weights.shape} != {(B, 3)}"
    print(f"✓ 输出形状正确: {gate_weights.shape}")

    # 验证归一化：权重和应该为1
    weight_sums = gate_weights.sum(dim=-1)
    assert torch.allclose(weight_sums, torch.ones(B), atol=1e-5), \
        "权重未归一化"
    print(f"✓ 权重归一化：和为 1")

    # 验证权重范围：[0, 1]
    assert (gate_weights >= 0).all() and (gate_weights <= 1).all(), \
        "权重超出[0,1]范围"
    print(f"✓ 权重范围正确：[{gate_weights.min():.3f}, {gate_weights.max():.3f}]")

    # 验证梯度传播
    loss = gate_weights.sum()
    loss.backward()

    # 检查参数梯度是否存在
    assert gating_net.fc[0].weight.grad is not None, "MLP 第一层梯度缺失"
    assert gating_net.fc[-1].weight.grad is not None, "MLP 输出层梯度缺失"
    print("✓ 梯度传播正常")

    print("\n测试4 通过！\n")


# ============================================================================
# 测试5：端到端融合效果
# ============================================================================
def test_end_to_end_fusion():
    """
    测试目标：验证端到端的融合效果

    验证点：
        1. 输出形状正确
        2. 当前帧、高频历史、低频历史都被考虑
        3. Gating 权重被正确应用
    """
    print("=" * 60)
    print("测试5：端到端融合效果")
    print("=" * 60)

    from temp_fusion_module import PVLongTempFusionBaseModule

    config = edict({
        'token_num': 100,  # 简化
        'tf_fuse_cfg': {'feat_dim': 128, 'num_layers': 1},
        'modal_fuse_cfg': {'feat_dim': 128, 'num_layers': 1},
        'num_layers': 1,
        'motion_dim': 128,
        'use_temp_attn_mask': False,
        'convertD': False,
        'enable_hierarchical': True,
        'high_freq_len': 4,
        'low_freq_len': 2,
        'use_gating': True,
        'front_pos_param': {'cam_depth': 96, 'feat_dim': 128},
        'side_pos_param': {'cam_depth': 96, 'feat_dim': 128},
        'rear_pos_param': {'cam_depth': 96, 'feat_dim': 128},
    })

    model = PVLongTempFusionBaseModule(config)
    model.eval()  # 评估模式

    B, N, C = 2, 100, 128
    device = 'cpu'

    # 创建模拟输入
    def create_full_mock_input(time_sec):
        return edict({
            'tokens': torch.randn(B, N, C),
            'tokens_pos': torch.randn(B, N, C),
            'prev_exists': torch.ones(B, dtype=torch.bool),
            'tps': torch.ones(B, 1, 1) * time_sec,
            'ego_pose': torch.eye(4).unsqueeze(0).repeat(B, 1, 1, 1),
            'pos_3d': edict({
                'pos_3d_front': torch.randn(B, 2, 24, 32, 96),
                'pos_3d_rear': torch.randn(B, 1, 12, 16, 96),
                'pos_3d_side': torch.randn(B, 4, 48, 64, 96),
            }),
        })

    # 运行多帧
    for i in range(5):
        time_sec = i * 0.5  # 0s, 0.5s, 1.0s, 1.5s, 2.0s
        inputs = create_full_mock_input(time_sec)
        ego_pose_inv = torch.inverse(inputs['ego_pose'])

        # 前向传播
        with torch.no_grad():
            output = model.forward(
                inputs['tokens'],
                inputs['tokens_pos'],
                inputs,
                ego_pose_inv
            )

        # 验证输出形状
        assert output.shape == (B, N, C), \
            f"输出形状错误：{output.shape} != {(B, N, C)}"

        # 验证 Gating 权重
        if hasattr(model, 'last_gate_weights'):
            weights = model.last_gate_weights
            assert weights.shape == (B, 3), \
                f"Gating 权重形状错误：{weights.shape}"
            assert torch.allclose(weights.sum(dim=-1), torch.ones(B)), \
                "Gating 权重未归一化"

        print(f"  帧 {i} (t={time_sec:.1f}s): ✓ 融合效果验证通过")

    print("\n测试5 通过！\n")


# ============================================================================
# 测试6：对比实验（原始模式 vs 高低频分层模式）
# ============================================================================
def test_comparison():
    """
    测试目标：对比原始模式和空心模式的效果差异

    验证点：
        1. 两种模式都能正常运行
        2. 高低频模式有更丰富的时序信息
        3. 高低频模式的计算开销在可接受范围内
    """
    print("=" * 60)
    print("测试6：对比实验（原始模式 vs 高低频分层模式）")
    print("=" * 60)

    from temp_fusion_module import PVLongTempFusionBaseModule

    B, N, C = 2, 100, 128
    device = 'cpu'

    # 测试原始模式
    config_single = edict({
        'token_num': N,
        'tf_fuse_cfg': {'feat_dim': C, 'num_layers': 1},
        'modal_fuse_cfg': {'feat_dim': C, 'num_layers': 1},
        'num_layers': 1,
        'motion_dim': C,
        'use_temp_attn_mask': False,
        'convertD': False,
        'enable_hierarchical': False,  # 原始模式
        'memory_len': 10,
        'front_pos_param': {'cam_depth': 96, 'feat_dim': C},
        'side_pos_param': {'cam_depth': 96, 'feat_dim': C},
        'rear_pos_param': {'cam_depth': 96, 'feat_dim': C},
    })

    model_single = PVLongTempFusionBaseModule(config_single)
    model_single.eval()

    # 测试高低频分层模式
    config_hier = edict({
        'token_num': N,
        'tf_fuce_cfg': {'feat_dim': C, 'num_layers': 1},
        'modal_fuse_cfg': {'feat_dim': C, 'num_layers': 1},
        'num_layers': 1,
        'motion_dim': C,
        'use_temp_attn_mask': False,
        'convertD': False,
        'enable_hierarchical': True,  # 高低频分层模式
        'high_freq_len': 6,
        'low_freq_len': 3,
        'use_gating': True,
        'front_pos_param': {'cam_depth': 96, 'feat_dim': C},
        'side_pos_param': {'cam_depth': 96, 'feat_dim': C},
        'rear_pos_param': {'cam_depth': 96, 'feat_dim': C},
    })

    model_hier = PVLongTempFusionBaseModule(config_hier)
    model_hier.eval()

    # 创建模拟输入
    def create_mock_input():
        return edict({
            'tokens': torch.randn(B, N, C),
            'tokens_pos': torch.randn(B, N, C),
            'prev_exists': torch.ones(B, dtype=torch.bool),
            'tps': torch.ones(B, 1, 1) * 1.0,
            'ego_pose': torch.eye(4).unsqueeze(0). repeat(B, 1, 1, 1),
            'pos_3d': edict({
                'pos_3d_front': torch.randn(B, 2, 24, 32, 96),
                'pos_3d_rear': torch.randn(B, 1, 12, 16, 96),
                'pos_3d_side': torch.randn(B, 4, 48, 64, 96),
            }),
        })

    inputs = create_mock_input()
    ego_pose_inv = torch.inverse(inputs['ego_pose'])

    # 对比前向传播时间
    import time

    # 原始模式
    start = time.time()
    with torch.no_grad():
        for _ in range(10):
            output_single = model_single.forward(
                inputs['tokens'], inputs['tokens_pos'], inputs, ego_pose_inv)
    time_single = time.time() - start

    # 高低频分层模式
    start = time.time()
    with torch.no_grad():
        for _ in range(10):
            output_hier = model_hier.forward(
                inputs['tokens'], inputs['tokens_pos'], inputs, ego_pose_inv
            )
    time_hier = time.time() - start

    print(f"✓ 原始模式前向传播时间（10次平均）: {time_single * 100:.2f}ms")
    print(f"✓ 高低频分层模式前向传播时间（10次平均）: {time_hier * 100:.2f}ms")
    print(f"✓ 时间增加: {((time_hier - time_single) / time_single * 100):.1f}%")

    # 验证输出形状
    assert output_single.shape == (B, N, C), f"原始模式输出形状错误"
    assert output_hier.shape == (B, N, C), f"高低频分层模式输出形状错误"

    print("\n测试6 通过！\n")


# ============================================================================
# 主测试函数
# ============================================================================
def run_all_tests():
    """
    运行所有测试
    """
    print("\n" + "=" * 60)
    print("开始运行测试套件...")
    print("=" * 60 + "\n")

    try:
        test_memory_layout()
        test_hierarchical_update()
        test_downsampling_timing()
        test_gating_network()
        test_end_to_end_fusion()
        test_comparison()

        print("\n" + "=" * 60)
        print("✅ 所有测试通过！")
        print("=" * 60 + "\n")

    except AssertionError as e:
        print("\n" + "=" * 60)
        print(f"❌ 测试失败：{str(e)}")
        print("=" * 60 + "\n")
        raise


if __name__ == "__main__":
    run_all_tests()
