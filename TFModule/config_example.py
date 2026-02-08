# ============================================================================
# 高低频分层记忆 + Gating 网络配置示例
# ============================================================================

# ============================================================================
# 示例1：原始模式（单一频率，向后兼容）
# ============================================================================
config_single_freq = {
    # 基础配置
    'memory_len': 10,
    'token_num': 838,  # (616 + 96 + 126)

    # 特征配置
    'tf_fuse_cfg': {
        'feat_dim': 256,
        'num_layers': 2,
    },
    'modal_fuse_cfg': {
        'feat_dim': 256,
        'num_layers': 2,
    },

    # 其他配置
    'num_layers': 2,
    'motion_dim': 256,
    'use_temp_attn_mask': False,
    'convertD': False,

    # 位置编码配置
    'front_pos_param': {
        'cam_depth': 96,
        'feat_dim': 256,
    },
    'side_pos_param': {
        'cam_depth': 96,
        'feat_dim': 256,
    },
    'rear_pos_param': {
        'cam_depth': 96,
        'feat_dim': 256,
    },

    # 显式禁用高低频分层
    'enable_hierarchical': False,
}


# ============================================================================
# 示例2：高低频分层模式 + Gating 融合（新功能）
# ============================================================================
config_hierarchical = {
    # 基础配置（memory_len 会自动计算）
    'token_num': 838,  # (616 + 96 + 126)

    # 特征配置
    'tf_fuse_cfg': {
        'feat_dim': 256,
        'num_layers': 2,
    },
    'modal_fuse_cfg': {
        'feat_dim': 256,
        'num_layers': 2,
    },

    # 其他配置
    'num_layers': 2,
    'motion_dim': 256,
    'use_temp_attn_mask': False,
    'convertD': False,

    # 位置编码配置
    'front_pos_param': {
        'cam_depth': 96,
        'feat_dim': 256,
    },
    'side_pos_param': {
        'cam_depth': 96,
        'feat_dim': 256,
    },
    'rear_pos_param': {
        'cam_depth': 96,
        'feat_dim': 256,
    },

    # ========================================================================
    # [新增] 高低频分层配置
    # ========================================================================
    # 启用高低频分层功能
    'enable_hierarchical': True,

    # 高频区配置：前3秒，2Hz采样
    'high_freq_len': 6,  # 6帧 @ 2Hz = 3秒覆盖

    # 低频区配置：后3秒，1Hz采样
    'low_freq_len': 3,   # 3帧 @ 1Hz = 3秒覆盖

    # 总 memory 长度 = 1(当前) + 6(高频) + 3(低频) = 10帧
    # 注意：memory_len 会自动计算，无需手动设置

    # 是否使用 Gating 网络
    'use_gating': True,

    # 可选：Ego pose 扰动（数据增强）
    'perturb_ego_pose': False,
    # 'perturb_params': {
    #     'sigma_t': 0.2,
    #     'sigma_r_deg': 2.0,
    #     'prob': 0.4,
    # },
}


# ============================================================================
# 示例3：调试配置（高频/低频帧数减少）
# ============================================================================
config_debug = {
    'token_num': 838,

    'tf_fuse_cfg': {
        'feat_dim': 256,
        'num_layers': 2,
    },
    'modal_fuse_cfg': {
        'feat_dim': 256,
        'num_layers': 2,
    },

    'num_layers': 2,
    'motion_dim': 256,
    'use_temp_attn_mask': False,
    'convertD': False,

    'front_pos_param': {
        'cam_depth': 96,
        'feat_dim': 256,
    },
    'side_pos_param': {
        'cam_depth': 96,
        'feat_dim': 256,
    },
    'rear_pos_param': {
        'cam_depth': 96,
        'feat_dim': 256,
    },

    # 调试模式：减少帧数
    'enable_hierarchical': True,
    'high_freq_len': 2,  # 2帧 @ 2Hz = 1秒
    'low_freq_len': 2,   # 2帧 @ 1Hz = 2秒
    'use_gating': True,
}


# ============================================================================
# 使用示例
# ============================================================================
"""
import torch
from easydict import EasyDict as edict
from temp_fusion_module import PVLongTempFusionBaseModule

# 创建模型
cfg = edict(config_hierarchical)
model = PVLongTempFusionBaseModule(cfg)

# 模拟输入
B, N, C = 4, 838, 256
tokens = torch.randn(B, N, C)
tokens_pos = torch.randn(B, N, C)

temp_fusion_inputs = edict({
    'prev_exists': torch.ones(B, dtype=torch.bool),
    'tps': torch.randn(B, 1, 1) * 10.0,  # 模拟时间戳
    'ego_pose': torch.eye(4).unsqueeze(0).repeat(B, 1, 1, 1),
    'pos_3d': edict({
        'pos_3d_front': torch.randn(B, 2, 24, 32, 96),
        'pos_3d_rear': torch.randn(B, 1, 12, 16, 96),
        'pos_3d_side': torch.randn(B, 4, 48, 64, 96),
    }),
})

ego_pose_inv = torch.inverse(torch.eye(4).unsqueeze(0).repeat(B, 1, 1, 1))

# 前向传播
output = model.forward(tokens, tokens_pos, temp_fusion_inputs, ego_pose_inv)

print(f"Output shape: {output.shape}")  # (4, 838, 256)

# 查看 Gating 权重
if hasattr(model, 'last_gate_weights'):
    print(f"Gate weights: {model.last_gate_weights}")  # (4, 3)
"""
"""
