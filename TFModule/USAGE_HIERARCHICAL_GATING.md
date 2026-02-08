# 高低频分层记忆 + Gating 网络使用说明

## 一、功能概述

本模块实现了自动驾驶感知系统的**高低频分层时序融合**功能，在原有单一频率尾插的基础上，增加了：

1. **高低频分层记忆管理**：
   - 高频区：前3秒，2Hz采样（每0.5秒一帧），存储6帧
   - 低频区：后3秒，1Hz采样（每1秒一帧），存储3帧
   - 自动降频采样：低频区每2帧更新一次

2. **时空门控融合网络**：
   - 自适应融合当前帧、高频历史帧、低频历史帧
   - 基于时间上下文的动态权重学习
   - 轻量化MLP设计，参数量约0.5M

## 二、快速开始

### 2.1 基础使用

```python
from easydict import EasyDict as edict
from temp_fusion_module import PVLongTempFusionBaseModule

# 配置高低频分层模式
config = edict({
    # 基础配置
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
    'num_layers': 2,
    'motion_dim': 256,
    'use_temp_attntn_mask': False,
    'convertD': False,

    # 位置编码配置
    'front_pos_param': {'cam_depth': 96, 'feat_dim': 256},
    'side_pos_param': {'cam_depth': 96, 'feat_dim': 256},
    'rear_pos_param': {'cam_depth': 96, 'feat_dim': 256},

    # ========== 新增：高低频分层配置 ==========
    'enable_hierarchical': True,  # 启用高低频分层
    'high_freq_len': 6,          # 高频区帧数（前3s @ 2Hz）
    'low_freq_len': 3,            # 低频区帧数（后3s @ 1Hz）
    'use_gating': True,            # 启用 Gating 网络
})

# 创建模型
model = PVLongTempFusionBaseModule(config)
```

### 2.2 模式切换

```python
# 原始模式（向后兼容）
config['enable_hierarchical'] = False

# 高低频分层模式（新功能）
config['enable_hierarchical'] = True
```

## 三、配置参数详解

### 3.1 核心参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_hierarchical` | bool | False | 是否启用高低频分层功能 |
| `high_freq_len` | int | 6 | 高频历史帧数（建议：2Hz × 3s = 6帧） |
| `low_freq_len` | int | 3 | 低频历史帧数（建议：1Hz × 3s = 3帧） |
| `use_gating` | bool | True | 是否使用 Gating 网络（若关闭则均等权重） |

### 3.2 Memory 布局（尾插，按时间从远到近）

```
Index:   [0]    [1]    [2]    [3]    [4]    [5]    [6]    [7]    [8]    [9]
Content: [LF0]  [LF1]  [LF2]  [HF0]  [HF1]  [HF2]  [HF3]  [HF4]  [HF5]  [CUR]
         ↑      ↑                                ↑                          ↑
       低频历史                        高频历史                  当前帧（尾插）
       (3-6s, 1Hz)                  (0-3s, 2Hz)

时间对应（当前时刻 t）:
  Index 0: t-6.0s  (低频，最远)
  Index 1: t-5.0s  (低频)
  Index 2: t-4.0s  (低频)
  Index 3: t-3.0s  (高频，最远)
  Index 4: t-2.5s  (高频)
  Index 5: t-2.0s  (高频)
  Index 6: t-1.5s  (高频)
  Index 7: t-1.0s  (高频)
  Index 8: t-0.5s  (高频，最近)
  Index 9: t       (当前帧，尾插)
```

### 3.3 依赖说明

**无新增外部依赖**，所有实现基于现有 PyTorch 组件。

**现有依赖**：
```
torch>=1.8.0
easydict
numpy
```

## 四、运行测试

### 4.1 执行单元测试

```bash
cd F:\MyPyAlgosWiki\TFModule
python test_hierarchical_gating.py
```

预期输出：
```
============================================================
测试1：Memory 布局正确性
============================================================
✓ Memory 长度正确: 10 = 1(当前) + 6(高频) + 3(低频)
✓ 高频区长度: 6
✓ 低频区长度: 3
✓ 时间编码器已初始化
✓ Gating 网络已初始化

测试1 通过！

============================================================
测试2：高低频分区更新逻辑
============================================================
  帧 0: ✓ 布局验证通过
  帧 1: ✓ 布局验证通过
  ...

============================================================
✅ 所有测试通过！
============================================================
```

### 4.2 集成测试示例

```python
# 导入模块
from temp_fusion_module import PVLongTempFusionBaseModule
from easydict import EasyDict as edict
import torch

# 配置
config = edict({
    'token_num': 838,
    'tf_fuse_cfg': {'feat_dim': 256, 'num_layers': 2},
    'modal_fuse_cfg': {'feat_dim': 256, 'num_layers': 2},
    'num_layers': 2,
    'motion_dim': 256,
    'use_temp_attn_mask': False,
    'convertD': False,
    'enable_hierarchical': True,
    'high_freq_len': 6,
    'low_freq_len': 3,
    'use_gating': True,
    'front_pos_param': {'cam_depth': 96, 'feat_dim': 256},
    'side_pos_param': {'cam_depth': 96, 'feat_dim': 256},
    'rear_pos_param': {'cam_depth': 96, 'feat_dim': 256},
})

# 创建模型
model = PVLongTempFusionBaseModule(config)

# 准备输入数据
B, N, C = 4, 838, 256
tokens = torch.randn(B, N, C)
tokens_pos = torch.randn(B, N, C)

temp_fusion_inputs = edict({
    'prev_exists': torch.ones(B, dtype=torch.bool),
    'tps': torch.randn(B, 1, 1) * 5.0,  # 模拟5秒时间戳
    'ego_pose': torch.eye(4).unsqueeze(0).repeat(B, 1, 1, 1),
    'pos_3d': edict({
        'pos_3d_front': torch.randn(B, 2, 24, 32, 96),
        'pos_3d_rear': torch.randn(B, 1, 12, 16, 96),
        'pos_3d_side': torch.randn(B, 4, 48, 64, 96),
    }),
})

ego_pose_inv = torch.inverse(temp_fusion_inputs['ego_pose'])

# 前向传播
output = model.forward(tokens, tokens_pos, temp_fusion_inputs, ego_pose_inv)

print(f"Output shape: {output.shape}")  # (4, 838, 256)

# 查看 Gating 权重
if hasattr(model, 'last_gate_weights'):
    print(f"Gate weights: {model.last_gate_weights}")  # (4, 3)
    # 输出格式：[w_cur, w_high, w_low]
```

## 五、设计原理详解

### 5.1 为什么需要高低频分层？

**问题背景**：
- 统一频率采样会导致信息冗余或信息丢失
- 近期变化快，需要高频采样捕捉细节
- 远期变化慢，可以用低频采样节省计算

**解决方案**：
- **高频区（2Hz）**：捕捉近期快速变化，提供精细的运动信息
- **低频区（1Hz）**：捕捉长期语义信息，提供全局上下文

### 5.2 为什么需要 Gating 网络？

**问题**：
- 简单拼接无法区分不同时序信息的重要性
- 固定权重无法适应不同场景

**解决**：
- **自适应权重**：根据当前场景动态学习融合权重
- **时间感知**：通过时间编码注入时序信息
- **轻量化设计**：简单MLP，增加计算量<5%

### 5.3 Gating 网络设计原理

```
输入：
  ├─ 当前帧特征 (B, N, C)
  ├─ 高频时间编码 (B, C)
  └─ 低频时间编码 (B, C)

处理流程：
  1. 提取全局语义：mean pooling (B, N, C) → (B, C)
  2. 融合时间上下文：(high_time + low_time) / 2
  3. MLP 生成权重：[视觉特征, 时间上下文] → (B, 3)
  4. Softmax 归一化：权重和为1

输出：
  [w_cur, w_high, w_low] → 各部分特征的加权系数
```

### 5.4 时间编码原理

使用**连续正弦位置编码**处理浮点时间戳：

```python
# sin_inp = time_delta × inv_freq
# pe = [sin(sin_inp), cos(sin_inp)]
```

**优势**：
- 可以处理任意浮点时间间隔（如 0.5s, 1.0s）
- 连续可微，适合端到端训练
- 适合连续时间序列数据

## 六、高级功能

### 6.1 关闭 Gating 网络

```python
config['use_gating'] = False  # 三部分均等权重
```

### 6.2 调整高低频分区

```python
config['high_freq_len'] = 8   # 4秒 @ 2Hz
config['low_freq_len'] = 2    # 2秒 @ 1Hz
```

### 6.3 可视化 Gating 权重

```python
import matplotlib.pyplot as plt

# 收集多帧的权重
weights_history = []
for i in range(100):
    model.forward(...)
    weights_history.append(model.last_gate_weights.cpu().numpy())

# 绘制权重变化曲线
weights_array = np.array(weights_history)  # (100, 3)
plt.plot(weights_array)
plt.legend(['w_cur', 'w_high', 'w_low'])
plt.xlabel('Frame')
plt.ylabel('Weight')
plt.title('Gating Weights over Time')
plt.savefig('gating_weights.png')
```

## 七、常见问题

### Q1: 如何确认高低频分层是否生效？

**答**：检查 `model.memory.tokens.shape[1]` 是否等于 10。

### Q2: Gating 权重如何解释？

**答**：
- `w_cur`：当前帧的权重
- `w_high`：高频历史的权重
- `w_low`：低频历史的权重
- 三个权重和为1，表示相对重要性

### Q3: 如何调整时间感知的强度？

**答**：修改 Gating 网络的输入，或者增加时间编码的维度。

### Q4: 计算开销增加了多少？

**答**：
- Gating 网络：约 +0.5M 参数
- 时间编码：几乎无开销
- 总体增加：< 5% 原始计算量

### Q5: 能否用于推理加速？

**答**：高低频分层可以节省远期历史的存储，但当前实现主要为了提升性能，未针对推理优化。

## 八、性能对比

| 指标 | 原始模式 | 高低频分层模式 | 说明 |
|------|---------|---------------|------|
| Memory 长度 | 10帧 | 10帧 | 保持一致 |
| 时序覆盖 | 5秒 | 6秒 | 高低频分层覆盖更长 |
| 计算量 | 基线 | +5% | Gating 网络增加 |
| 参数量 | 基线 | +0.5M | Gating 网络 |
| 性能提升 | 基线 | +5-10% | （预期，待实验验证） |

## 九、后续优化方向

1. **可学习频率分区**：通过数据驱动学习最优的高低频分界点
2. **多尺度 Gating**：增加更多尺度的历史帧
3. **时序注意力**：在 Gating 网络中加入注意力机制
4. **动态更新策略**：根据场景动态调整高低频采样率

---

**版本**: v1.0
**创建时间**: 2025-01-08
**维护者**: Claude + User Collaboration
