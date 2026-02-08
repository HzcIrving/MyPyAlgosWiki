# 时序融合模块高低频分层 + Gating 网络设计方案

## 一、整体架构设计

### 1.1 Memory 存储结构

```
原尾插结构（单一频率）：
Memory = [历史1, 历史2, ..., 历史N-1, 当前帧]
          ↑                              ↑
        Index0                      IndexN-1 (尾插)

新高低频分层结构（尾插，按时间从远到近）：
Memory = [低频历史(1Hz, 3帧), 高频历史(2Hz, 6帧), 当前帧(1帧)]
          ↓                              ↓                 ↓
        Index0-2                      Index3-8         Index9 (尾插)

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

### 1.2 时序规则

```
时间轴：    t-6    t-5.5   t-5    t-4.5  t-4    t-3.5  t-3    t-2.5  t-2    t-1.5  t-1    t-0.5  t
          │      │      │      │      │      │      │      │      │      │      │      │  │
频率：      1Hz    ↓      2Hz    ↓      ↓      ↓      ↓      ↓      ↓      ↓      ↓      ↓  │
低频：      LF0                           ────────────────→ LF1   LF2    LF3
                                                      ↑      ↑      ↑
高频：            HF0    HF1    HF2    HF3    HF4    HF5    HF6    ────→ 溢出到低频
                                                           ↓
当前：                                                                     CUR
```

## 二、Gating 网络设计

### 2.1 设计思路

参考 DeepSeek Engram 的门控机制，但针对时序数据特点进行适配：

**核心思想**：
- 不同时序范围的信息重要性不同
- 高频历史（近期）提供精细的运动信息
- 低频历史（远期）提供长期的语义信息
- 通过门控网络自适应地学习融合权重

**关键改进**：
1. 加入时间感知：通过连续时间编码注入时序信息
2. 多尺度融合：对当前帧、高频历史、低频历史分别编码
3. 轻量化设计：简单的 MLP 结构，避免引入大量参数

### 2.2 网络结构

```
输入：
  - current_token: (B, N, C)  当前帧特征
  - high_freq_tokens: (B, N_high, C)  高频历史帧
  - low_freq_tokens: (B, N_low, C)   低频历史帧
  - high_time_embs: (B, C)  高频时间编码（平均）
  - low_time_embs: (B, C)   低频时间编码（平均）

处理流程：
  1. 全局语义提取：对当前帧做空间池化
  2. 时间上下文融合：拼接高频/低频时间编码
  3. 门控权重生成：通过 MLP 生成三部分权重
  4. 自适应融合：对各部分特征进行加权

输出：
  - gate_weights: (B, 3)  [w_cur, w_high, w_low]
```

### 2.3 网络结构定义

```python
class FrequencyGating(nn.Module):
    """
    时空耦合门控网络

    功能：对当前帧、高频历史帧、低频历史帧进行自适应融合

    输入：
        - current_token: (B, N, C) 当前帧特征
        - high_time_emb: (B, C) 高频历史的时间编码
        - low_time_emb: (B, C) 低频历史的时间编码

    输出：
        - gate_weights: (B, 3) 三部分融合权重 [w_cur, w_high, w_low]
    """
    def __init__(self, feat_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or feat_dim // 2

        self.fc = nn.Sequential(
            nn.Linear(feat_dim * 2, hidden_dim),  # [视觉特征, 时间上下文]
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3)  # 输出三部分权重
        )
```

## 三、代码修改策略

### 3.1 新增内容

1. **新增配置参数**：
   - `high_freq_len`: 高频历史帧数（默认6）
   - `low_freq_len`: 低频历史帧数（默认3）
   - `total_memory_len`: 自动计算 = 1 + high_freq_len + low_freq_len

2. **新增类**：
   - `FrequencyGating`: 门控网络

3. **新增方法**：
   - `_update_memory_hierarchical`: 分层记忆更新
   - `_compute_gate_weights`: 计算门控权重
   - `_apply_gating`: 应用门控融合

### 3.2 修改内容

1. **`__init__`**：
   - 添加高低频配置参数
   - 初始化时间编码器
   - 初始化门控网络

2. **`_update_memory`**：
   - 实现分层更新逻辑
   - 高频区：每帧更新
   - 低频区：每2帧更新一次（降频采样）

3. **`forward`**：
   - 分离高频/低频历史帧
   - 计算时间编码
   - 通过门控网络融合
   - 应用权重到特征

### 3.3 保持不变

1. **空间位置编码**：`get_pv_3dpe_with_hist`
2. **Token 合并**：`merge_pv_memory_tokens`
3. **Attention 机制**：`temporal_attn`, `frame_attn`
4. **Memory 刷新**：`refresh_memory`, `refresh_ego_pose`

## 四、关键实现细节

### 4.1 Memory 索引布局

```
Index:    [0]      [1]   [2]   [3]   [4]   [5]   [6]    [7]    [8]    [9]
Content:  [当前帧] [高频0][高频1][高频2][高频3][高频4][高频5] [低频0][低频1][低频2]
          ↓       ↓                                            ↓
          CUR     High Freq (0-3s, 2Hz)                      Low Freq (3-6s, 1Hz)
```

### 4.2 更新计数器

使用 `update_counter` 跟踪帧数，实现降频采样：
- 奇数次：低频区保持不变
- 偶数次：接收高频区溢出的帧

### 4.3 时间编码注入

```python
# 计算时间差
time_deltas = current_timestamp - memory_timestamps  # (B, T)

# 生成正弦位置编码
time_embs = self.time_embedder(time_deltas)  # (B, T, C)

# 对高频/低频分别取平均
high_time_emb = (time_embs[:, 1:1+high_freq_len] * high_mask).mean(dim=1)
low_time_emb = (time_embs[:, 1+high_freq_len:] * low_mask).mean(dim=1)
```

## 五、测试建议

### 5.1 单元测试

```python
# 测试1：Memory 布局正确性
def test_memory_layout():
    module = PVLongTempFusionBaseModule(config)
    # 输入10帧数据
    for i in range(10):
        module.forward(...)
    # 验证：
    assert module.memory.tokens.shape[1] == 10  # [1当前 + 6高频 + 3低频]
    assert module.memory.tokens[:, 0].shape  # 当前帧在 index 0

# 测试2：高低频分区正确性
def test_frequency_separation():
    # 前6帧应该存入高频区
    for i in range(6):
        module.forward(...)
    assert module.memory.valid_frames_in_high_freq() == 6

    # 第7帧应该触发溢出到低频区
    module.forward(...)
    assert module.memory.valid_frames_in_low_freq() == 1

# 测试3：降频采样正确性
def test_downsampling():
    # 高频区溢出时，低频区应该每2帧更新一次
    for i in range(10):
        prev_low_count = module.memory.get_low_freq_count()
        module.forward(...)
        curr_low_count = module.memory.get_low_freq_count()

        if i % 2 == 1:  # 奇数次
            assert curr_low_count == prev_low_count + 1  # 低频区更新
        else:  # 偶数次
            assert curr_low_count == prev_low_count  # 低频区保持
```

### 5.2 集成测试

```python
# 测试4：Gating 网络输出
def test_gating_network():
    module = PVLongTempFusionBaseModule(config)
    weights = module.gating_net(current_token, high_time_emb, low_time_emb)

    # 验证输出形状
    assert weights.shape == (B, 3)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(B))  # softmax 归一化

    # 验证权重范围
    assert (weights >= 0).all() and (weights <= 1).all()

# 测试5：端到端融合效果
def test_end_to_end_fusion():
    module = PVLongTempFusionBaseModule(config)
    output = module.forward(tokens, ...)

    # 验证输出形状
    assert output.shape == (B, N, C)

    # 验证梯度传播
    loss = output.sum()
    loss.backward()
    assert module.gating_net.fc[0].weight.grad is not None
```

### 5.3 可视化测试

```python
# 测试6：权重分布可视化
def test_weight_distribution():
    weights_history = []
    for i in range(100):
        output = module.forward(...)
        weights_history.append(module.last_gate_weights.detach())

    # 绘制权重随时间的变化
    import matplotlib.pyplot as plt
    plt.plot(weights_history)
    plt.legend(['w_cur', 'w_high', 'w_low'])
    plt.xlabel('Frame')
    plt.ylabel('Weight')
    plt.title('Gating Weights over Time')
    plt.savefig('gating_weights.png')
```

## 六、依赖说明

### 6.1 新增依赖

无新增外部依赖。所有实现基于现有的 PyTorch 组件。

### 6.2 现有依赖

```
torch>=1.8.0
torchvision
easydict
numpy
```

### 6.3 安装方式

```bash
# 基础环境
conda create -n tfmodule python=3.8
conda activate tfmodule
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install easydict
```

## 七、性能考虑

### 7.1 计算开销

| 组件 | 参数量 | 计算量(FLOPs) | 说明 |
|------|--------|---------------|------|
| Gating 网络 | ~1M | O(B × C²) | 轻量级 MLP |
| 时间编码 | 0 | O(B × T × C) | 简单三角函数计算 |
| 分层更新 | 0 | O(1) | 仅索引操作 |

### 7.2 内存开销

- 额外存储：update_counter (1 int)
- Gating 网络：约 1M 参数
- 总体增加：< 5% 原始内存占用

## 八、兼容性保证

### 8.1 向后兼容

- 通过配置开关控制是否启用高低频功能
- 默认行为与原模块保持一致

### 8.2 配置示例

```python
# 原始模式（单频率）
config = {
    'memory_len': 10,
    'enable_hierarchical': False,  # 关闭高低频分层
}

# 新模式（高低频分层）
config = {
    'memory_len': 10,  # 1当前 + 6高频 + 3低频
    'enable_hierarchical': True,  # 启用高低频分层
    'high_freq_len': 6,
    'low_freq_len': 3,
    'use_gating': True,
}
```

## 九、设计原理总结

### 9.1 为什么这样设计？

1. **符合人类视觉系统**：
   - 中央凹（高频）处理近期、精细信息
   - 周边视觉（低频）处理远期、概览信息

2. **符合时序数据特性**：
   - 近期变化快，需要高采样率
   - 远期变化慢，可以用低采样率

3. **计算效率优化**：
   - 避免对所有历史帧同等对待
   - 通过门控网络自适应学习重要性

### 9.2 与原方案对比

| 方面 | 原尾插方案 | 高低频分层方案 |
|------|-----------|---------------|
| 时序覆盖 | 统一频率 | 差异化频率 |
| 计算效率 | O(T × T) | O(T_h² + T_l²) |
| 信息利用 | 统一对待 | 自适应加权 |
| 参数量 | 基线 | +5% |
| 性能提升 | 基线 | +5-10% (预期) |

---

**文档版本**: v1.0
**创建时间**: 2025-01-08
**作者**: Claude + User Collaboration
