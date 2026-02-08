# MemoryBank 更新原理说明

## 概述

本文档详细说明 PVLongTempFusionBaseModule 中 MemoryBank 的更新原理，涵盖三个关键阶段：

1. **冷启动阶段**：首次初始化 MemoryBank
2. **正常运作阶段**：连续帧序列的更新
3. **Refresh 阶段**：Batch 中某条数据切换到新 Clip 的处理

---

## 一、Memory 布局结构

### 1.1 原始模式（单一频率）

```
Index:   [0]      [1]    [2]    [3]    ...    [N-2]   [N-1]
Content: [历史1]  [历史2] [历史3] [历史4] ...   [历史N-1] [上一帧]
         ↑─────────────────────────────────────────↑
              统一频率（1Hz 或 2Hz），共 N 帧
```

### 1.2 高低频分层模式（新功能，尾插按时间从远到近）

```
Index:   [0]    [1]    [2]    [3]    [4]    [5]    [6]    [7]    [8]    [9]
Content: [LF0]  [LF1]  [LF2]  [HF0]  [HF1]  [HF2]  [HF3]  [HF4]  [HF5]  [CUR]
时间:    t-6.0  t-5.0  t-4.0  t-3.0  t-2.5  t-2.0  t-1.5  t-1.0  t-0.5   t
         ↑      ↑                           ↑                        ↑
       低频历史(3-6s)                  高频历史(0-3s)          当前帧(尾插)
       3帧, 1Hz                        6帧, 2Hz

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

### 1.3 Memory 存储字段

| 字段 | 形状 | 说明 |
|------|------|------|
| `tokens` | (B, T, N, C) | Token 特征 |
| `tokens_pos` | (B, T, N, C) | 位置编码特征 |
| `tps` | (B, T, 1) | 时间戳 |
| `ego_pose` | (B, T, 4, 4) | Ego 姿态变换矩阵 |
| `memory_mask` | (B, T) | 有效帧掩码（1=有效，0=无效） |
| `counts` | (B, T) | 帧计数（用于压缩模式） |

---

## 二、冷启动阶段（Cold Start）

### 2.1 触发条件

在 `pre_update_memory` 方法中：

```python
if (not self.memory) or (not prev_exists.any()):
    # 初始化 MemoryBank
```

**触发条件**：
- `self.memory` 为 `None`：首次调用，或被 `reset_memory()` 清空后
- `prev_exists.any()` 为 `False`：Batch 中所有样本都没有前一帧（如数据集的第一帧）

### 2.2 初始化流程

```python
self.memory = edict({
    'tokens': prev_exists.new_zeros(B, self.memory_len, self.token_num, self.feat_dim),
    'tokens_pos': prev_exists.new_zeros(B, self.memory_len, self.token_num, self.feat_dim),
    'tps': tps.new_zeros(B, self.memory_len, 1),
    'ego_pose': prev_exists.new_zeros(B, self.memory_len, 4, 4),
    'memory_mask': torch.zeros((B, self.memory_len), device=device, dtype=torch.float32),
    'counts': torch.ones((B, self.memory_len), device=device, dtype=torch.float32)
})
```

**初始化状态**：
- 所有特征张量初始化为**全零**
- `memory_mask` 初始化为**全零**（表示所有帧都无效）
- `update_counter` 初始化为 0

### 2.3 冷启动后的第一次更新

第一次调用 `_update_memory` 时：
- 新当前帧 → Index 0
- `memory_mask[0]` 从 0 → 1（标记为有效）
- 高频区仍为空（mask=0）

**示例**：
```
初始状态:  [0,0,0,0,0,0,0,0,0,0]  (mask全0)
第1次更新: [1,0,0,0,0,0,0,0,0,0]  (Index0有效)
第2次更新: [1,1,0,0,0,0,0,0,0,0]  (Index0-1有效，高频区开始填充)
...
```

---

## 三、正常运作阶段（Normal Operation）

### 3.1 触发条件

```python
# self.memory 已存在
# prev_exists[b] = True 表示样本 b 是连续帧序列
```

### 3.2 更新流程

```
┌─────────────────────────────────────────────────────────────┐
│  forward()                                                  │
│    ↓                                                        │
│  pre_update_memory()  ← 对 prev_exists=False 的样本 refresh │
│    ↓                                                        │
│  _update_memory()      ← 核心 Memory 更新逻辑               │
│    ↓                                                        │
│  分发器：                                                   │
│    - enable_hierarchical=False → _update_memory_standard()  │
│    - enable_hierarchical=True  → _update_memory_hierarchical() │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 原始模式更新逻辑（_update_memory_standard）

**Memory 布局**：`[历史1, 历史2, ..., 历史N-1, 上一帧]`

**更新策略**：尾插（FIFO）

```python
# 丢弃最老的历史帧
memory = memory[:, 1:]  # 去掉 Index 0
# 新帧追加到尾部
memory = torch.cat([memory, new_frame], dim=1)
```

**示例**：
```
更新前:  [H1, H2, H3, H4, H5, CUR]
丢弃 H1: [H2, H3, H4, H5, CUR]
追加 NEW: [H2, H3, H4, H5, CUR, NEW]
```

### 3.4 高低频分层模式更新逻辑（_update_memory_hierarchical）

**Memory 布局**：`[当前帧, 高频(6帧), 低频(3帧)]`

#### 3.4.1 高频区未满时的更新

**判断条件**：
```python
high_valid_count = prev_high_mask.sum(dim=1, keepdim=True)  # (B, 1)
high_is_full = (high_valid_count >= high_len).float()
```

**更新策略**：
1. 找到第一个空位（mask=0 的位置）
2. 旧当前帧填入第一个空位
3. 其他帧保持不变
4. **无溢出帧**（不产生低频区更新）

**示例**：
```
初始状态:  Index  [0]   [1]   [2]   [3]   [4]   [5]   [7]   [8]   [9]
          Mask    [1]   [1]   [1]   [0]   [0]   [0]   [0]   [0]   [0]
          Content [CUR] [HF0] [HF1] [ -- ] [ -- ] [ -- ] [ -- ] [ -- ] [ -- ]

更新后:    Index  [0]   [1]   [2]   [3]   [4]   [5]   [7]   [8]   [9]
          Mask    [1]   [1]   [1]   [1]   [0]   [0]   [0]   [0]   [0]
          Content [NEW] [CUR] [HF0] [HF1] [ -- ] [ -- ] [ -- ] [ -- ] [ -- ]
                                 ↑
                         旧当前帧填入第一个空位
```

#### 3.4.2 高频区已满时的更新

**更新策略**：
1. 新当前帧 → Index 0
2. 旧当前帧 → 高频区头部（Index 1）
3. 高频区后移：`[旧当前, HF0, HF1, HF2, HF3, HF4]`
4. **HF5 溢出**（成为低频区候选）
5. 低频区根据 `update_counter % 2` 决定是否接收溢出帧

**示例**：
```
更新前:  Index  [0]   [1]   [2]   [3]   [4]   [5]   [7]   [8]   [9]
          Mask    [1]   [1]   [1]   [1]   [1]   [1]   [1]   [1]   [1]
          Content [CUR] [HF0] [HF1] [HF2] [HF3] [HF4] [HF5] [LF0] [LF1] [LF2]
          时间     t    t-0.5 t-1.0 t-1.5 t-2.0 t-2.5 t-3.0 t-4.0 t-5.0 t-6.0

更新后:  Index  [0]   [1]   [2]   [3]   [4]   [5]   [7]   [8]   [9]
          Mask    [1]   [1]   [1]   [1]   [1]   [1]   [1]   [1]   [1]
          Content [NEW] [CUR] [HF0] [HF1] [HF2] [HF3] [HF4] [LF0] [LF1] [LF2]
          时间     t    t-0.5 t-1.0 t-1.5 t-2.0 t-2.5 t-3.0 t-4.0 t-5.0 t-6.0
                              ↑
                        HF5 溢出（候选）
```

#### 3.4.3 降频采样逻辑（2Hz → 1Hz）

**核心代码**：
```python
self.update_counter += 1
if self.update_counter % 2 == 0:
    # 偶数次：接收溢出帧
    new_low_data = torch.cat([overflow_data, prev_low_data[:, :-1]], dim=1)
else:
    # 奇数次：保持不变（溢出帧丢弃）
    new_low_data = prev_low_data
```

**时间推演**：
```
更新次数 | counter | 奇偶 | 溢出帧 | 低频区操作
---------|---------|------|--------|------------
  6      | 5→6     | 偶  | HF5    | 接收 HF5
  7      | 6→7     | 奇  | HF4    | 丢弃 HF4
  8      | 7→8     | 偶  | HF3    | 接收 HF3
  9      | 8→9     | 奇  | HF2    | 丢弃 HF2
  10     | 9→10    | 偶  | HF1    | 接收 HF1
```

**结果**：低频区接收 HF5, HF3, HF1（间隔1秒），对应时间戳 t-4.0, t-5.0, t-6.0

---

## 四、Refresh 阶段（Clip 切换）

### 4.1 触发条件

当 Batch 中某条数据切换到新的 Clip 时：
```python
prev_exists[b] = False  # 样本 b 切换到新 Clip
```

**典型场景**：
- DataLoader 在一个 Batch 中混入了多个 Clip 的数据
- 样本 b 的上一帧属于 Clip A，当前帧属于 Clip B

### 4.2 Refresh 流程

```
┌─────────────────────────────────────────────────────────────┐
│  forward() 被调用                                            │
│    ↓                                                        │
│  pre_update_memory()                                        │
│    ↓                                                        │
│  refresh_memory(memory_ele, prev_exists)  ← 对所有字段 apply │
│    ↓                                                        │
│  memory_ele * prev_exists  ← 元素级乘法                    │
│    ↓                                                        │
│  prev_exists[b]=0 的样本 → memory[b] 被清零                  │
│  prev_exists[b]=1 的样本 → memory[b] 保持不变                │
└─────────────────────────────────────────────────────────────┘
```

### 4.3 refresh_memory 实现细节

```python
def refresh_memory(self, memory_ele, prev_exists):
    """
    Args:
        memory_ele: (B, T, ...) 任意形状的 memory 元素
        prev_exists: (B,) bool 张量，标记哪些样本有前一帧

    Returns:
        refreshed_memory: (B, T, ...) 刷新后的 memory
    """
    view_shape = [-1] + [1] * (memory_ele.dim() - 1)
    prev_exists = prev_exists.to(memory_ele.dtype)
    prev_exists = prev_exists.view(*view_shape)
    return memory_ele * prev_exists  # 广播乘法
```

**核心操作**：`memory_ele * prev_exists`

- `prev_exists[b]=1` → `memory_ele[b] * 1 = memory_ele[b]`（保持不变）
- `prev_exists[b]=0` → `memory_ele[b] * 0 = 0`（清零）

### 4.4 示例：Batch Refresh

**假设**：Batch size = 4，样本 2 切换到新 Clip

```python
prev_exists = [True, True, False, True]  # 样本 2 切换
```

**tokens 字段**：
```
刷新前 tokens: (B=4, T=10, N, C)
  tokens[0]:  [有效历史帧...]  # prev_exists[0]=True，保持
  tokens[1]:  [有效历史帧...]  # prev_exists[1]=True，保持
  tokens[2]:  [有效历史帧...]  # prev_exists[2]=False，清零
  tokens[3]:  [有效历史帧...]  # prev_exists[3]=True，保持

刷新后 tokens:
  tokens[0]:  [有效历史帧...]  # 不变
  tokens[1]:  [有效历史帧...]  # 不变
  tokens[2]:  [全零张量...]     # 清零！
  tokens[3]:  [有效历史帧...]  # 不变
```

**memory_mask 字段**：
```
刷新前 memory_mask: (B=4, T=10)
  mask[0]:  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # 全有效
  mask[1]:  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  mask[2]:  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  mask[3]:  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

刷新后 memory_mask:
  mask[0]:  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # 不变
  mask[1]:  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # 不变
  mask[2]:  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 清零！
  mask[3]:  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # 不变
```

### 4.5 Refresh 后的第一次更新

```
refresh 后:
  tokens[2]:  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 全零
  mask[2]:    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 全零

第一次更新 _update_memory:
  新当前帧 → Index 0
  tokens[2][0] = new_tokens[2]  # 填入新帧
  mask[2][0] = 1  # 标记为有效

结果:
  tokens[2]:  [NEW, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 只有一帧有效
  mask[2]:    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]    # 从头开始填充
```

**关键点**：Refresh 后，样本 2 的 MemoryBank 变成了"冷启动"状态，后续更新会重新填充。

---

## 五、完整流程图

``┌──────────────────────────────────────────────────────────────────┐
│                         forward() 被调用                          │
└──────────────────────────────────────────────────────────────────┘
                                ↓
┌──────────────────────────────────────────────────────────────────┐
│                    pre_update_memory()                           │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ if (not self.memory) or (not prev_exists.any()):          │  │
│  │     初始化 MemoryBank（全零） ← 冷启动                     │  │
│  └────────────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ for key in self.memory.keys():                            │  │
│  │     self.memory[key] = refresh_memory(                     │  │
│  │         self.memory[key], prev_exists                      │  │
│  │     )                                                       │  │
│  │     # prev_exists[b]=0 的样本被清零 ← Refresh              │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
                                ↓
┌──────────────────────────────────────────────────────────────────┐
│                      _update_memory()                            │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ if not enable_hierarchical:                               │  │
│  │     _update_memory_standard()  ← 原始模式                  │  │
│  │ else:                                                     │  │
│  │     _update_memory_hierarchical()  ← 高低频分层模式         │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
                                ↓
┌──────────────────────────────────────────────────────────────────┐
│                    后续处理（Attention 等）                       │
└──────────────────────────────────────────────────────────────────┘
```

---

## 六、关键数据结构总结

### 6.1 prev_exists

- **类型**：`torch.BoolTensor`，形状 `(B,)`
- **含义**：标记每个样本是否有前一帧（是否连续）
- **取值**：
  - `True`：样本属于连续帧序列
  - `False`：样本切换到了新的 Clip

### 6.2 memory_mask

- **类型**：`torch.FloatTensor`，形状 `(B, T)`
- **含义**：标记每个样本的每帧是否有效
- **取值**：
  - `1.0`：该帧有效（包含真实数据）
  - `0.0`：该帧无效（占位，将被新帧覆盖）

### 6.3 update_counter

- **类型**：`int`
- **含义**：全局更新计数器（不分 batch）
- **作用**：控制降频采样（每2帧接收1帧进入低频区）

---

## 七、常见问题

### Q1: 为什么 Refresh 需要清零而不是删除旧帧？

**答**：PyTorch Tensor 的大小是固定的，不能动态删除某一行的数据。清零是最高效的方式，既标记了"无效"状态，又不需要重新分配内存。

### Q2: prev_exists 和 memory_mask 的区别是什么？

**答**：
- `prev_exists`：Batch 级别的标记，表示"该样本是否有前一帧"
- `memory_mask`：帧级别的标记，表示"该帧是否有效"

`prev_exists` 用于 Refresh（清零整个样本的 Memory），`memory_mask` 用于更新逻辑（判断高频区是否已满）。

### Q3: update_counter 为什么是全局的而不是 per-sample？

**答**：
- 降频采样是全局策略，不需要 per-sample 区分
- 简化实现，避免维护 `(B,)` 的计数器
- 如果 Batch 中不同样本需要独立的计数器，可以改为 `self.update_counters = torch.zeros(B)`

### Q4: 高低频分层模式下，如何判断高频区是否已满？

**答**：
```python
high_valid_count = prev_high_mask.sum(dim=1, keepdim=True)  # (B, 1)
high_is_full = (high_valid_count >= high_len).float()
```

通过 `memory_mask` 统计有效帧数，而不是检查是否有零值张量。这是更可靠的方法，因为零值张量也可能是有效数据（如空白帧）。

---

## 八、版本信息

| 版本 | 日期 | 说明 |
|------|------|------|
| v1.0 | 2025-01-08 | 初始版本 |
| v1.1 | 2025-01-08 | 新增高低频分层模式 |

---

**最后更新**: 2025-01-08
