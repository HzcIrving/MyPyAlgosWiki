# CLAUDE.md

重要规则：你的所有回复，无论任何情况，必须使用简体中文。
即使我用英文提问、代码注释是英文，你的思考过程和最终答案也必须是简体中文。

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 开发需求
❯ 现有一个已实现的时序融合模块（核心文件为temp_memory_bank.py），需基于该模块新增高频+低频时序帧管     
  理逻辑，并开发对应的gating网络完成多维度帧数据的融合。                                               
                                                                                                       
  【核心开发需求】                                                                                     
  1. 时序帧频率规则新增：                                                                              
  - 在memoryBank中实现时序帧的差异化频率管理：                                                         
  - 前3秒：采用2Hz频率，即每0.5秒生成/存储一帧数据；                                                   
  - 后3秒：采用1Hz频率，即每1秒生成/存储一帧数据；                                                     
  - 需保证帧的时间戳精准对应，且能正确区分“高频历史帧（前3s）”和“低频历史帧（后3s）”。                 
  2. Gating网络开发：                                                                                  
  - 参考DeepSeek Engram的gating机制（或同类成熟的时序数据门控方案），设计一个合理的gate网络；          
  - 该gate网络需实现“当前帧 + 高频历史帧 + 低频历史帧”的自适应融合；                                   
  - 网络结构需轻量化，适配现有memoryBank的计算逻辑，不引入冗余依赖。                                   
                                                                                                       
  【开发约束】                                                                                         
  1. 最小修改准则：仅新增必要的代码逻辑，不修改、不破坏memoryBank原有的核心逻辑（如token管理、队列     
  存储、时序融合的基础流程）；                                                                         
  2. 编码规范：所有新增/修改的代码必须严格符合PEP8规范（包括变量命名、行长度、缩进、注释格式等）；     
  3. 兼容性：新增逻辑需与原有代码无缝衔接，不出现变量名冲突、逻辑断点等问题。                          
                                                                                                       
  【输出要求】                                                                                         
  1. 输出完整的修改后代码（temp_memory_bank.py），用清晰的注释标注所有新增/修改的代码行，并注明修      
  改原因；                                                                                             
  2. 单独解释gating网络的设计思路：                                                                    
  - 参考的DeepSeek Engram核心要点；                                                                    
  - 本gate网络的结构、输入输出维度、核心计算逻辑；                                                     
  - 为何该设计适配“当前帧+高频+低频”的融合场景；                                                       
  3. 给出关键逻辑的测试建议（如如何验证频率规则、如何测试gate网络的融合效果）；                        
  4. 列出新增的依赖（如有），并说明安装方式。   

## Repository Overview

这是一个个人算法和机器学习知识库/参考仓库 (`MyPyAlgosWiki`)，包含了各种算法、数据结构和机器学习概念的 Python 实现。`TFModule` 子目录专门包含用于多视图感知系统的时间融合模块。

## 环境配置

### 主要依赖
```bash
# 核心依赖
conda create -n tfmodule python=3.8
conda activate tfmodule
conda install pytorch=2.0.1 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install easydict
```

### 依赖位置
- `../tokenizer/pinhole_tokenizer/model/attention.py`: Attention 模块
- `../tokenizer/pinhole_tokenizer/utils/position_embedding_3d.py`: 3D 位置编码
- `../MemoryLenFusion.py`: 简化的轨迹记忆融合示例

## TFModule: 时间融合模块

`TFModule` 目录包含 `temp_fusion_module.py`，实现了用于自动驾驶感知系统的 **PV (全景视图) 长期时间融合** 模块。该模块融合来自多个摄像头（前视、侧视、后视）的多时间帧信息。

### 核心组件

**PVLongTempFusionBaseModule** - 主要的时间融合类：
- **高低频分层记忆库**: 近期高频 (2Hz) + 远期低频 (1Hz) 分层存储
- **时空门控融合**: 自适应加权当前帧、高频历史、低频历史
- **Ego pose 对齐**: 将 3D 位置从当前帧显式变换到历史帧坐标系
- **连续时间编码**: 正弦位置编码处理浮点时间戳
- **双重注意力机制**:
  - `temporal_attn`: 跨帧的时间自注意力
  - `frame_attn`: 当前帧 tokens 的空间注意力
- **多摄像头支持**: 处理前视 (2个摄像头)、侧视 (4个摄像头)、后视 (1个摄像头)

### 架构流程 (更新)

```
输入: (B, N, C) 当前帧 tokens
     每个摄像头视图的 3D 位置嵌入
     Ego poses (4x4 变换矩阵)
     前一帧的记忆库

1. 记忆更新 (Shift + Downsample)
   ├─ 当前帧 → 高频记忆头部
   ├─ 高频末帧 → 低频候选
   └─ 低频记忆每2帧更新一次

2. 3D 位置嵌入计算
   └─ Ego Pose 显式变换 → 位置归一化 → MLP编码

3. 时间编码 + 门控
   ├─ 计算连续时间位置编码
   ├─ 门控网络生成三部分权重
   └─ Token' = (Token + Time_PE) × (1 + Weight)

4. 双重注意力融合
   ├─ 时间注意力: 跨帧 Self-Attention
   └─ 空间注意力: 当前帧内 Spatial Attention

输出: (B, N, C) 当前帧融合后的 tokens
```

### 配置参数

模块使用 `easydict.EasyDict` 进行配置，主要参数：

**频率分层参数** (新功能):
- `high_freq_len`: 高频历史帧数量 (如 6 帧，对应 0~3s @ 2Hz)
- `low_freq_len`: 低频历史帧数量 (如 3 帧，对应 3~6s @ 1Hz)
- `memory_len`: 自动计算为 `1 + high_freq_len + low_freq_len` (当前帧 + 高频 + 低频)

**基础参数**:
- `token_num`: 每帧的总 token 数 (前视/侧视/后视: 616 + 96 + 126)
- `feat_dim`: token 的特征维度
- `num_layers`: transformer 层数
- `motion_dim`: 运动特征维度
- `convertD`: 坐标变换模式的布尔值
- `pos_hidden_dim`: 位置嵌入的隐藏层维度 (默认 512)
- `perturb_ego_pose`: 训练时是否扰动 ego pose (默认 False)

**已弃用参数** (保留兼容性):
- `mem_compress`: 新功能已替代此功能，建议配置中关闭

### 记忆管理 (新架构)

模块实现**高低频分层记忆**策略：

**记忆布局**:
```
Index 0:         当前帧 (Current Frame)
Index 1~H:       高频历史 (High Freq History) - 每帧更新
Index H+1~End:   低频历史 (Low Freq History) - 每2帧更新 (降频采样)
```

**更新逻辑** (`_update_memory` 方法):
1. 当前帧 → 移入高频记忆的头部
2. 高频记忆的末尾帧 → 溢出，成为低频记忆的候选
3. 通过 `update_counter % 2 == 0` 控制降频: 偶数次接收溢出帧，奇数次保持不变

记忆库存储内容：
- `tokens`: (B, T, N, C) - token 特征
- `tokens_pos`: (B, T, N, C) - 位置嵌入
- `tps`: (B, T, 1) - 时间戳
- `ego_pose`: (B, T, 4, 4) - 变换矩阵
- `memory_mask`: (B, T) - 有效帧标志
- `counts`: (B, T) - 帧计数

### 坐标系统

模块使用显式 ego pose 变换进行时间对齐：
- 当前帧 3D 点变换到历史帧坐标系
- `hist_pos_3d = ego_pose_inv @ (memory_ego_pose @ pos_3d)`
- 3D 位置随后被归一化并编码为位置嵌入

### 辅助类

**ContinuousSinusoidalPosEmbed**: 连续时间位置编码
- 处理浮点时间戳 (如 delta_t = 0.55s)
- 输入: (B, T, 1) 时间增量
- 输出: (B, T, C) 正弦位置编码

**FrequencyGating**: 时空耦合门控网络 (新功能)
- 功能: 对当前帧、高频历史、低频历史三部分进行自适应加权
- 输入:
  - `current_token`: (B, N, C) 当前帧 token
  - `high_time_embed`: (B, C) 高频历史的平均时间编码
  - `low_time_embed`: (B, C) 低频历史的平均时间编码
- 输出: (B, 3) 三个权重 [w_cur, w_high, w_low]
- 应用方式: `token' = (token + time_pe) * (1.0 + weight)`

**TransformerEncoder**: Transformer 编码器封装
- 使用自定义 `Attention` 模块
- 支持多层堆叠

### 入口点

- `forward(tokens, tokens_pos, temp_fusion_inputs, ego_pose_inv)`: 非流式模式的主前向传播
- `forward_convertD(memory_pv_tokens, pos_3d_for_longtemp, memory_hist_to_cur)`: 坐标变换模式的替代前向传播
- `reset_memory()`: 清空记忆库
- `pre_update_memory(params)`: 为新帧准备记忆

### 调试和监控

- 模块包含调试钩子 (`_register_once_backward_hook`)，在首次反向传播后打印内存使用情况
- 支持训练时的 ego pose 扰动用于数据增强 (`perturb_ego_pose`)
- 位置嵌入使用学习的 MLP (`pos_embed`) 压缩 3D 坐标

### 内存使用统计

首次反向传播后，模块会自动打印：
- 模块参数数量 (百万)
- 参数内存 (MB)
- 梯度内存 (MB)
- 总内存 (MB)

### 新功能特性 (已实现)

**1. 高低频分层记忆** ✅
- 近期历史 (0~3s) 使用高频采样 (2Hz)，保持6帧
- 远期历史 (3~6s) 使用降频采样 (1Hz)，保持3帧
- 实现 `temp_fusion_module.py:277-363` 的 `_update_memory` 方法

**2. 时空门控网络** ✅
- 自适应学习当前帧、高频历史、低频历史的融合权重
- 结合视觉特征和时间上下文进行门控决策
- 实现 `temp_fusion_module.py:40-76` 的 `FrequencyGating` 类

**3. 连续时间编码** ✅
- 使用正弦位置编码处理连续时间戳
- 支持浮点时间增量 (如 0.55s)
- 实现 `temp_fusion_module.py:9-27` 的 `ContinuousSinusoidalPosEmbed` 类

## 相关文件

- `../MemoryLenFusion.py`: 用于规划的简单轨迹记忆融合示例
- `../streamPETR/`: 相关的多视图 3D 检测项目 (StreamPETR - ICCV 2023)
- `../README.md`: 项目的 IO 代码段和常用 NumPy 操作速查

## 开发注意事项

1. **相对导入**: 模块使用相对导入 `from ..tokenizer.pinhole_tokenizer...`，需要确保正确的包结构
2. **依赖外部模块**: 需要 `Attention` 和 `PositionEmbedding3D` 类来自 `tokenizer` 目录
3. **记忆状态管理**:
   - `reset_memory()` 应在序列开始时调用以清空记忆库
   - `update_counter` 跟踪帧数，用于控制低频记忆的降频采样
4. **训练 vs 推理**: ego pose 扰动仅在训练时启用 (`self.training` 为 True 时)
5. **维度管理**: 注意 `merge_pv_memory_tokens` 将多摄像头 tokens 合并为单一序列
6. **频率分层配置**:
   - 确保配置中设置 `high_freq_len` 和 `low_freq_len`
   - `memory_len` 会自动计算，无需手动设置
   - 建议关闭旧的 `mem_compress` 参数

### 数据流关键点

**记忆索引** (`temp_fusion_module.py:420-467`):
```python
# Memory 布局: [Current, High_Freq, Low_Freq]
cur_token = self.memory['tokens'][:, 0:1]
high_tokens = self.memory['tokens'][:, 1:1+high_freq_len]
low_tokens = self.memory['tokens'][:, 1+high_freq_len:]
```

**门控权重应用** (`temp_fusion_module.py:462-466`):
```python
# 权重作为乘数: token' = token × (1 + weight)
cur_weighted = cur_input * (1.0 + w_cur)
high_weighted = high_input * (1.0 + w_high)
low_weighted = low_input * (1.0 + w_low)
```
