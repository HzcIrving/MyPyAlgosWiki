import math
import torch
import torch.nn as nn
from easydict import EasyDict as edict
import torch.nn.functional as F
from ..tokenizer.pinhole_tokenizer.model.attention import Attention
from ..tokenizer.pinhole_tokenizer.utils.position_embedding_3d import PositionEmbedding3D 

class ContinuousSinusoidalPosEmbed(nn.Module):
    """
    连续时间位置编码 (Continuous Sinusoidal Position Embedding)
    处理 MBC 产生的浮点时间戳 (例如 delta_t = 0.55s)
    """
    def __init__(self, dim, scale=10000.0):
        super().__init__()
        self.dim = dim
        self.scale = scale
        inv_freq = 1.0 / (self.scale ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, time_delta):
        """
        time_delta: (B, T) 或 (B*N, T) 单位通常为秒
        pe: (B, T, C)
        """
        # (B, T, 1) * (D/2) -> (B, T, D/2)
        sin_inp = time_delta.unsqueeze(-1) * self.inv_freq # B*T 
        pe = torch.cat((sin_inp.sin(), sin_inp.cos()), dim=-1)
        return pe

class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = edict(config)
        self.num_layers = self.cfg.num_layers
        self.attn_encoder = nn.ModuleList([Attention(config) for _ in range(self.num_layers)])

    def forward(self, query, key, query_pos, key_pos, layer_idx):
        query = self.attn_encoder[layer_idx](query, key, None, query_pos=query_pos, key_pos=key_pos)[0]
        return query


# ============================================================================
# [子模块] TemporalFrequencyGating - 时序感知的频域门控模块
# 功能：在时序融合后，对当前帧、高频融合特征、低频融合特征进行自适应融合
# 设计思路：先让各部分经过时序 Attention 获取上下文，再进行 Gating 融合
# ============================================================================
class TemporalFrequencyGating(nn.Module):
    """
    时序感知的频域门控模块 (Temporal-Aware Frequency Gating Module)

    功能：
        在时序 Attention 融合后，对当前帧、高频历史融合特征、低频历史融合特征
        进行自适应加权融合

    设计思路：
        1. 各部分特征先经过时序 Attention，获得时序上下文
        2. 提取全局语义特征（空间池化）
        3. 融合时间上下文信息
        4. 通过轻量级 MLP 生成三部分融合权重
        5. 对各部分特征进行加权，实现自适应融合

    输入维度：
        - current_feat: (B, N, C) 经过时序融合的当前帧特征
        - high_freq_feat: (B, N, C) 经过时序融合的高频区特征
        - low_freq_feat: (B, N, C) 经过时序融合的低频区特征
        - high_time_emb: (B, C) 高频历史的时间编码
        - low_time_emb: (B, C) 低频历史的时间编码

    输出维度：
        - fused_feat: (B, N, C) 融合后的最终特征
        - gate_weights: (B, 3) 三部分融合权重 [w_cur, w_high, w_low]

    参数量：
        - hidden_dim = C // 2 时，约 2C² + 3C ≈ 0.5M (C=256)
    """

    def __init__(self, feat_dim, hidden_dim=None):
        """
        Args:
            feat_dim (int): 特征维度 C
            hidden_dim (int): 隐藏层维度，默认为 feat_dim // 2
        """
        super(TemporalFrequencyGating, self).__init__()
        hidden_dim = hidden_dim or feat_dim // 2

        # 门控网络：简单的两层 MLP
        # 输入：[全局视觉特征, 时间上下文] 拼接后的向量
        # 输出：3 个标量权重
        self.fc = nn.Sequential(
            nn.Linear(feat_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3)  # 输出 [w_cur, w_high, w_low]
        )

    def forward(self, current_feat, high_freq_feat, low_freq_feat, high_time_emb, low_time_emb):
        """
        Args:
            current_feat: (B, N, C) 经过时序融合的当前帧特征
            high_freq_feat: (B, N, C) 经过时序融合的高频区特征
            low_freq_feat: (B, N, C) 经过时序融合的低频区特征
            high_time_emb: (B, C) 高频历史的时间编码
            low_time_emb: (B, C) 低频历史的时间编码

        Returns:
            fused_feat: (B, N, C) 融合后的最终特征
            gate_weights: (B, 3) 三部分融合权重 [w_cur, w_high, w_low]
        """
        B, N, C = current_feat.shape

        # [步骤1] 提取当前帧的全局语义特征
        # 对空间维度 N 进行平均池化，得到 (B, C) 的全局描述
        global_feat = current_feat.mean(dim=1)  # (B, C)

        # [步骤2] 融合时间上下文（优化：处理空区域情况）
        # 判断各部分是否有有效帧（通过检查时间编码的模长）
        high_valid = (high_time_emb.abs().sum(dim=-1, keepdim=True) > 1e-6).float()  # (B, 1)
        low_valid = (low_time_emb.abs().sum(dim=-1, keepdim=True) > 1e-6).float()   # (B, 1)

        # 基于有效性进行加权平均，避免空区域拉低 time_context
        total_valid = high_valid + low_valid + 1e-6  # 避免除以 0
        time_context = (high_time_emb * high_valid + low_time_emb * low_valid) / total_valid  # (B, C)

        # [步骤3] 拼接视觉特征和时间上下文
        gate_input = torch.cat([global_feat, time_context], dim=-1)  # (B, 2C)

        # [步骤4] 通过 MLP 生成门控权重
        weights = self.fc(gate_input)  # (B, 3)

        # [步骤5] Softmax 归一化，确保权重和为 1
        weights = F.softmax(weights, dim=-1)  # (B, 3)

        # [步骤6] 提取三部分权重
        w_cur = weights[:, 0].view(B, 1, 1)  # (B, 1, 1)
        w_high = weights[:, 1].view(B, 1, 1)  # (B, 1, 1)
        w_low = weights[:, 2].view(B, 1, 1)  # (B, 1, 1)

        # [步骤7] 加权融合三部分特征
        fused_feat = w_cur * current_feat + w_high * high_freq_feat + w_low * low_freq_feat

        return fused_feat, weights

    def get_last_weights(self):
        """
        获取最后一次 forward 的门控权重（用于可视化/分析）

        Returns:
            weights: (B, 3) 或 None
        """
        # 这个方法需要在外部手动保存权重，这里仅作为接口定义
        return None


class PVLongTempFusionBaseModule(nn.Module):
    """
    Non-Stream模式 PV TempFusion | egoPose显式Align | 高低频分层记忆
    """
    def __init__(self, config) -> None:
        super().__init__()
        self.cfg = edict(config)

        # ============================================================================
        # [新增] 高低频分层记忆配置
        # ============================================================================
        # 启用开关：是否启用高低频分层功能（默认关闭，保持向后兼容）
        self.enable_hierarchical = self.cfg.get('enable_hierarchical', False)

        if self.enable_hierarchical:
            # 高低频分层模式
            # 高频区：前3秒，2Hz采样 → 6帧（每0.5秒一帧）
            # 低频区：后3秒，1Hz采样 → 3帧（每1秒一帧）
            # 总长度：1当前帧 + 6高频 + 3低频 = 10帧
            self.high_freq_len = self.cfg.get('high_freq_len', 6)
            self.low_freq_len = self.cfg.get('low_freq_len', 3)
            self.memory_len = 1 + self.high_freq_len + self.low_freq_len
            self.use_gating = self.cfg.get('use_gating', True)
        else:
            # 原始模式：单一频率
            self.memory_len = self.cfg.memory_len
            self.high_freq_len = 0
            self.low_freq_len = 0
            self.use_gating = False

        self.token_num = self.cfg.token_num  # (616 + 96 +126)
        self.feat_dim = self.cfg.tf_fuse_cfg.feat_dim
        self.num_layers = self.cfg.num_layers
        self.motion_dim = self.cfg.motion_dim
        self.use_temp_attn_mask = self.cfg.use_temp_attn_mask
        cross_modal_fuse_cfg = self.cfg.modal_fuse_cfg
        global_temp_fuse_cfg = self.cfg.tf_fuse_cfg
        self.convertD = self.cfg.convertD
        if self.convertD:
            global_temp_fuse_cfg.convertD = self.convertD
        self.temporal_attn = TransformerEncoder(global_temp_fuse_cfg)
        self.frame_attn = TransformerEncoder(cross_modal_fuse_cfg)

        front_pos_param = self.cfg.get('front_pos_param', None)
        side_pos_param = self.cfg.get('side_pos_param', None)
        rear_pos_param = self.cfg.get('rear_pos_param', None)
        self.front_pos_embed_3d = PositionEmbedding3D(front_pos_param)
        self.side_pos_embed_3d = PositionEmbedding3D(side_pos_param)
        self.rear_pos_embed_3d = PositionEmbedding3D(rear_pos_param)
        self.front_depth = front_pos_param.get('cam_depth', 96)
        self.perturb_ego_pose_flag = self.cfg.get('perturb_ego_pose', False)
        self.perturb_params = self.cfg.get('perturb_params', None)

        self.pos_hidden_dim = self.cfg.get('pos_hidden_dim', 512)
        self.mem_compress = self.cfg.get('mem_compress', False)
        self.pos_embed = nn.Sequential(
            nn.Linear(self.front_depth * 3, self.pos_hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_hidden_dim * 2, self.pos_hidden_dim)
        )
        self.pos_norm = nn.LayerNorm(self.pos_hidden_dim)

        # ============================================================================
        # [新增] 时间编码器和时序感知门控网络
        # ============================================================================
        if self.enable_hierarchical:
            # 连续时间位置编码器：处理浮点时间戳（如 0.5s, 1.0s）
            self.time_embedder = ContinuousSinusoidalPosEmbed(self.feat_dim)

            # Attention Weighted Pooling：可学习的 query 用于聚合多帧特征
            # 参考思路：类似 ViT 的 class token，但这里是用于时序维度的聚合
            self.high_freq_pool_query = nn.Parameter(torch.randn(1, 1, 1, self.feat_dim))
            self.low_freq_pool_query = nn.Parameter(torch.randn(1, 1, 1, self.feat_dim))
            # 缩放初始化
            nn.init.normal_(self.high_freq_pool_query, std=0.02)
            nn.init.normal_(self.low_freq_pool_query, std=0.02)

            # 时序感知门控网络：在时序融合后，自适应融合当前帧、高频历史、低频历史
            if self.use_gating:
                self.gating_net = TemporalFrequencyGating(
                    feat_dim=self.feat_dim,
                    hidden_dim=self.feat_dim // 2
                )
                # 用于保存最后一次的门控权重（用于可视化/分析）
                self.register_buffer('last_gate_weights', torch.zeros(1, 3))

        # ============================================================================
        # [新增] 更新计数器：用于实现低频区的降频采样
        # ============================================================================
        self.update_counter = 0

        self.reset_memory()
        ## logger
        self._printed_once = False
        self._register_once_backward_hook(module_name="PVLongTempFusion")

        self._init_params()

    def _init_params(self):
        scale = 1/math.sqrt(2*self.num_layers)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if any(x in name for x in ['ffn', 'proj']):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02 * scale)
                else:
                    nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                    if module.bias is not None:
                        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                        if fan_in != 0:
                            bound = 1 / math.sqrt(fan_in)
                            nn.init.uniform_(module.bias, -bound, bound)
            elif isinstance(module, nn.LayerNorm):
                if module.weight is not None: # 规避adaLN的layernorm
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _attention_weighted_pooling(self, tokens, query, key=None, mask=None):
        """
        Scaled Dot-Product Attention Weighted Pooling

        参考：
            - Set Transformer (JAIR 2019): 使用可学习的 induced points 聚合集合特征
            - Perceiver (ICLR 2022): 使用 latent array 通过 cross-attention 聚合输入
            - ViT (ICLR 2021): class token 机制

        核心思想：
            使用可学习的 query 作为"诱导点"(induced points)，
            通过标准的 Q-K-V attention 机制自适应地聚合时序特征。

        Args:
            tokens: (B, T, N, C) 时序特征，作为 Value (和 Key)
            query: (1, 1, 1, C) 或 (B, 1, 1, C) 可学习的 query
            key: (B, T, N, C) 或 None，如果为 None 则用 tokens 作为 key
            mask: (B, T) 可选的 mask，标记有效帧 (1=有效, 0=无效)

        Returns:
            pooled: (B, N, C) 聚合后的特征
            attn_weights: (B, T) attention 权重（用于可视化）
        """
        B, T, N, C = tokens.shape

        # 如果没有提供 key，使用 tokens 作为 key（类似 self-attention）
        if key is None:
            key = tokens

        # 扩展 query 维度
        if query.shape[0] == 1:
            query = query.expand(B, -1, -1, -1)  # (B, 1, 1, C)
        # Query 形状: (B, 1, N, C) - 在时序维度上只有 1 个 query

        # 将 tokens 重塑为 (B, N, T, C) 以便在时序维度 T 上做 attention
        # 这样每个 patch 可以独立地聚合时序信息
        tokens = tokens.permute(0, 2, 1, 3).contiguous()  # (B, N, T, C)
        key = key.permute(0, 2, 1, 3).contiguous()  # (B, N, T, C)

        # Scaled Dot-Product Attention
        # Q: (B, N, 1, C), K: (B, N, T, C) -> scores: (B, N, 1, T)
        scores = torch.matmul(query.unsqueeze(2), key.transpose(-2, -1)) / math.sqrt(C)
        scores = scores.squeeze(2)  # (B, N, T)

        # 应用 mask（如果有）
        if mask is not None:
            # mask: (B, T) -> (B, 1, T) -> (B, N, T)
            mask = mask.unsqueeze(1).expand(B, N, T)
            scores = scores.masked_fill(mask < 0.5, float('-inf'))

        # 在 T 维度上做 softmax，得到每帧的权重
        attn_weights = F.softmax(scores, dim=-1)  # (B, N, T)

        # 加权聚合
        # attn_weights: (B, N, T), tokens: (B, N, T, C)
        pooled = torch.matmul(attn_weights.unsqueeze(-1), tokens).squeeze(2)  # (B, N, C)

        # 返回聚合后的特征和平均后的 attention 权重（在 N 维度上平均）
        attn_weights_avg = attn_weights.mean(dim=1)  # (B, T)

        return pooled, attn_weights_avg

    def perturb_ego_pose(self, ego_pose, sigma_t=0.2, sigma_r_deg=2.0, prob=0.4):
        if self.convertD:
            return ego_pose, None 
        
        B = ego_pose.size(0)
        device = ego_pose.device
        dtype = ego_pose.dtype

        mask = (torch.rand(B, device=device) < prob).bool() # True: disturb
        mask_f = mask.float()
        yaw_noise = torch.randn(B, device=device, dtype=dtype) \
                    * math.radians(sigma_r_deg)
        yaw_noise = yaw_noise * mask_f
        cos_yaw = torch.cos(yaw_noise)
        sin_yaw = torch.sin(yaw_noise)
        R_yaw = torch.zeros(B, 3, 3, device=device, dtype=dtype)
        R_yaw[:, 0, 0] = cos_yaw
        R_yaw[:, 0, 1] = -sin_yaw
        R_yaw[:, 1, 0] = sin_yaw
        R_yaw[:, 1, 1] = cos_yaw
        R_yaw[:, 2, 2] = 1.0
        # Trans 
        z_noise = torch.zeros(B, device=device)
        t_noise = torch.randn(B, 3, device=device, dtype=dtype) * sigma_t
        t_noise[:, 2] = z_noise
        t_noise = t_noise * mask_f.unsqueeze(1)
        
        perturb = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)
        perturb[:, :3, :3] = R_yaw
        perturb[:, :3, 3] = t_noise
        ego_pose_perturbed = perturb @ ego_pose
        return ego_pose_perturbed, mask

    def get_pv_3dpe_with_hist(self, pos_3d, memory_ego_pose, ego_pose_inv, geo_params, n_cams):
        """
        把当前帧 front camera 的 3D 采样点，显式地通过 ego-pose 变换到历史帧坐标系下，
        再生成“历史帧对齐到当前帧”的 3D positional embedding，用于 hist→cur 的 attention 对齐。
        """
        h_pos = geo_params[0]
        w_pos = geo_params[1]
        pos_3d = pos_3d.reshape(self.b, -1, h_pos*w_pos*self.front_depth, 3)
        pos_3d = torch.cat((pos_3d, torch.ones_like(pos_3d[..., 0:1])), dim=3).permute(0, 1, 3, 2)
        memory_pos_embedding_total = []
        # [修改点13] 更新注释以反映尾插逻辑
        # 原头插: index 0: current frame, 1~T: historical frames
        # 现尾插: index 0~T-2: historical frames, index T-1: current frame (上一帧)
        for i in range(self.memory_len):
            hist_pos_3d = ego_pose_inv.unsqueeze(1) @ (memory_ego_pose[:, i].unsqueeze(1) @ pos_3d)
            hist_pos_3d = hist_pos_3d.permute(0, 1, 3, 2)
            hist_pos_3d = hist_pos_3d.reshape(self.b*n_cams, h_pos, w_pos, self.front_depth, -1)[...,:3].flatten(3, 4)
            hist_pos_3d_norm = self.front_pos_embed_3d.normalize_coords3d(hist_pos_3d)
            # hist_pos_3d_norm = hist_pos_3d_norm.reshape(self.b*n_cams, h_pos, w_pos, -1, 3).flatten(3, 4)
            memory_pos_embedding = self.pos_embed(hist_pos_3d_norm)
            memory_pos_embedding = self.pos_norm(memory_pos_embedding)
            memory_pos_embedding = memory_pos_embedding.flatten(1, 2).reshape(self.b, n_cams, h_pos* w_pos, -1)
            # memory_pos_embedding_fronts = [i[:, 0] for i in torch.split(memory_pos_embedding, [1, 1], dim=1)]
            memory_pos_embedding_total.append(memory_pos_embedding) 
        
        return memory_pos_embedding_total

    def merge_pv_memory_tokens(self, front_list, side_list, rear_list):
        T = len(front_list)
        per_t = []
        for t in range(T):
            front = front_list[t].squeeze(1)
            side  = side_list[t].flatten(1, 2)
            rear  = rear_list[t].squeeze(1)
            per_t.append(torch.cat([front, side, rear], dim=1))
        return torch.stack(per_t, dim=1)

    def truncate_pos3d(self, pos3d, ncams:int, cam_names: str=""):
        b_t_ncams, h, w, D = pos3d.shape
        b_t = b_t_ncams // ncams
        cur_pos_3d = pos3d.view(b_t, b_t_ncams//b_t, h, w, D) # front 1
        if cam_names != "side":
            cur_pos_3d = cur_pos_3d[:, 0] # 取front
            cur_pos_3d = cur_pos_3d.reshape(self.b, -1, h, w, D)
            cur_pos_3d = cur_pos_3d[:, -1] # 取当前帧
        else:
            cur_pos_3d = cur_pos_3d.reshape(self.b, -1, ncams, h, w, D)
            cur_pos_3d = cur_pos_3d[:, -1].reshape(-1, h, w, D) # 取当前帧
        return cur_pos_3d

    def forward(self, tokens, tokens_pos, temp_fusion_inputs, ego_pose_inv):
        prev_exists = temp_fusion_inputs.prev_exists 
        tps = temp_fusion_inputs.tps 
        ego_pose = temp_fusion_inputs.ego_pose 
        
        pre_update_params = edict(
            prev_exists=prev_exists,
            tps=tps
        )
        self.pre_update_memory(pre_update_params) 

        if self.perturb_ego_pose_flag and self.training:
            sigma_r_deg = self.perturb_params.sigma_r_deg
            sigma_t = self.perturb_params.sigma_t
            prob = self.perturb_params.prob
            ego_pose, _ = self.perturb_ego_pose(ego_pose, sigma_t, sigma_r_deg, prob)
            
        memory_inputs = edict(
            tokens=tokens,
            tokens_pos=tokens_pos,
            ego_pose=ego_pose,
            tps=tps, 
        )
        
        ## ----------- 压缩 or not -----------------------------
        if self.mem_compress:
            self._update_memory_with_compress(memory_inputs)
        else:
            self._update_memory(memory_inputs) 
        
        self.ego_pose_inv = ego_pose_inv
        cur_pos_3d_front = temp_fusion_inputs.pos_3d.pos_3d_front
        cur_pos_3d_rear = temp_fusion_inputs.pos_3d.pos_3d_rear
        cur_pos_3d_side = temp_fusion_inputs.pos_3d.pos_3d_side
        cur_pos_3d_front = self.truncate_pos3d(pos3d=cur_pos_3d_front, ncams=2, cam_names="front")
        cur_pos_3d_rear = self.truncate_pos3d(pos3d=cur_pos_3d_rear, ncams=1, cam_names="rear")
        cur_pos_3d_side = self.truncate_pos3d(pos3d=cur_pos_3d_side, ncams=4, cam_names="side")

        _, h_pos_front, w_pos_front, _ = cur_pos_3d_front.shape
        _, h_pos_rear, w_pos_rear, _ = cur_pos_3d_rear.shape
        _, h_pos_side, w_pos_side, _ = cur_pos_3d_side.shape

        memory_pos_embedding_front_total = self.get_pv_3dpe_with_hist(cur_pos_3d_front, self.memory.ego_pose, 
                                            ego_pose_inv, (h_pos_front, w_pos_front), n_cams=1) 
        memory_pos_embedding_rear_total = self.get_pv_3dpe_with_hist(cur_pos_3d_rear, self.memory.ego_pose, 
                                            ego_pose_inv, (h_pos_rear, w_pos_rear), n_cams=1) 
        memory_pos_embedding_sides_total = self.get_pv_3dpe_with_hist(cur_pos_3d_side, self.memory.ego_pose, 
                                            ego_pose_inv, (h_pos_side, w_pos_side), n_cams=4)

        memory_pos_embedding_fusion = self.merge_pv_memory_tokens(
                                        front_list=memory_pos_embedding_front_total,
                                        side_list=memory_pos_embedding_sides_total,
                                        rear_list=memory_pos_embedding_rear_total
                                    )

        # ============================================================================
        # [新增] 高低频分层 + 时序感知 Gating 融合逻辑
        # ============================================================================
        if self.enable_hierarchical:
            # ========================================================================
            # [高低频分层模式] 先时序融合，再 Gating
            # ========================================================================
            # Memory 尾插布局（按时间从远到近）:
            #   [低频历史(3帧), 高频历史(6帧), 当前帧(1帧)]
            #    Index 0-2      Index 3-8        Index 9
            #
            # 时间对应：
            #   Index 0-2: t-6.0, t-5.0, t-4.0 (低频，最远3秒)
            #   Index 3-8: t-3.0, t-2.5, t-2.0, t-1.5, t-1.0, t-0.5 (高频，0-3秒)
            #   Index 9:   当前帧 t
            #
            # 新的处理流程：
            #   1. 分离三部分：低频历史、高频历史、当前帧
            #   2. 计算并注入时间编码
            #   3. 拼接后进入 Temporal Attention（时序融合）
            #   4. Attention 后，再次分离三部分
            #   5. 通过 Gating 网络对融合后的特征进行加权融合
            # ========================================================================

            # [步骤1] 分离三部分（按尾插布局顺序）
            low_tokens = self.memory.tokens[:, 0:self.low_freq_len]  # (B, 3, N, C) 低频历史（最远3秒）
            high_tokens = self.memory.tokens[:, self.low_freq_len:self.low_freq_len + self.high_freq_len]  # (B, 6, N, C) 高频历史（0-3秒）
            cur_token = self.memory.tokens[:, -1:]  # (B, 1, N, C) 当前帧（最后一帧）

            # [步骤2] 计算时间编码并注入
            # 计算时间差：(当前时间 - 历史时间)
            time_deltas = temp_fusion_inputs.tps - self.memory['tps']  # (B, T, 1)

            # 生成正弦位置编码
            time_embs = self.time_embedder(time_deltas.squeeze(-1))  # (B, T, C)

            # 分离高频/低频时间编码（用于 Gating）
            low_time_embs = time_embs[:, 0:self.low_freq_len]  # (B, 3, C) 低频时间编码
            high_time_embs = time_embs[:, self.low_freq_len:self.low_freq_len + self.high_freq_len]  # (B, 6, C) 高频时间编码

            # 对高频/低频时间编码取平均（考虑 mask）
            low_mask = self.memory['memory_mask'][:, 0:self.low_freq_len].unsqueeze(-1)  # (B, 3, 1)
            high_mask = self.memory['memory_mask'][:, self.low_freq_len:self.low_freq_len + self.high_freq_len].unsqueeze(-1)  # (B, 6, 1)

            # 加权平均时间编码（排除无效帧）
            high_time_emb = (high_time_embs * high_mask).sum(dim=1) / (high_mask.sum(dim=1) + 1e-6)  # (B, C)
            low_time_emb = (low_time_embs * low_mask).sum(dim=1) / (low_mask.sum(dim=1) + 1e-6)  # (B, C)

            # [步骤3] 注入时间信息到 token
            # 扩展时间编码以匹配 token 的空间维度
            time_embs_expanded = time_embs.unsqueeze(2).expand_as(self.memory.tokens)  # (B, T, N, C)

            # 为三部分分别注入时间信息（按尾插顺序）
            low_tokens = low_tokens + time_embs_expanded[:, 0:self.low_freq_len]  # (B, 3, N, C)
            high_tokens = high_tokens + time_embs_expanded[:, self.low_freq_len:self.low_freq_len + self.high_freq_len]  # (B, 6, N, C)
            cur_token = cur_token + time_embs_expanded[:, -1:]  # (B, 1, N, C)

            # 拼接三部分：[低频历史, 高频历史, 当前帧]（按时间从远到近）
            tokens_fusion_temp = torch.cat([low_tokens, high_tokens, cur_token], dim=1)  # (B, 10, N, C)
            tokens_fusion_temp_pos = memory_pos_embedding_fusion
        else:
            # ========================================================================
            # [原始模式] 单一频率尾插逻辑
            # ========================================================================
            tokens_fusion_temp = self.memory.tokens[:, :-1]  # (B, T-1, N, C) 取历史帧
            tokens_fusion_temp = torch.cat([tokens_fusion_temp, tokens.unsqueeze(1)], dim=1)
            tokens_fusion_temp_pos = memory_pos_embedding_fusion

        # ============================================================================
        # [时序融合] Temporal Attention + Frame Attention
        # ============================================================================
        B, T, N, C = tokens_fusion_temp.shape
        for i in range(self.num_layers):
            tokens_fusion_temp = tokens_fusion_temp.permute(0, 2, 1, 3).flatten(0, 1) # (B*N, T, C)
            tokens_fusion_temp_pos = tokens_fusion_temp_pos.permute(0, 2, 1, 3).flatten(0, 1)

            # Temporal Self-Attention
            tokens_fusion_temp = self.temporal_attn(
                query=tokens_fusion_temp,
                key=tokens_fusion_temp,
                query_pos=tokens_fusion_temp_pos,
                key_pos=tokens_fusion_temp_pos,
                layer_idx=i)

            tokens_fusion_temp = tokens_fusion_temp.view(B, N, T, C).permute(0, 2, 1, 3)

            # 取当前帧进行 Frame Attention
            cur_tokens_fusion_temp = tokens_fusion_temp[:, -1, ...]
            cur_tokens_fusion_temp_pos = tokens_fusion_temp_pos.view(B, N, T, C).permute(0, 2, 1, 3)[:, -1, ...]

            # Cross Frame Self-Attention
            cur_tokens_fusion_temp = self.frame_attn(
                query=cur_tokens_fusion_temp,
                key=cur_tokens_fusion_temp,
                query_pos=cur_tokens_fusion_temp_pos,
                key_pos=cur_tokens_fusion_temp_pos,
                layer_idx=i)

            # 把处理后的当前帧放回
            tokens_fusion_temp = torch.cat([tokens_fusion_temp[:, :-1], cur_tokens_fusion_temp.unsqueeze(1)], dim=1)
            tokens_fusion_temp = tokens_fusion_temp.reshape(B, T, N, C)
            tokens_fusion_temp_pos = tokens_fusion_temp_pos.reshape(B, T, N, C)

        # ============================================================================
        # [高低频分层模式] 时序融合后的 Gating 融合
        # ============================================================================
        if self.enable_hierarchical and self.use_gating:
            # 从时序融合后的特征中分离三部分（按尾插布局）
            # 低频区特征：Index 0-2
            low_freq_tokens = tokens_fusion_temp[:, 0:self.low_freq_len]  # (B, 3, N, C)
            # 高频区特征：Index 3-8
            high_freq_tokens = tokens_fusion_temp[:, self.low_freq_len:self.low_freq_len + self.high_freq_len]  # (B, 6, N, C)
            # 当前帧特征：Index 9（最后一帧）
            current_feat = tokens_fusion_temp[:, -1]  # (B, N, C)

            # 使用 Attention Weighted Pooling 聚合多帧特征
            # 获取对应的 mask
            low_freq_mask = self.memory['memory_mask'][:, 0:self.low_freq_len]  # (B, 3)
            high_freq_mask = self.memory['memory_mask'][:, self.low_freq_len:self.low_freq_len + self.high_freq_len]  # (B, 6)

            # 对低频区进行 Attention Weighted Pooling
            low_freq_fused, low_attn_weights = self._attention_weighted_pooling(
                tokens=low_freq_tokens,
                query=self.low_freq_pool_query,
                mask=low_freq_mask
            )  # (B, N, C)

            # 对高频区进行 Attention Weighted Pooling
            high_freq_fused, high_attn_weights = self._attention_weighted_pooling(
                tokens=high_freq_tokens,
                query=self.high_freq_pool_query,
                mask=high_freq_mask
            )  # (B, N, C)

            # 通过 Gating 网络进行自适应融合
            fused_feat, gate_weights = self.gating_net(
                current_feat=current_feat,
                high_freq_feat=high_freq_fused,
                low_freq_feat=low_freq_fused,
                high_time_emb=high_time_emb,
                low_time_emb=low_time_emb
            )

            # 保存权重用于可视化/分析
            self.last_gate_weights = gate_weights.detach()
            # 可选：保存 attention 权重
            self.last_high_freq_attn_weights = high_attn_weights.detach()
            self.last_low_freq_attn_weights = low_attn_weights.detach()

            # 返回融合后的特征
            return fused_feat
        else:
            # [原始模式或不使用 Gating] 返回当前帧
            return tokens_fusion_temp[:, -1]

    def get_pv_3dpe_with_hist_convertD(self, pos_3d, memory_hist_to_cur, geo_params, n_cams):
        h_pos = geo_params[0]
        w_pos = geo_params[1]
        pos_3d = pos_3d.reshape(self.b, -1, h_pos*w_pos*self.front_depth, 3)
        pos_3d = torch.cat((pos_3d, torch.ones_like(pos_3d[..., 0:1])), dim=3).permute(0, 1, 3, 2)
        memory_pos_embedding_total = []
        # [修改点14] forward_convertD 中的注释也需更新
        # 原头插: index 0: current frame, 1~T: historical frames
        # 现尾插: index 0~T-2: historical frames, index T-1: current frame (上一帧)
        for i in range(self.memory_len):
            hist_pos_3d = memory_hist_to_cur[:, i] @ pos_3d
            hist_pos_3d = hist_pos_3d.permute(0, 1, 3, 2)
            hist_pos_3d = hist_pos_3d.reshape(self.b*n_cams, h_pos, w_pos, self.front_depth, -1)[...,:3].flatten(3, 4)
            hist_pos_3d_norm = self.front_pos_embed_3d.normalize_coords3d(hist_pos_3d)
            # hist_pos_3d_norm = hist_pos_3d_norm.reshape(self.b*n_cams, h_pos, w_pos, -1, 3).flatten(3, 4)
            memory_pos_embedding = self.pos_embed(hist_pos_3d_norm)
            memory_pos_embedding = self.pos_norm(memory_pos_embedding)
            memory_pos_embedding = memory_pos_embedding.flatten(1, 2).reshape(self.b, n_cams, h_pos* w_pos, -1)
            # memory_pos_embedding_fronts = [i[:, 0] for i in torch.split(memory_pos_embedding, [1, 1], dim=1)]
            memory_pos_embedding_total.append(memory_pos_embedding) 
        
        return memory_pos_embedding_total

    def forward_convertD(self, memory_pv_tokens, pos_3d_for_longtemp, memory_hist_to_cur) :
        # 转D这里直接是t5->t1迭帧; TODO 
        with torch.no_grad():
            b = memory_pv_tokens.size()[0]
            self.b = b
            cur_pos_3d_front = pos_3d_for_longtemp['pos_3d_front']
            bncams, h, w, D = cur_pos_3d_front.shape 
            cur_pos_3d_front = cur_pos_3d_front.view(b, bncams//b, h, w, D)[:, 0] # front 1
            cur_pos_3d_rear = pos_3d_for_longtemp['pos_3d_rear']
            cur_pos_3d_side = pos_3d_for_longtemp['pos_3d_side']
            _, h_pos_front, w_pos_front, _ = cur_pos_3d_front.shape
            _, h_pos_rear, w_pos_rear, _ = cur_pos_3d_rear.shape
            _, h_pos_side, w_pos_side, _ = cur_pos_3d_side.shape 
            
            # pos_3d, memory_hist_to_cur, geo_params, n_cams
            memory_pos_embedding_front_total = self.get_pv_3dpe_with_hist_convertD(cur_pos_3d_front, memory_hist_to_cur, (h_pos_front, w_pos_front), n_cams=1) 
            memory_pos_embedding_rear_total = self.get_pv_3dpe_with_hist_convertD(cur_pos_3d_rear, memory_hist_to_cur, (h_pos_rear, w_pos_rear), n_cams=1) 
            memory_pos_embedding_sides_total = self.get_pv_3dpe_with_hist_convertD(cur_pos_3d_side, memory_hist_to_cur, (h_pos_side, w_pos_side), n_cams=4) 
            
            memory_pos_embedding_fusion = self.merge_pv_memory_tokens(
                                        front_list =memory_pos_embedding_front_total, 
                                        side_list=memory_pos_embedding_sides_total,
                                        rear_list=memory_pos_embedding_rear_total
                                    )
            
            tokens_fusion_temp = memory_pv_tokens
            # hist_tokens_pos = self.memory.tokens_pos      # (B, T, N, C)
            tokens_fusion_temp_pos = memory_pos_embedding_fusion  
            
            B, T, N, C = tokens_fusion_temp.shape
            for i in range(self.num_layers): 
                tokens_fusion_temp = tokens_fusion_temp.permute(0, 2, 1, 3).flatten(0, 1) # (B*N, T, C)
                tokens_fusion_temp_pos = tokens_fusion_temp_pos.permute(0, 2, 1, 3).flatten(0, 1)
                # Temporal CA | query, key, query_pos, key_pose, layer_idx
                # [修改点9] forward_convertD 也需要适配尾插逻辑
                # 原头插注释：
                #   tokens_fusion_current = tokens_fusion_temp[:, :1]    # index 0 是当前帧
                #   tokens_fusion_history = tokens_fusion_temp[:, 1:]
                # 现尾插：当前帧在 index -1
                tokens_fusion_temp = self.temporal_attn(
                    query=tokens_fusion_temp,
                    key=tokens_fusion_temp,
                    query_pos=tokens_fusion_temp_pos,
                    key_pos=tokens_fusion_temp_pos,
                    layer_idx=i) # (B*N, T, C) SA

                tokens_fusion_temp = tokens_fusion_temp.view(B, N, T, C).permute(0, 2, 1, 3)
                # [修改点10] 取当前帧（从 index 0 改为 index -1）
                cur_tokens_fusion_temp = tokens_fusion_temp[:, -1, ...]
                cur_tokens_fusion_temp_pos = tokens_fusion_temp_pos.view(B, N, T, C).permute(0, 2, 1, 3)[:, -1, ...]
                # Cross Frame SA
                cur_tokens_fusion_temp = self.frame_attn(
                    query=cur_tokens_fusion_temp,
                    key=cur_tokens_fusion_temp,
                    query_pos=cur_tokens_fusion_temp_pos,
                    key_pos=cur_tokens_fusion_temp_pos,
                    layer_idx=i) # (B, N, C) SA

                # [修改点11] 把处理后的当前帧放回（从 index 0 改为 index -1）
                tokens_fusion_temp = torch.cat([tokens_fusion_temp[:, :-1], cur_tokens_fusion_temp.unsqueeze(1)], dim=1)
                tokens_fusion_temp = tokens_fusion_temp.reshape(B, T, N, C)
                tokens_fusion_temp_pos = tokens_fusion_temp_pos.reshape(B, T, N, C)

        # [修改点12] 返回当前帧（从 index 0 改为 index -1）
        return tokens_fusion_temp[:, -1]

    def reset_memory(self):
        """
        重置记忆库

        [新增] 高低频分层模式：同时重置更新计数器
        """
        self.memory = None
        self.update_counter = 0

    def pre_update_memory(self, params):
        prev_exists = params.prev_exists
        tps = params.tps
        B = prev_exists.shape[0]
        self.b = prev_exists.shape[0]
        device = prev_exists.device
        eye = torch.eye(4, device=device, dtype=prev_exists.dtype)

        if (not self.memory) or (not prev_exists.any()):
            self.memory = edict({
                'tokens': prev_exists.new_zeros(B, self.memory_len, self.token_num, self.feat_dim),
                'tokens_pos': prev_exists.new_zeros(B, self.memory_len, self.token_num, self.feat_dim),
                'tps': tps.new_zeros(B, self.memory_len, 1),
                # 'ego_pose': eye.view(1, 1, 4, 4).repeat(B, self.memory_len, 1, 1),
                'ego_pose': prev_exists.new_zeros(B, self.memory_len, 4, 4),
                'memory_mask': torch.zeros((B, self.memory_len), device=device, dtype=torch.float32),
                'counts': torch.ones((B, self.memory_len), device=device, dtype=torch.float32)
            })

        for key in self.memory.keys():
            # if key == 'ego_pose':
            #     self.memory['ego_pose'] = self.refresh_ego_pose(
            #         self.memory['ego_pose'][:, :self.memory_len], prev_exists
            #     )
            # else:
            self.memory[key] = self.refresh_memory(
                    self.memory[key][:, :self.memory_len], prev_exists
                )

    def merge_memory(self, raw_memory_dict, compressed_memory, merge_indices, counts, B):
        ## 帧间压缩 
        device = merge_indices 
        for key, tensor in raw_memory_dict.items():
            new_tensor_list = [] 
            for b in range(B): 
                idx = merge_indices[b] 
                t_tensor = tensor[b] # (T+1)

                # 基于merge_indices来切片 
                part_a = t_tensor[:idx]
                item1  = t_tensor[idx]
                item2  = t_tensor[idx+1]
                part_c = t_tensor[idx+2:]
                
                if key == 'counts':
                    merged_item = item1 + item2
                elif key == 'memory_mask':
                    # Mask 保持为 1
                    merged_item = item1 
                else:
                    c1 = counts[b, idx]
                    c2 = counts[b, idx+1]
                    total_c = c1 + c2 + 1e-6 # 计数加权
                    merged_item = (item1 * c1 + item2 * c2) / total_c # 加权平均
                
                # 重新拼接: T+1 -> T
                new_t_tensor = torch.cat([part_a, merged_item.unsqueeze(0), part_c], dim=0)
                new_tensor_list.append(new_t_tensor)
            compressed_memory[key] = torch.stack(new_tensor_list, dim=0)
        return compressed_memory 
            
    def compress_memory_bank(self, raw_memory_dict):
        """
        1. 非压缩模式 - Bank还未存满，此时数据继续进 
        2. 压缩模式 - Bank存满，需要进行压缩
        """
        tokens = raw_memory_dict['tokens']    # (B, T+1, N, C)
        mask   = raw_memory_dict['memory_mask'] # (B, T+1)
        counts = raw_memory_dict['counts']  # (B, T+1) 
        B, _, N, C = tokens.shape
        if_full_flag = (mask[:, -1] > 0.5).all() 
        
        if not if_full_flag:
            compressed_memory = edict() 
            for k, v in raw_memory_dict.items():
                compressed_memory.update({k: v[:, :-1]}) # 不足 memorylen 帧, 直接尾删，插入最新帧
            return compressed_memory 

        flatten_tokens = tokens.reshape(B, self.memory_len, -1) # (B, T, N*C) 
        curr_frames = flatten_tokens[:, :-1, :]
        next_frames = flatten_tokens[:, 1:, :]
        sims = F.cosine_similarity(curr_frames, next_frames, dim=-1) 
        
        # 排序找到最相似 
        merge_indices = torch.argmax(sims, dim=-1)  
        compressed_memory = edict() 
        compressed_memory = self.merge_memory(raw_memory_dict, compressed_memory, merge_indices, counts, B)
        
        return compressed_memory 
    
    def _update_memory_with_compress(self, inputs):
        """
        [修正] MA-LMM Memory Compress - 尾插版本

        修正说明：原代码使用头插 [新帧, 旧帧]，现改为尾插 [旧帧, 新帧]
        以与其他部分的尾插逻辑保持一致。
        """
        B = inputs.tokens.size(0)
        device = inputs.tokens.device

        if 'counts' not in self.memory:
            self.memory['counts'] = torch.ones_like(self.memory['memory_mask'])

        raw_memory = edict()
        target_keys = ['tokens', 'tokens_pos', 'tps', 'ego_pose']
        for key in target_keys:
            if key in inputs:
                curr_val = inputs[key].unsqueeze(1).detach()
                hist_val = self.memory[key]
                # 修正：尾插逻辑 [旧帧, 新帧]
                raw_memory[key] = torch.cat([hist_val, curr_val], dim=1)

        # 拼接 Mask (新帧 valid=1) - 尾插
        new_mask = torch.ones((B, 1), device=device, dtype=torch.float32)
        raw_memory['memory_mask'] = torch.cat([self.memory['memory_mask'], new_mask], dim=1)

        # 拼接 Counts (新帧 count=1) - 尾插
        new_count = torch.ones((B, 1), device=device, dtype=self.memory['counts'].dtype)
        raw_memory['counts'] = torch.cat([self.memory['counts'], new_count], dim=1)
        
        ## ---------------  COMPRESS核心修改 -------------------------- 
        compressed_memory = self.compress_memory_bank(raw_memory) 
        ## ------------------------------------------------------------
        
        self.memory = compressed_memory 

    def _update_memory_standard(self, inputs):
        """
        [原始模式] 单一频率尾插逻辑

        Memory 布局: [历史1, 历史2, ..., 历史N-1, 上一帧]
        更新方式:   丢弃最老的历史帧，新帧追加到尾部
        新布局:     [历史2, ..., 历史N-1, 上一帧, 新帧]

        Args:
            inputs: 包含 tokens, tokens_pos, ego_pose, tps 的新帧数据
        """
        B = inputs.tokens.size(0)
        device = inputs.tokens.device

        for key in inputs.keys():
            # 尾插：保留旧帧的第1个到最后一个，新帧放到最后
            self.memory[key] = torch.cat([self.memory[key][:, 1:],
                                          inputs[key].unsqueeze(1).detach()], dim=1)

        # mask 也采用尾插：新帧的 mask 放到最后
        self.memory['memory_mask'] = torch.cat([self.memory['memory_mask'][:, 1:],
                                                 torch.ones((B, 1), device=device, dtype=torch.float32)], dim=1)

    def _update_memory_hierarchical(self, inputs):
        """
        [优化版] 高低频分层记忆更新逻辑（级联队列实现）

        Memory 尾插布局（按时间从远到近）:
          [低频历史(3帧), 高频历史(6帧), 当前帧(1帧)]
           Index 0-2      Index 3-8        Index 9

        更新策略（级联队列，FIFO左移）:
          1. 新当前帧 → Index 9（尾插，最后一帧）
          2. 旧当前帧（Index 9）→ 高频区尾部（Index 8）
          3. 高频区头部（Index 3）溢出 → 低频区（降频）
          4. 低频区头部（Index 0）丢弃

        相比原版优势：
          - 简化逻辑：移除复杂的空位查找和 gather 操作
          - 更高效：直接切片拼接，无需条件判断
          - 更清晰：级联队列逻辑符合时序流动直觉

        Args:
            inputs: 包含 tokens, tokens_pos, ego_pose, tps 的新帧数据
        """
        B = inputs.tokens.size(0)
        device = inputs.tokens.device
        high_len = self.high_freq_len
        low_len = self.low_freq_len

        # ========== Step 1: 准备新数据 ==========
        target_keys = ['tokens', 'tokens_pos', 'tps', 'ego_pose']
        new_cur_data = {k: inputs[k].unsqueeze(1).detach() for k in target_keys if k in inputs}
        new_cur_mask = torch.ones((B, 1), device=device, dtype=torch.float32)

        # ========== Step 2: 切片现有 Memory（按尾插布局顺序） ==========
        # 布局: [低频历史(0:low_len), 高频历史(low_len:low_len+high_len), 当前帧(-1:)]
        prev_low_data = {k: self.memory[k][:, 0:low_len] for k in target_keys}
        prev_low_mask = self.memory['memory_mask'][:, 0:low_len]
        prev_high_data = {k: self.memory[k][:, low_len:low_len + high_len] for k in target_keys}
        prev_high_mask = self.memory['memory_mask'][:, low_len:low_len + high_len]
        prev_cur_data = {k: self.memory[k][:, -1:] for k in target_keys}  # 最后一帧是当前帧
        prev_cur_mask = self.memory['memory_mask'][:, -1:]

        # ========== Step 3: 级联更新 ==========
        # 3.1 高频区更新：左移，旧当前帧补到尾部
        #     [HF0, HF1, HF2, HF3, HF4, HF5] -> [HF1, HF2, HF3, HF4, HF5, Old_Cur]
        #     高频区头部（Index 3）溢出
        new_high_data = {}
        new_high_mask = torch.zeros((B, high_len), device=device)

        for k in target_keys:
            if high_len > 0:
                # 左移：丢弃头部（Index 0），尾部补旧当前帧
                new_high_data[k] = torch.cat([prev_high_data[k][:, 1:], prev_cur_data[k]], dim=1)
                # Mask 同样左移，补 1（旧当前帧总是有效）
                new_high_mask = torch.cat([prev_high_mask[:, 1:], prev_cur_mask.squeeze(-1).unsqueeze(1)], dim=1)
            else:
                new_high_data[k] = torch.zeros(B, 0, *prev_cur_data[k].shape[2:], device=device, dtype=prev_cur_data[k].dtype)
                new_high_mask = torch.zeros((B, 0), device=device)

        # 3.2 低频区更新：降频采样（每2帧接收1次溢出）
        self.update_counter += 1

        if low_len > 0 and self.update_counter % 2 == 0:
            # 偶数次：接收溢出
            # 低频区左移：[LF0, LF1, LF2] -> [LF1, LF2, HF_overflow]
            new_low_data = {}
            for k in target_keys:
                # 高频溢出（Index 0）补到低频区尾部
                high_overflow = prev_high_data[k][:, :1]  # 高频区头部溢出
                new_low_data[k] = torch.cat([prev_low_data[k][:, 1:], high_overflow], dim=1)

            # Mask 更新
            high_overflow_mask = prev_high_mask[:, :1]  # 高频区头部 mask
            new_low_mask = torch.cat([prev_low_mask[:, 1:], high_overflow_mask], dim=1)
        else:
            # 奇数次或 low_len=0：保持不变
            new_low_data = prev_low_data
            new_low_mask = prev_low_mask

        # ========== Step 4: 更新 Memory ==========
        # 尾插布局：按时间从远到近 [低频历史, 高频历史, 当前帧]
        for k in target_keys:
            self.memory[k] = torch.cat([new_low_data[k], new_high_data[k], new_cur_data[k]], dim=1)

        self.memory['memory_mask'] = torch.cat([new_low_mask, new_high_mask, new_cur_mask], dim=1)

    def _update_memory(self, inputs):
        """
        [核心] 记忆库更新逻辑 - 分发器

        === 模式选择 ===
        根据 enable_hierarchical 标志选择更新策略：
        - False: 使用原始尾插逻辑（向后兼容）
        - True: 使用高低频分层逻辑（新功能）

        类似于 _update_memory_with_compress 的设计模式，通过开关控制调用哪个具体实现。

        Args:
            inputs: 包含 tokens, tokens_pos, ego_pose, tps 的新帧数据
        """
        if not self.enable_hierarchical:
            self._update_memory_standard(inputs)
        else:
            self._update_memory_hierarchical(inputs)

    def refresh_memory(self, memory_ele, prev_exists):
        view_shape = [-1] + [1] * (memory_ele.dim() - 1)
        # 将 prev_exists 转换为与 memory_ele 相同的数据类型
        prev_exists = prev_exists.to(memory_ele.dtype)
        prev_exists = prev_exists.view(*view_shape)
        return memory_ele * prev_exists

    def refresh_ego_pose(self, ego_pose, prev_exists):
        """
        prev_exists 1 ->  keep ego_pose
        prev_exists 0 ->  eye(4,4)
        """
        B, T, _, _ = ego_pose.shape
        device = ego_pose.device
        dtype = ego_pose.dtype
        # (B, 1, 1, 1)
        mask = prev_exists.view(B, 1, 1, 1).to(dtype)
        eye = torch.eye(4, device=device, dtype=dtype)\
                .view(1, 1, 4, 4)
        return ego_pose * mask + eye * (1.0 - mask)

    def _register_once_backward_hook(self, module_name):
        def _hook(grad):
            if not self._printed_once:
                self._printed_once = True
                param_count = self._get_param_count()
                mem = self._get_temp_memory_usage()
                print("\n================= Model Stats After First Backward =================")
                print(f"{module_name} Module Parameters: {param_count:.3f} M")
                print(f"{module_name} Module Memory: {mem}")
                print("===================================================================\n")
            return grad

        for p in self.parameters():
            if p.requires_grad:
                p.register_hook(_hook)
                break

    def _get_temp_memory_usage(self):
        param_mem = 0
        grad_mem = 0
        for name, p in self.named_parameters(recurse=True):
            if p.data is not None:
                param_mem += p.data.nelement() * p.data.element_size()
            if p.grad is not None:
                grad_mem += p.grad.nelement() * p.grad.element_size()
        return {
            "param_MB": round(param_mem / 1024 / 1024, 2),
            "grad_MB": round(grad_mem / 1024 / 1024, 2),
            "total_MB": round((param_mem + grad_mem) / 1024 / 1024, 2),
        }

    def _get_param_count(self):
        total = 0
        for p in self.parameters():
            total += p.numel()
        return round(total / 1e6, 2)