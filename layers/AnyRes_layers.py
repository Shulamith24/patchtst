"""
AnyRes Layers for Multi-Scale Time Series Processing
Adapted from LLaVA-NeXT's AnyRes concept for PatchTST
"""

__all__ = ['MultiScalePatchEmbedding', 'CrossScaleAttention', 'AnyResHead']

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple


class MultiScalePatchEmbedding(nn.Module):
    """
    多尺度Patch嵌入层
    将输入时间序列按多个尺度进行Patching，类似LLaVA AnyRes的多分辨率子图
    
    Args:
        c_in: 输入通道数
        context_window: 输入序列长度
        d_model: 嵌入维度
        patch_scales: 多尺度patch配置列表，如 [8, 16, 32]
        use_global_token: 是否使用全局token（类似thumbnail）
        pe: 位置编码类型
        learn_pe: 是否学习位置编码
    """
    
    def __init__(
        self,
        c_in: int,
        context_window: int,
        d_model: int,
        patch_scales: List[int] = [8, 16, 32],
        use_global_token: bool = True,
        pe: str = 'zeros',
        learn_pe: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.c_in = c_in
        self.context_window = context_window
        self.d_model = d_model
        self.patch_scales = patch_scales
        self.use_global_token = use_global_token
        
        # 每个尺度的patch投影层
        self.patch_projections = nn.ModuleDict()
        self.patch_nums = {}
        
        for patch_len in patch_scales:
            stride = patch_len  # 默认使用non-overlapping patches
            patch_num = (context_window - patch_len) // stride + 1
            self.patch_nums[patch_len] = patch_num
            
            # 线性投影: patch_len -> d_model
            self.patch_projections[str(patch_len)] = nn.Linear(patch_len, d_model)
        
        # 尺度嵌入（区分不同尺度的patches）
        self.scale_embeddings = nn.ParameterDict({
            str(patch_len): nn.Parameter(torch.zeros(1, 1, 1, d_model))
            for patch_len in patch_scales
        })
        
        # 位置编码（每个尺度独立）
        self.pos_encodings = nn.ParameterDict()
        for patch_len in patch_scales:
            patch_num = self.patch_nums[patch_len]
            W_pos = self._create_pos_encoding(patch_num, d_model, pe)
            self.pos_encodings[str(patch_len)] = nn.Parameter(W_pos, requires_grad=learn_pe)
        
        # 全局token（类似AnyRes的thumbnail）
        if use_global_token:
            self.global_proj = nn.Sequential(
                nn.Linear(context_window, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            self.scale_embeddings['global'] = nn.Parameter(torch.zeros(1, 1, 1, d_model))
        
        self.dropout = nn.Dropout(dropout)
        
        # 初始化
        self._init_weights()
    
    def _create_pos_encoding(self, q_len: int, d_model: int, pe: str) -> Tensor:
        """创建位置编码"""
        if pe == 'zeros':
            W_pos = torch.empty((q_len, d_model))
            nn.init.uniform_(W_pos, -0.02, 0.02)
        elif pe == 'sincos':
            W_pos = torch.zeros(q_len, d_model)
            position = torch.arange(0, q_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
            W_pos[:, 0::2] = torch.sin(position * div_term)
            W_pos[:, 1::2] = torch.cos(position * div_term)
            W_pos = W_pos - W_pos.mean()
            W_pos = W_pos / (W_pos.std() * 10)
        else:
            W_pos = torch.empty((q_len, d_model))
            nn.init.uniform_(W_pos, -0.02, 0.02)
        return W_pos
    
    def _init_weights(self):
        """初始化权重"""
        for name, param in self.scale_embeddings.items():
            nn.init.zeros_(param)
        
        for name, module in self.patch_projections.items():
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            x: [B, C, L] 输入时间序列 (Channel-first format)
        
        Returns:
            Dict[str, Tensor]: 每个尺度的嵌入
                - str(patch_len): [B, C, patch_num, d_model]
                - 'global': [B, C, 1, d_model] (if use_global_token)
        """
        B, C, L = x.shape
        outputs = {}
        
        # 1. 多尺度Patch嵌入
        for patch_len in self.patch_scales:
            stride = patch_len
            
            # Patching: [B, C, L] -> [B, C, patch_num, patch_len]
            patches = x.unfold(dimension=-1, size=patch_len, step=stride)
            
            # 投影: [B, C, patch_num, patch_len] -> [B, C, patch_num, d_model]
            embedded = self.patch_projections[str(patch_len)](patches)
            
            # 添加位置编码和尺度嵌入
            pos_enc = self.pos_encodings[str(patch_len)]  # [patch_num, d_model]
            scale_emb = self.scale_embeddings[str(patch_len)]  # [1, 1, 1, d_model]
            
            embedded = embedded + pos_enc.unsqueeze(0).unsqueeze(0)  # 位置编码
            embedded = embedded + scale_emb  # 尺度嵌入
            embedded = self.dropout(embedded)
            
            outputs[str(patch_len)] = embedded
        
        # 2. 全局token（类似thumbnail）
        if self.use_global_token:
            # 对每个通道做全局投影
            global_feat = self.global_proj(x)  # [B, C, d_model]
            global_feat = global_feat.unsqueeze(2)  # [B, C, 1, d_model]
            global_feat = global_feat + self.scale_embeddings['global']
            outputs['global'] = global_feat
        
        return outputs
    
    def get_total_patches(self) -> int:
        """获取所有尺度的总patch数"""
        total = sum(self.patch_nums.values())
        if self.use_global_token:
            total += 1
        return total


class CrossScaleAttention(nn.Module):
    """
    跨尺度注意力融合模块
    将不同尺度的特征通过自注意力机制融合
    
    Args:
        d_model: 嵌入维度
        n_heads: 注意力头数
        n_layers: Transformer层数
        d_ff: FFN维度
        dropout: dropout率
        norm: 归一化类型 ('BatchNorm' or 'LayerNorm')
    """
    
    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        norm: str = 'LayerNorm',
        res_attention: bool = True
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        # 跨尺度Transformer层
        self.layers = nn.ModuleList([
            CrossScaleTransformerLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                norm=norm,
                res_attention=res_attention
            )
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        multi_scale_features: Dict[str, Tensor],
        return_separated: bool = False
    ) -> Tensor:
        """
        Args:
            multi_scale_features: Dict[str, Tensor]
                每个尺度的特征 [B, C, N_i, d_model]
            return_separated: 是否返回分离的尺度特征
        
        Returns:
            fused: [B, C, N_total, d_model] 融合后的特征
        """
        # 记录每个尺度的patch数量（用于后续分离）
        scale_sizes = {}
        all_features = []
        
        for scale_name, feat in multi_scale_features.items():
            # feat: [B, C, N_i, d_model]
            scale_sizes[scale_name] = feat.shape[2]
            all_features.append(feat)
        
        # 拼接所有尺度: [B, C, N_total, d_model]
        x = torch.cat(all_features, dim=2)
        B, C, N_total, D = x.shape
        
        # Reshape for Transformer: [B*C, N_total, d_model]
        x = x.view(B * C, N_total, D)
        
        # 通过Transformer层
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        
        # Reshape back: [B, C, N_total, d_model]
        x = x.view(B, C, N_total, D)
        
        if return_separated:
            # 分离回各尺度
            separated = {}
            start = 0
            for scale_name in multi_scale_features.keys():
                size = scale_sizes[scale_name]
                separated[scale_name] = x[:, :, start:start+size, :]
                start += size
            return x, separated
        
        return x


class CrossScaleTransformerLayer(nn.Module):
    """跨尺度Transformer单层"""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        norm: str = 'LayerNorm',
        res_attention: bool = True
    ):
        super().__init__()
        
        self.res_attention = res_attention
        
        # Multi-Head Self-Attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Normalization
        if 'batch' in norm.lower():
            self.norm1 = nn.Sequential(
                Transpose(1, 2),
                nn.BatchNorm1d(d_model),
                Transpose(1, 2)
            )
            self.norm2 = nn.Sequential(
                Transpose(1, 2),
                nn.BatchNorm1d(d_model),
                Transpose(1, 2)
            )
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B*C, N, d_model]
        Returns:
            [B*C, N, d_model]
        """
        # Self-Attention with residual
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class Transpose(nn.Module):
    """辅助模块：维度转置"""
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims = dims
        self.contiguous = contiguous
    
    def forward(self, x: Tensor) -> Tensor:
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        return x.transpose(*self.dims)


class AnyResHead(nn.Module):
    """
    AnyRes预测头
    将融合后的多尺度特征转换为预测输出
    
    Args:
        n_vars: 变量数（通道数）
        d_model: 嵌入维度
        total_patches: 总patch数
        target_window: 预测长度
        head_dropout: dropout率
        individual: 是否每个通道独立的预测头
        head_type: 预测头类型 ('flatten', 'attention', 'pooling')
    """
    
    def __init__(
        self,
        n_vars: int,
        d_model: int,
        total_patches: int,
        target_window: int,
        head_dropout: float = 0.0,
        individual: bool = False,
        head_type: str = 'flatten'
    ):
        super().__init__()
        
        self.n_vars = n_vars
        self.d_model = d_model
        self.total_patches = total_patches
        self.target_window = target_window
        self.individual = individual
        self.head_type = head_type
        
        nf = d_model * total_patches
        
        if head_type == 'flatten':
            if individual:
                self.linears = nn.ModuleList([
                    nn.Linear(nf, target_window) for _ in range(n_vars)
                ])
                self.dropouts = nn.ModuleList([
                    nn.Dropout(head_dropout) for _ in range(n_vars)
                ])
            else:
                self.flatten = nn.Flatten(start_dim=-2)
                self.linear = nn.Linear(nf, target_window)
                self.dropout = nn.Dropout(head_dropout)
        
        elif head_type == 'attention':
            # 使用注意力pooling
            self.query = nn.Parameter(torch.randn(1, 1, target_window, d_model))
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=4,
                dropout=head_dropout,
                batch_first=True
            )
            self.proj = nn.Linear(d_model, 1)
        
        elif head_type == 'pooling':
            # 简单的池化 + 线性
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.linear = nn.Linear(d_model, target_window)
            self.dropout = nn.Dropout(head_dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, C, N_total, d_model] 融合后的多尺度特征
        
        Returns:
            [B, C, target_window] 预测输出
        """
        B, C, N, D = x.shape
        
        if self.head_type == 'flatten':
            if self.individual:
                outputs = []
                for i in range(self.n_vars):
                    z = x[:, i, :, :].reshape(B, -1)  # [B, N*D]
                    z = self.linears[i](z)  # [B, target_window]
                    z = self.dropouts[i](z)
                    outputs.append(z)
                x = torch.stack(outputs, dim=1)  # [B, C, target_window]
            else:
                x = x.reshape(B, C, -1)  # [B, C, N*D]
                x = self.linear(x)  # [B, C, target_window]
                x = self.dropout(x)
        
        elif self.head_type == 'attention':
            # x: [B, C, N, D]
            x = x.view(B * C, N, D)  # [B*C, N, D]
            query = self.query.expand(B * C, -1, -1, -1).reshape(B * C, self.target_window, D)
            out, _ = self.cross_attn(query, x, x)  # [B*C, target_window, D]
            out = self.proj(out).squeeze(-1)  # [B*C, target_window]
            x = out.view(B, C, self.target_window)
        
        elif self.head_type == 'pooling':
            # x: [B, C, N, D] -> 池化 N 维度
            x = x.view(B * C, N, D).permute(0, 2, 1)  # [B*C, D, N]
            x = self.pool(x).squeeze(-1)  # [B*C, D]
            x = self.linear(x)  # [B*C, target_window]
            x = self.dropout(x)
            x = x.view(B, C, self.target_window)
        
        return x
