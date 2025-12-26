"""
AnyRes-PatchTST: Multi-Scale Patch Time Series Transformer
Inspired by LLaVA-NeXT's AnyRes dynamic resolution technique
"""

__all__ = ['AnyResPatchTST']

from typing import Callable, Optional, List
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from layers.AnyRes_layers import (
    MultiScalePatchEmbedding,
    CrossScaleAttention,
    AnyResHead
)
from layers.PatchTST_backbone import TSTEncoder
from layers.PatchTST_layers import series_decomp
from layers.RevIN import RevIN


class Model(nn.Module):
    """
    AnyRes-PatchTST: 多尺度自适应Patch时间序列Transformer
    
    核心创新:
    1. 多尺度Patching: 同时使用多种patch_len捕获不同时间粒度的特征
    2. 全局Token: 提供整体序列上下文（类似AnyRes的thumbnail）
    3. 跨尺度注意力融合: 通过Transformer融合多尺度特征
    
    Args:
        configs: 配置对象，包含以下字段：
            - enc_in: 输入通道数
            - seq_len: 输入序列长度
            - pred_len: 预测长度
            - d_model: 模型维度
            - n_heads: 注意力头数
            - e_layers: encoder层数
            - d_ff: FFN维度
            - dropout: dropout率
            - patch_scales: 多尺度patch长度列表，如 "8,16,32"
            - use_global_token: 是否使用全局token
            - cross_scale_layers: 跨尺度融合层数
            - individual: 是否每个通道独立预测
            - revin: 是否使用RevIN归一化
    """
    
    def __init__(
        self,
        configs,
        max_seq_len: Optional[int] = 1024,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        norm: str = 'BatchNorm',
        attn_dropout: float = 0.,
        act: str = "gelu",
        key_padding_mask: bool = 'auto',
        padding_var: Optional[int] = None,
        attn_mask: Optional[Tensor] = None,
        res_attention: bool = True,
        pre_norm: bool = False,
        store_attn: bool = False,
        pe: str = 'zeros',
        learn_pe: bool = True,
        pretrain_head: bool = False,
        head_type: str = 'flatten',
        verbose: bool = False,
        **kwargs
    ):
        super().__init__()
        
        # 基础参数
        self.c_in = configs.enc_in
        self.context_window = configs.seq_len
        self.target_window = configs.pred_len
        
        self.n_layers = configs.e_layers
        self.n_heads = configs.n_heads
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.dropout = configs.dropout
        self.fc_dropout = configs.fc_dropout
        self.head_dropout = configs.head_dropout
        
        self.individual = configs.individual
        
        # AnyRes特有参数
        # 解析patch_scales: "8,16,32" -> [8, 16, 32]
        if hasattr(configs, 'patch_scales') and configs.patch_scales:
            if isinstance(configs.patch_scales, str):
                self.patch_scales = [int(x) for x in configs.patch_scales.split(',')]
            else:
                self.patch_scales = configs.patch_scales
        else:
            # 默认多尺度配置
            self.patch_scales = [8, 16, 32]
        
        self.use_global_token = getattr(configs, 'use_global_token', True)
        self.cross_scale_layers = getattr(configs, 'cross_scale_layers', 2)
        self.cross_scale_fusion = getattr(configs, 'cross_scale_fusion', 'attention')
        
        # RevIN归一化
        self.revin = configs.revin
        self.affine = configs.affine
        self.subtract_last = configs.subtract_last
        if self.revin:
            self.revin_layer = RevIN(self.c_in, affine=self.affine, subtract_last=self.subtract_last)
        
        # 序列分解（可选）
        self.decomposition = configs.decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(configs.kernel_size)
        
        # =============== AnyRes核心组件 ===============
        
        # 1. 多尺度Patch嵌入
        self.multi_scale_embedding = MultiScalePatchEmbedding(
            c_in=self.c_in,
            context_window=self.context_window,
            d_model=self.d_model,
            patch_scales=self.patch_scales,
            use_global_token=self.use_global_token,
            pe=pe,
            learn_pe=learn_pe,
            dropout=self.dropout
        )
        
        # 2. 每个尺度的独立Encoder（共享或独立可配置）
        self.scale_encoders = nn.ModuleDict()
        self.share_encoder = getattr(configs, 'share_encoder', True)
        
        if self.share_encoder:
            # 共享编码器
            max_patch_num = max(self.multi_scale_embedding.patch_nums.values())
            self.shared_encoder = TSTEncoder(
                q_len=max_patch_num,
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_k=d_k,
                d_v=d_v,
                d_ff=self.d_ff,
                norm=norm,
                attn_dropout=attn_dropout,
                dropout=self.dropout,
                activation=act,
                res_attention=res_attention,
                n_layers=self.n_layers,
                pre_norm=pre_norm,
                store_attn=store_attn
            )
        else:
            # 每个尺度独立编码器
            for patch_len in self.patch_scales:
                patch_num = self.multi_scale_embedding.patch_nums[patch_len]
                self.scale_encoders[str(patch_len)] = TSTEncoder(
                    q_len=patch_num,
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    d_k=d_k,
                    d_v=d_v,
                    d_ff=self.d_ff,
                    norm=norm,
                    attn_dropout=attn_dropout,
                    dropout=self.dropout,
                    activation=act,
                    res_attention=res_attention,
                    n_layers=self.n_layers,
                    pre_norm=pre_norm,
                    store_attn=store_attn
                )
        
        # 3. 跨尺度注意力融合
        if self.cross_scale_fusion == 'attention':
            self.cross_scale_attn = CrossScaleAttention(
                d_model=self.d_model,
                n_heads=self.n_heads,
                n_layers=self.cross_scale_layers,
                d_ff=self.d_ff,
                dropout=self.dropout,
                norm=norm,
                res_attention=res_attention
            )
        elif self.cross_scale_fusion == 'concat':
            # 简单拼接后通过线性层
            total_patches = self.multi_scale_embedding.get_total_patches()
            self.fusion_proj = nn.Linear(total_patches * self.d_model, total_patches * self.d_model)
        elif self.cross_scale_fusion == 'weighted_sum':
            # 学习权重加权求和
            n_scales = len(self.patch_scales) + (1 if self.use_global_token else 0)
            self.scale_weights = nn.Parameter(torch.ones(n_scales) / n_scales)
        
        # 4. 预测头
        self.total_patches = self.multi_scale_embedding.get_total_patches()
        self.head = AnyResHead(
            n_vars=self.c_in,
            d_model=self.d_model,
            total_patches=self.total_patches,
            target_window=self.target_window,
            head_dropout=self.head_dropout,
            individual=self.individual,
            head_type=head_type
        )
        
        if verbose:
            print(f"AnyRes-PatchTST initialized:")
            print(f"  - Patch scales: {self.patch_scales}")
            print(f"  - Patch nums: {self.multi_scale_embedding.patch_nums}")
            print(f"  - Total patches: {self.total_patches}")
            print(f"  - Use global token: {self.use_global_token}")
            print(f"  - Cross-scale fusion: {self.cross_scale_fusion}")
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, L, C] 输入时间序列 (Batch, SeqLen, Channels)
        
        Returns:
            [B, pred_len, C] 预测输出
        """
        # x: [B, L, C] -> [B, C, L] (Channel-first)
        x = x.permute(0, 2, 1)
        
        # RevIN归一化
        if self.revin:
            x = x.permute(0, 2, 1)  # [B, L, C]
            x = self.revin_layer(x, 'norm')
            x = x.permute(0, 2, 1)  # [B, C, L]
        
        # 序列分解（可选）
        if self.decomposition:
            x_perm = x.permute(0, 2, 1)  # [B, L, C]
            res, trend = self.decomp_module(x_perm)
            # 这里简化处理，只用残差分量
            x = res.permute(0, 2, 1)  # [B, C, L]
        
        # =============== AnyRes核心流程 ===============
        
        # 1. 多尺度Patch嵌入
        # multi_scale_feats: Dict[str, Tensor], 每个 [B, C, N_i, d_model]
        multi_scale_feats = self.multi_scale_embedding(x)
        
        # 2. 每个尺度通过Encoder
        B, C = x.shape[:2]
        encoded_feats = {}
        
        for scale_name, feat in multi_scale_feats.items():
            # feat: [B, C, N_i, d_model]
            N_i = feat.shape[2]
            
            if scale_name == 'global':
                # 全局token不经过encoder，直接使用
                encoded_feats[scale_name] = feat
            else:
                # Reshape for encoder: [B*C, N_i, d_model]
                feat_reshaped = feat.view(B * C, N_i, self.d_model)
                
                if self.share_encoder:
                    encoded = self.shared_encoder(feat_reshaped)
                else:
                    encoded = self.scale_encoders[scale_name](feat_reshaped)
                
                # Reshape back: [B, C, N_i, d_model]
                encoded = encoded.view(B, C, N_i, self.d_model)
                encoded_feats[scale_name] = encoded
        
        # 3. 跨尺度融合
        if self.cross_scale_fusion == 'attention':
            fused = self.cross_scale_attn(encoded_feats)  # [B, C, N_total, d_model]
        
        elif self.cross_scale_fusion == 'concat':
            # 简单拼接
            all_feats = [encoded_feats[str(s)] for s in self.patch_scales]
            if self.use_global_token:
                all_feats.append(encoded_feats['global'])
            fused = torch.cat(all_feats, dim=2)  # [B, C, N_total, d_model]
        
        elif self.cross_scale_fusion == 'weighted_sum':
            # 加权求和（需要对齐patch数，这里用池化）
            weights = F.softmax(self.scale_weights, dim=0)
            all_feats = [encoded_feats[str(s)] for s in self.patch_scales]
            if self.use_global_token:
                all_feats.append(encoded_feats['global'])
            
            # 池化到相同维度后加权
            pooled = []
            for i, feat in enumerate(all_feats):
                # 对patch维度池化
                pooled_feat = feat.mean(dim=2, keepdim=True)  # [B, C, 1, d_model]
                pooled.append(weights[i] * pooled_feat)
            
            fused = sum(pooled)  # [B, C, 1, d_model]
            # 扩展回总patch数
            fused = fused.expand(-1, -1, self.total_patches, -1)
        
        # 4. 预测头
        out = self.head(fused)  # [B, C, pred_len]
        
        # RevIN反归一化
        if self.revin:
            out = out.permute(0, 2, 1)  # [B, pred_len, C]
            out = self.revin_layer(out, 'denorm')
            out = out.permute(0, 2, 1)  # [B, C, pred_len]
        
        # [B, C, pred_len] -> [B, pred_len, C]
        out = out.permute(0, 2, 1)
        
        return out
