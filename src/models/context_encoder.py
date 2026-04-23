import torch
import torch.nn as nn
from lightglue import LightGlue

class ContextAwareEncoder(nn.Module):
    def __init__(self, features='superpoint', num_layers=2):
        super().__init__()
        # 加载官方预训练模型
        base_model = LightGlue(features=features)
        
        # 提取 LightGlue 的旋转位置编码器 (RoPE)
        self.posenc = base_model.posenc
        
        # 我们只需要前 num_layers 层的自注意力机制
        self.num_layers = num_layers
        self.transformers = nn.ModuleList([
            base_model.transformers[i] for i in range(num_layers)
        ])
        
        # 冻结所有权重，仅作离线特征提取
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, kpts, desc, image_size):
        """
        kpts: [B, N, 2] 像素坐标
        desc: [B, N, 256] 原始 SuperPoint 描述子
        image_size: [B, 2] 图像的 (Width, Height)
        """
        # 1. 坐标归一化到 [-1, 1] 范围，适配 RoPE
        kpts_normalized = (kpts / image_size) * 2.0 - 1.0
        
        # 2. 利用归一化坐标计算旋转位置编码
        encoding = self.posenc(kpts_normalized)
        
        # 3. 依次通过前 L 层的 Self-Attention
        for i in range(self.num_layers):
            # 修正：同时传入描述子和位置编码，它会返回更新后的描述子
            desc = self.transformers[i].self_attn(desc, encoding)
            
        # 4. L2 归一化 (Faiss 聚类前必备)
        context_desc = torch.nn.functional.normalize(desc, p=2, dim=-1)
        
        return context_desc