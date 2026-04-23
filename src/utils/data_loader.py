import os
import torch
from pathlib import Path
from lightglue import SuperPoint
from lightglue.utils import load_image

class RealSequenceLoader:
    def __init__(self, data_dir: str, config: dict, device: torch.device):
        """
        初始化真实水下图像序列加载器
        """
        self.data_dir = Path(data_dir)
        # 支持常见图像格式，按文件名排序以保证时序连贯性
        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
        self.image_paths = sorted([
            p for p in self.data_dir.iterdir() if p.suffix.lower() in valid_extensions
        ])
        
        if not self.image_paths:
            raise FileNotFoundError(f"在 {data_dir} 中未找到有效图像！")
            
        self.device = device
        self.max_keypoints = config.get('max_keypoints', 1024)
        self.nms_radius = config.get('nms_radius', 4)
        
        # 实例化 LightGlue 自带的 SuperPoint 提取器
        self.extractor = SuperPoint(
            max_num_keypoints=self.max_keypoints,
            nms_radius=self.nms_radius
        ).eval().to(self.device)

    def __len__(self):
        return len(self.image_paths)

    def get_frame_data(self, idx: int):
        """
        读取单帧图像并提取原始 SuperPoint 特征
        返回与 ContextAwareEncoder 接口对齐的张量
        """
        img_path = self.image_paths[idx]
        
        # load_image 返回 [C, H, W] 格式的张量，且已归一化到 [0, 1]
        image = load_image(img_path).to(self.device)
        _, H, W = image.shape
        
        # 组装 image_size 为 [1, 2]，适配你现有的 ContextAwareEncoder
        image_size = torch.tensor([[float(W), float(H)]], device=self.device)
        
        with torch.no_grad():
            # 增加 Batch 维度: [1, C, H, W]
            pred = self.extractor({'image': image.unsqueeze(0)})
            
            # LightGlue 封装的提取器输出:
            # keypoints: [1, N, 2] (图像坐标系下的像素坐标)
            # descriptors: [1, N, 256] 
            kpts = pred['keypoints']
            desc = pred['descriptors']
            
        return {
            'frame_id': idx,
            'image_path': str(img_path),
            'image_size': image_size,
            'keypoints': kpts,
            'descriptors': desc
        }