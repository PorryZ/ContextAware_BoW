import sys
import os
import torch
import numpy as np
import random
import time
from pathlib import Path

# 将工程根目录加入系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.context_encoder import ContextAwareEncoder
from lightglue import SuperPoint
from lightglue.utils import load_image
from src.retrieval.faiss_cluster import build_visual_vocabulary

def get_all_image_paths(root_dir):
    """递归获取目录下所有支持的图像路径"""
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
    root_path = Path(root_dir)
    image_paths = []
    for ext in valid_extensions:
        image_paths.extend(list(root_path.rglob(f"*{ext}")))
        image_paths.extend(list(root_path.rglob(f"*{ext.upper()}")))
    return image_paths

def main():
    print("=== Phase 1: 工业级全局水下视觉字典构建 ===")
    
    # ---------------------------------------------------------
    # 1. 核心参数配置
    # ---------------------------------------------------------
    TRAIN_DATA_DIR = "./data/global_vocab_train" # 指向你存放所有数据集的根目录
    VOCAB_SIZE = 10000        # 目标视觉单词数量 (建议 10000 - 20000)
    MAX_POOL_FEATURES = 2_000_000 # 特征池容量上限 (200万个256维特征仅占约 2GB RAM，绝对安全)
    MAX_KPTS_PER_IMAGE = 1024 # 每张图最多提取特征数
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"-> 运行设备: {device}")

    # ---------------------------------------------------------
    # 2. 搜集并打乱全局图像 (保证特征多样性)
    # ---------------------------------------------------------
    all_images = get_all_image_paths(TRAIN_DATA_DIR)
    if not all_images:
        raise FileNotFoundError(f"在 {TRAIN_DATA_DIR} 中未找到任何图像！请检查路径。")
    
    # 核心：打乱图像顺序！这样即使中途达到 MAX_POOL_FEATURES 停止，
    # 提取到的特征也均匀散布于各个不同的子数据集中。
    random.seed(42)
    random.shuffle(all_images)
    print(f"-> 共扫描到 {len(all_images)} 张训练图像，已随机打乱。")

    # ---------------------------------------------------------
    # 3. 初始化模型
    # ---------------------------------------------------------
    print("-> 正在初始化 SuperPoint 与 ContextAwareEncoder...")
    extractor = SuperPoint(max_num_keypoints=MAX_KPTS_PER_IMAGE, nms_radius=4).eval().to(device)
    encoder = ContextAwareEncoder(features='superpoint', num_layers=2).eval().to(device)

    # ---------------------------------------------------------
    # 4. 抽取上下文特征 (带有内存保护熔断机制)
    # ---------------------------------------------------------
    feature_pool = []
    current_feature_count = 0
    t0 = time.time()
    
    print(f"\n-> 开始特征提取 (上限设定为 {MAX_POOL_FEATURES} 个特征点)...")
    for idx, img_path in enumerate(all_images):
        try:
            image = load_image(img_path).to(device)
            _, H, W = image.shape
            image_size = torch.tensor([[float(W), float(H)]], device=device)
            
            with torch.no_grad():
                # 提取原始特征
                pred = extractor({'image': image.unsqueeze(0)})
                kpts, desc = pred['keypoints'], pred['descriptors']
                
                # 注入上下文拓扑 (如果没提取到特征则跳过)
                if kpts.shape[1] == 0:
                    continue
                    
                context_desc = encoder(kpts, desc, image_size)
                context_desc_np = context_desc.squeeze(0).cpu().numpy()
                
                feature_pool.append(context_desc_np)
                current_feature_count += context_desc_np.shape[0]
                
        except Exception as e:
            print(f"⚠️ 读取 {img_path.name} 失败，已跳过: {e}")
            
        # 进度播报
        if (idx + 1) % 100 == 0:
            print(f"  已处理 {idx + 1} 张图 | 累计收集特征点: {current_feature_count} 个")
            
        # 内存熔断机制：达到安全上限，立刻停止提取！
        if current_feature_count >= MAX_POOL_FEATURES:
            print(f"✅ 已达到特征池安全上限 ({MAX_POOL_FEATURES})，停止提取以保护内存。")
            break

    print(f"-> 特征提取完毕！耗时 {time.time() - t0:.2f} 秒。")

    # ---------------------------------------------------------
    # 5. 特征拼接与高维聚类
    # ---------------------------------------------------------
    print("\n-> 正在拼接高维特征矩阵...")
    # np.vstack 可以极其高效地将分散的数组合并成一个巨大的 [N, 256] 矩阵
    pool_descriptors = np.vstack(feature_pool)
    
    # 如果略微超过上限，可以进行截断以确保一致性
    if pool_descriptors.shape[0] > MAX_POOL_FEATURES:
        pool_descriptors = pool_descriptors[:MAX_POOL_FEATURES, :]
        
    print(f"-> 最终参与聚类的矩阵维度: {pool_descriptors.shape} (约占用 {pool_descriptors.nbytes / (1024**2):.1f} MB 内存)")

    # 训练视觉字典
    print(f"\n-> 启动 Faiss K-Means 聚类 (生成 {VOCAB_SIZE} 个视觉单词)...")
    vocabulary = build_visual_vocabulary(pool_descriptors, num_words=VOCAB_SIZE, n_iter=20)
    
    # ---------------------------------------------------------
    # 6. 固化字典模型
    # ---------------------------------------------------------
    os.makedirs('outputs', exist_ok=True)
    out_path = f"outputs/global_underwater_vocab_k{VOCAB_SIZE}.npy"
    np.save(out_path, vocabulary)
    print(f"\n🎉 大功告成！高泛化能力水下视觉字典已成功保存至: {out_path}")

if __name__ == '__main__':
    main()