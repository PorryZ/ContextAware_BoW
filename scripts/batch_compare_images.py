import sys
import os
import torch
import numpy as np
import faiss
import argparse
from pathlib import Path
from lightglue import SuperPoint
from lightglue.utils import load_image

# 将工程根目录加入系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.context_encoder import ContextAwareEncoder

def extract_and_encode(image_path, extractor, encoder, device):
    """
    读取单张图像并提取 Context-Aware 特征
    """
    try:
        image = load_image(image_path).to(device)
        _, H, W = image.shape
        image_size = torch.tensor([[float(W), float(H)]], device=device)
        
        with torch.no_grad():
            pred = extractor({'image': image.unsqueeze(0)})
            kpts = pred['keypoints']
            desc = pred['descriptors']
            
            context_desc = encoder(kpts, desc, image_size)
            
        return context_desc.squeeze(0).cpu().numpy()
    except Exception as e:
        print(f"提取特征失败: {image_path}, 错误: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="计算批量图像的两两上下文BoW相似度矩阵")
    parser.add_argument('--img_dir', type=str, required=True, help="包含待测试图像的文件夹路径")
    parser.add_argument('--vocab', type=str, default='vocab/vocab_5000.npy', help="离线字典库路径")
    parser.add_argument('--max_imgs', type=int, default=-1, help="最大处理图像数量 (测试用)")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"-> 运行设备: {device}")

    # ---------------------------------------------------------
    # 1. 验证字典并初始化量化器
    # ---------------------------------------------------------
    if not os.path.exists(args.vocab):
        raise FileNotFoundError(f"找不到字典文件 {args.vocab}，请先运行离线建库脚本！")
        
    print(f"-> 正在加载离线字典: {args.vocab}")
    vocab_matrix = np.load(args.vocab)
    num_words, dim = vocab_matrix.shape
    
    # 建立 Faiss IP (Inner Product) 索引
    quantizer = faiss.IndexFlatIP(dim) 
    quantizer.add(vocab_matrix)

    # ---------------------------------------------------------
    # 2. 初始化神经网络
    # ---------------------------------------------------------
    print("-> 正在初始化 SuperPoint 与 ContextAwareEncoder...")
    extractor = SuperPoint(max_num_keypoints=1024, nms_radius=4).eval().to(device)
    encoder = ContextAwareEncoder(features='superpoint', num_layers=2).eval().to(device)

    # ---------------------------------------------------------
    # 3. 读取图像列表并提取特征直方图
    # ---------------------------------------------------------
    img_dir = Path(args.img_dir)
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
    # 排序保证结果矩阵的行列对应关系一致 [cite: 6]
    image_paths = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in valid_extensions])
    
    if args.max_imgs > 0:
        image_paths = image_paths[:args.max_imgs]
        
    num_images = len(image_paths)
    print(f"\n-> 找到 {num_images} 张有效图像，开始特征提取与量化...")
    
    # 存储所有图像的归一化直方图
    histograms = []
    valid_image_names = []

    for idx, img_path in enumerate(image_paths):
        desc = extract_and_encode(img_path, extractor, encoder, device)
        if desc is None or desc.shape[0] == 0:
            print(f"⚠️ 警告: 图像 {img_path.name} 提取失败或未检测到特征点。")
            continue
            
        # 寻找最近邻单词 ID
        _, words = quantizer.search(desc, 1)
        
        # 统计词频并 L2 归一化 [cite: 32]
        hist = np.bincount(words.flatten(), minlength=num_words).astype(np.float32)
        hist /= (np.linalg.norm(hist) + 1e-8)
        
        histograms.append(hist)
        valid_image_names.append(img_path.name)
        
        print(f"  [{idx+1}/{num_images}] 处理完成: {img_path.name} ({desc.shape[0]} 个特征点)")

    # ---------------------------------------------------------
    # 4. 计算 N x N 相似度矩阵
    # ---------------------------------------------------------
    print("\n-> 正在计算两两相似度矩阵...")
    # 将列表转换为矩阵 [N, num_words]
    hist_matrix = np.vstack(histograms)
    
    # 矩阵乘法直接得到所有图两两之间的余弦相似度 [N, N]
    similarity_matrix = np.dot(hist_matrix, hist_matrix.T)

    # ---------------------------------------------------------
    # 5. 打印与保存结果
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("相似度矩阵 (Similarity Matrix):")
    
    # 打印简化的矩阵视图 (最多显示前 10x10)
    display_size = min(len(valid_image_names), 10)
    
    # 打印表头
    header = "      " + "".join([f"[{i:^5}]" for i in range(display_size)])
    print(header)
    
    for i in range(display_size):
        row_str = f"[{i:^3}] "
        for j in range(display_size):
            row_str += f"{similarity_matrix[i, j]:.4f} "
        print(row_str + f" <- {valid_image_names[i]}")
        
    if num_images > display_size:
         print(f"... (省略后续 {num_images - display_size} 行/列)")

    print("="*50)
    
    # 将完整的矩阵保存为 npy 文件，供后续绘图或分析使用
    out_file = 'outputs/batch_similarity_matrix.npy'
    os.makedirs('outputs', exist_ok=True)
    np.save(out_file, similarity_matrix)
    print(f"✅ 完整的相似度矩阵已保存至: {out_file}")

if __name__ == '__main__':
    main()