import sys
import os
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_loader import RealSequenceLoader
from src.models.context_encoder import ContextAwareEncoder
from src.retrieval.faiss_cluster import build_visual_vocabulary
from src.retrieval.tfidf_index import ContextBoWDatabase

def main():
    print("=== Phase 2: Context-Aware BoW (Real Data Pipeline) ===")
    
    # ---------------------------------------------------------
    # 0. 参数配置与环境初始化
    # ---------------------------------------------------------
    # 替换为你的真实水下数据集路径
    DATA_DIR = "./datasets/xuyi2_3/left" 
    
    config = {
        'max_keypoints': 1024, # 提取的最大特征点数
        'nms_radius': 4,       # 非极大值抑制半径
        'vocab_size': 500,     # 真实场景需要更大的词典
        'kmeans_iter': 20
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"-> 运行设备: {device}")

    # ---------------------------------------------------------
    # 1. 实例化核心模块
    # ---------------------------------------------------------
    # 初始化数据加载器
    try:
        loader = RealSequenceLoader(data_dir=DATA_DIR, config=config, device=device)
        print(f"-> 成功加载序列，共包含 {len(loader)} 帧图像")
    except FileNotFoundError as e:
        print(f"⚠️ 路径错误: {e}")
        print("请创建一个测试文件夹并放入几张图片用于测试。")
        return

    # 初始化上下文编码器
    encoder = ContextAwareEncoder(features='superpoint', num_layers=2).to(device)
    encoder.eval()

    # ---------------------------------------------------------
    # 2. 遍历序列：特征提取与上下文编码
    # ---------------------------------------------------------
    all_context_descriptors = [] 
    frames_data = []             

    print("\n--- 步骤 1: 提取单帧上下文特征 ---")
    
    for idx in range(len(loader)):
        # a. 提取原始 SuperPoint 特征
        frame_data = loader.get_frame_data(idx)
        
        # b. 通过编码器注入拓扑上下文
        with torch.no_grad():
            context_desc = encoder(
                frame_data['keypoints'], 
                frame_data['descriptors'], 
                frame_data['image_size']
            )
            
            # 转为 NumPy: [N, 256] (此时 N 在不同帧中可能不同)
            context_desc_np = context_desc.squeeze(0).cpu().numpy()
            
            all_context_descriptors.append(context_desc_np)
            frames_data.append(context_desc_np)
            
            if idx % 10 == 0 or idx == len(loader) - 1:
                print(f"已处理帧 {idx}/{len(loader)-1}，当前帧有效特征点数: {context_desc_np.shape[0]}")

    # ---------------------------------------------------------
    # 3. 离线字典构建 (聚类)
    # ---------------------------------------------------------
    print("\n--- 步骤 2: 构建视觉字典 ---")
    # np.vstack 可以安全地拼接长度不同的 [N_i, 256] 矩阵
    pool_descriptors = np.vstack(all_context_descriptors)
    print(f"-> 参与聚类的特征总数: {pool_descriptors.shape[0]}")
    
    vocabulary = build_visual_vocabulary(
        pool_descriptors, 
        num_words=config['vocab_size'], 
        n_iter=config['kmeans_iter']
    )

    # ---------------------------------------------------------
    # 4. 在线检索与相似度验证
    # ---------------------------------------------------------
    print("\n--- 步骤 3: TF-IDF 数据库构建与查询 ---")
    db = ContextBoWDatabase(vocabulary)
    
    for frame_id, desc_np in enumerate(frames_data):
        db.add_frame(desc_np)
        
    db.compute_tfidf_database()
    
    # 假设查询序列的最后一帧
    query_id = len(frames_data) - 1
    
    # 使用你在类内写好的 query 方法，验证 NMS(相邻帧抑制) 逻辑
    top_matches, top_scores = db.query(query_frame_id=query_id, top_k=5)
    
    print(f"\n查询帧 ID: {query_id}")
    for match_id, score in zip(top_matches, top_scores):
        if score > 0:
            print(f" -> 与 历史帧 {match_id} 的余弦相似度: {score:.4f}")

if __name__ == '__main__':
    main()