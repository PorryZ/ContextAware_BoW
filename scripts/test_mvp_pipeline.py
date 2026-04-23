import sys
import os
import torch
import numpy as np

# 将工程根目录加入系统路径，以便能够导入 src 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.context_encoder import ContextAwareEncoder
from src.retrieval.faiss_cluster import build_visual_vocabulary
from src.retrieval.tfidf_index import ContextBoWDatabase

def main():
    print("=== Phase 1: Context-Aware BoW MVP 测试 ===")
    
    # ---------------------------------------------------------
    # 1. 初始化模型与伪造数据
    # ---------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 假设图片分辨率为 640x480
    image_size = torch.tensor([[640.0, 480.0]]).to(device)
    
    # 实例化我们的上下文编码器 (使用 2 层自注意力)
    encoder = ContextAwareEncoder(features='superpoint', num_layers=2).to(device)
    encoder.eval()

    # 模拟收集 5 帧图像，每帧提取 200 个 SuperPoint 特征点
    num_frames = 5
    num_keypoints = 200
    
    all_context_descriptors = [] # 用于离线训练字典
    frames_data = []             # 保存每帧数据用于在线测试

    print("\n--- 步骤 1: 提取单帧上下文特征 ---")
    with torch.no_grad():
        for frame_id in range(num_frames):
            # 伪造像素坐标 [1, 200, 2] 和原始描述子 [1, 200, 256]
            mock_kpts = torch.rand(1, num_keypoints, 2).to(device) * torch.tensor([640.0, 480.0]).to(device)
            mock_desc = torch.randn(1, num_keypoints, 256).to(device)
            
            # 通过编码器获取带有拓扑信息的特征
            context_desc = encoder(mock_kpts, mock_desc, image_size)
            
            # 转换为 NumPy，形状为 [200, 256]
            context_desc_np = context_desc.squeeze(0).cpu().numpy()
            
            all_context_descriptors.append(context_desc_np)
            frames_data.append(context_desc_np)
            print(f"帧 {frame_id} 处理完毕，特征维度: {context_desc_np.shape}")

    # ---------------------------------------------------------
    # 2. 离线字典构建 (聚类)
    # ---------------------------------------------------------
    print("\n--- 步骤 2: 构建视觉字典 ---")
    # 将所有帧的特征拼接起来，总计 5 * 200 = 1000 个特征
    pool_descriptors = np.vstack(all_context_descriptors)
    
    # 聚类成 50 个视觉单词 (因为是测试，数量设小一点)
    vocab_size = 50 
    vocabulary = build_visual_vocabulary(pool_descriptors, num_words=vocab_size, n_iter=10)
    print(f"生成的视觉字典形状: {vocabulary.shape}")

    # ---------------------------------------------------------
    # 3. 在线检索与相似度验证
    # ---------------------------------------------------------
    print("\n--- 步骤 3: TF-IDF 数据库构建与查询 ---")
    db = ContextBoWDatabase(vocabulary)
    
    # 将前面提取的 5 帧依次加入数据库
    for frame_id, desc_np in enumerate(frames_data):
        db_id = db.add_frame(desc_np)
        print(f"帧 {frame_id} 加入数据库，ID 为 {db_id}")
        
    # 计算 TF-IDF 矩阵
    db.compute_tfidf_database()
    
    # 假设当前来到第 4 帧，我们在数据库中检索它的相似帧 (这里为了测试取消了相邻帧抑制)
    query_id = 4
    
    # 为了测试效果，我们临时把 database_vectors 直接拿来算，不经过类内的 query 函数(因为类内有屏蔽相邻帧的逻辑)
    query_vector = db.database_vectors[query_id]
    similarity_scores = db.database_vectors.dot(query_vector.T).toarray().flatten()
    
    print(f"\n查询帧 ID: {query_id}")
    for idx, score in enumerate(similarity_scores):
        print(f" -> 与 帧 {idx} 的余弦相似度: {score:.4f}")

if __name__ == '__main__':
    main()