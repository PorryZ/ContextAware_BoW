# scripts/evaluate_loop_closure.py
import sys, os, torch, numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_loader import RealSequenceLoader
from src.models.context_encoder import ContextAwareEncoder
from src.retrieval.tfidf_index import ContextBoWDatabase

def main():
    DATA_DIR = "./datasets/xuyi2_3/left" 
    config = {'max_keypoints': 1024, 'nms_radius': 4}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 加载硬盘上离线构建好的字典
    print("正在加载离线词典...")
    vocabulary = np.load('outputs/vocab_5000.npy')
    db = ContextBoWDatabase(vocabulary)
    
    loader = RealSequenceLoader(data_dir=DATA_DIR, config=config, device=device)
    encoder = ContextAwareEncoder(features='superpoint', num_layers=2).to(device)
    encoder.eval()

    print(f"开始在线检索流程，共 {len(loader)} 帧...")
    # 2. 遍历所有帧：量化即销毁 (不会引发内存泄漏)
    for idx in range(len(loader)):
        frame_data = loader.get_frame_data(idx)
        with torch.no_grad():
            context_desc = encoder(frame_data['keypoints'], frame_data['descriptors'], frame_data['image_size'])
            context_desc_np = context_desc.squeeze(0).cpu().numpy()
            
            # 核心：量化特征加入数据库，随后 context_desc_np 会被自动释放
            db.add_frame(context_desc_np)
            
        if idx % 500 == 0:
            print(f"已处理并量化 帧 {idx}...")

    # 3. 计算 TF-IDF 矩阵
    print("\n正在计算全局 TF-IDF 矩阵...")
    db.compute_tfidf_database()
    
    # 4. 测试回环检索 (例如测试最后一帧)
    query_id = len(loader) - 1
    top_matches, top_scores = db.query(query_frame_id=query_id, top_k=5)
    
    print(f"\n帧 {query_id} 的 Top-5 回环候选：")
    for match_id, score in zip(top_matches, top_scores):
        if score > 0:
            print(f" -> 与 历史帧 {match_id} 相似度: {score:.4f}")

if __name__ == '__main__':
    main()