# scripts/build_offline_vocab.py
import sys, os, torch, numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_loader import RealSequenceLoader
from src.models.context_encoder import ContextAwareEncoder
from src.retrieval.faiss_cluster import build_visual_vocabulary

def main():
    DATA_DIR = "./datasets/xuyi2_3/left" # 替换为你的真实路径
    config = {'max_keypoints': 1024, 'nms_radius': 4}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    loader = RealSequenceLoader(data_dir=DATA_DIR, config=config, device=device)
    encoder = ContextAwareEncoder(features='superpoint', num_layers=2).to(device)
    encoder.eval()

    all_context_descriptors = []
    sample_stride = 20 # 核心：每 20 帧抽样一次，防止内存爆炸
    
    print(f"开始抽取特征建立字典，总帧数 {len(loader)}，抽样步长 {sample_stride}...")
    for idx in range(0, len(loader), sample_stride):
        frame_data = loader.get_frame_data(idx)
        with torch.no_grad():
            context_desc = encoder(frame_data['keypoints'], frame_data['descriptors'], frame_data['image_size'])
            all_context_descriptors.append(context_desc.squeeze(0).cpu().numpy())
            
        if idx % 1000 == 0:
            print(f"已抽取帧 {idx}...")

    pool_descriptors = np.vstack(all_context_descriptors)
    print(f"参与聚类的特征总数: {pool_descriptors.shape[0]}")
    
    # 聚类并固化到硬盘
    vocabulary = build_visual_vocabulary(pool_descriptors, num_words=5000, n_iter=20)
    
    os.makedirs('outputs', exist_ok=True)
    np.save('outputs/vocab_5000.npy', vocabulary)
    print("✅ 字典已成功保存至 outputs/vocab_5000.npy")

if __name__ == '__main__':
    main()