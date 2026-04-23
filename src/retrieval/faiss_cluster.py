import faiss
import numpy as np
import time

def build_visual_vocabulary(descriptors_np, num_words=10000, n_iter=20):
    print(f"开始聚类 {descriptors_np.shape[0]} 个特征 -> {num_words} 个视觉单词...")
    d = descriptors_np.shape[1]
    
    # 动态检测当前环境是否成功加载了 GPU 模块
    use_gpu = False
    if use_gpu:
        print("💡 检测到 Faiss GPU 支持，启动 CUDA 加速聚类...")
        # 初始化 GPU 资源
        res = faiss.StandardGpuResources()
    else:
        print("⚠️ [警告] 未检测到 Faiss GPU 支持，自动降级为 CPU 聚类...")
    
    # 配置 K-Means 参数 (spherical=True 适配 L2 归一化的特征)
    kmeans = faiss.Kmeans(
        d=d, 
        k=num_words, 
        niter=n_iter, 
        verbose=True, 
        gpu=use_gpu,  # 根据检测结果动态决定
        spherical=True 
    )
    
    t0 = time.time()
    kmeans.train(descriptors_np)
    print(f"聚类完成，耗时: {time.time() - t0:.2f} 秒")
    
    # 返回聚类中心作为视觉字典
    return kmeans.centroids