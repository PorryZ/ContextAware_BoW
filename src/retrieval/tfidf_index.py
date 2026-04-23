import faiss
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer

class ContextBoWDatabase:
    def __init__(self, vocabulary):
        """
        vocabulary: [K, 256] 的视觉单词矩阵
        """
        self.num_words = vocabulary.shape[0]
        self.d = vocabulary.shape[1]
        
        cpu_index = faiss.IndexFlatIP(self.d) # IP = Inner Product (等价于余弦相似度)
        
        # 动态检测 GPU 支持
        if hasattr(faiss, 'StandardGpuResources'):
            res = faiss.StandardGpuResources()
            self.quantizer = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        else:
            self.quantizer = cpu_index
            
        self.quantizer.add(vocabulary)
        
        self.tfidf_transformer = TfidfTransformer(norm='l2', use_idf=True)
        self.database_tf_counts = []
        self.database_vectors = None

    def add_frame(self, frame_desc_np):
        """
        frame_desc_np: [N, 256] 当前帧经过上下文编码的特征
        返回当前帧在数据库中的 ID
        """
        # 1. 特征量化: 寻找每个特征最匹配的视觉单词 ID
        _, word_ids = self.quantizer.search(frame_desc_np, 1)
        word_ids = word_ids.flatten()
        
        # 2. 统计 Term Frequency (词频)
        tf_counts = np.bincount(word_ids, minlength=self.num_words)
        self.database_tf_counts.append(tf_counts)
        
        return len(self.database_tf_counts) - 1

    def compute_tfidf_database(self):
        """
        在所有帧添加完毕后，计算整个序列的 TF-IDF 矩阵
        """
        # 转换为稀疏矩阵 [NumFrames, NumWords]
        tf_matrix = csr_matrix(self.database_tf_counts)
        # 拟合并计算 TF-IDF
        self.database_vectors = self.tfidf_transformer.fit_transform(tf_matrix)

    def query(self, query_frame_id, top_k=5):
        """
        计算 query 帧与数据库中所有历史帧的余弦相似度
        """
        if self.database_vectors is None:
            self.compute_tfidf_database()
            
        query_vector = self.database_vectors[query_frame_id]
        
        # 稀疏矩阵点乘直接得出余弦相似度 (因为已经 l2 归一化)
        similarity_scores = self.database_vectors.dot(query_vector.T).toarray().flatten()
        
        # 排除自身以及相邻的若干帧 (防止时间相邻帧被误认为回环)
        window = 50 
        min_idx = max(0, query_frame_id - window)
        max_idx = min(len(similarity_scores), query_frame_id + window)
        similarity_scores[min_idx:max_idx] = 0.0 
        
        # 获取 Top-K 候选帧
        best_matches = np.argsort(similarity_scores)[::-1][:top_k]
        best_scores = similarity_scores[best_matches]
        
        return best_matches, best_scores