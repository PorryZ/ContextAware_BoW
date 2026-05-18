# ContextAware_BoW

一个面向**回环检测（Loop Closure）**的原型工程：
- 用 **SuperPoint** 提取局部特征；
- 用 **LightGlue 前几层 Self-Attention + RoPE** 给特征注入上下文；
- 用 **Faiss K-Means** 构建视觉词典；
- 用 **BoW + TF-IDF** 建立帧级检索库并做相似帧查询。

---

## 1. 给新人的 30 秒总览

这个仓库可以理解成三段流水线：

1. **特征编码（神经网络侧）**  
   图像 -> SuperPoint 关键点/描述子 -> ContextAwareEncoder（带位置编码与注意力） -> 上下文化描述子。
2. **词典构建（离线）**  
   大量描述子 -> Faiss KMeans -> 视觉单词（vocabulary）。
3. **检索与回环（在线或评估）**  
   每帧描述子量化成词频 -> 全序列 TF-IDF -> 余弦相似度检索 -> 回环候选。

---

## 2. 代码库结构（按新人上手顺序）

```text
ContextAware_BoW/
├─ src/
│  ├─ models/
│  │  └─ context_encoder.py        # 上下文特征编码器（核心网络）
│  ├─ retrieval/
│  │  ├─ faiss_cluster.py          # 离线词典聚类
│  │  └─ tfidf_index.py            # BoW数据库 + TF-IDF检索
│  └─ utils/
│     └─ data_loader.py            # 真实序列读取 + SuperPoint特征提取
├─ scripts/
│  ├─ test_mvp_pipeline.py         # 最小可跑通 demo（伪造数据）
│  ├─ run_real_data.py             # 真实数据全流程（提特征->聚类->检索）
│  ├─ build_offline_vocab.py       # 工业化离线词典构建
│  ├─ evaluate_loop_closure.py     # 使用已保存词典做在线回环评估
│  └─ batch_compare_images.py      # 批量图像两两相似度矩阵
└─ datasets/                        # 示例数据（仅少量样本）
```

---

## 3. 核心原理（建议重点吃透）

### 3.1 ContextAwareEncoder 在做什么

文件：`src/models/context_encoder.py`

核心思路：
1. 从 LightGlue 拿到预训练模块；
2. 只截取前 `num_layers` 层 transformer 的 self-attention；
3. 用关键点坐标计算 RoPE 位置编码；
4. 让描述子在同帧内做注意力交互，得到“上下文化”描述子；
5. 最后做 L2 归一化，方便后续余弦/IP 度量。

> 直觉上，这一步把“孤立局部点”变成了“带局部拓扑关系的点”。

### 3.2 视觉词典为何要离线构建

文件：`src/retrieval/faiss_cluster.py`

- 把大量 256 维描述子聚类成 K 个中心；
- 每个中心就是一个视觉单词（visual word）；
- 后续在线阶段只需最近邻量化，计算非常快。

这相当于把连续高维特征离散化成“词袋空间”。

### 3.3 TF-IDF 检索与回环判定

文件：`src/retrieval/tfidf_index.py`

- `add_frame`：把每帧特征量化并统计词频（TF）；
- `compute_tfidf_database`：全序列上拟合 IDF，得到稀疏 TF-IDF 向量；
- `query`：与所有历史帧做余弦相似度，且默认屏蔽前后 50 帧时间邻域，避免“伪回环”。

---

## 4. 每个脚本怎么用

> 先激活环境：`conda activate CAB`

### 4.1 最小验证（不依赖真实数据）

```bash
python scripts/test_mvp_pipeline.py
```

用途：验证“编码->聚类->检索”是否逻辑跑通。

### 4.2 真实数据端到端流程

```bash
python scripts/run_real_data.py
```

默认读取 `./datasets/xuyi2_3/left`，会：
- 逐帧提取并编码；
- 当场聚类建词典；
- 构建 TF-IDF 并查询最后一帧相似历史帧。

### 4.3 大规模离线建库（建议先做）

```bash
python scripts/build_offline_vocab.py
```

输出默认在 `outputs/global_underwater_vocab_k10000.npy`。

### 4.4 使用现成词典做回环评估

```bash
python scripts/evaluate_loop_closure.py
```

脚本当前默认读取 `outputs/vocab_5000.npy`。

### 4.5 图像集合两两相似度分析

```bash
python scripts/batch_compare_images.py --img_dir <你的图像目录> --vocab <词典路径>
```

会输出并保存相似度矩阵 `outputs/batch_similarity_matrix.npy`。

---

## 5. 新人必须了解的“关键点与坑”

1. **路径是硬编码的**：多个脚本默认数据路径不同，先统一你本地目录。  
2. **词典文件名不统一**：`run_real_data.py` 与 `evaluate_loop_closure.py` 的默认词典路径命名不同，建议标准化。  
3. **GPU逻辑需复核**：`faiss_cluster.py` 里 `use_gpu` 当前固定为 `False`；`tfidf_index.py` 则自动检测 `StandardGpuResources`。  
4. **内存策略要明确**：`build_offline_vocab.py` 有 `MAX_POOL_FEATURES` 熔断，避免一次性堆太多特征。  
5. **窗口抑制影响结果**：`query` 里默认屏蔽 ±50 帧，做实验时需明确是否开启。  

---

## 6. 给后续学习的建议（按阶段）

### 阶段A：先跑通（1~2天）
- 跑 `test_mvp_pipeline.py`，确认依赖无误；
- 用 100~300 张图跑 `run_real_data.py`，观察每帧特征数量与查询结果；
- 把关键中间结果保存下来（词频分布、TopK候选）。

### 阶段B：做可解释实验（3~7天）
- 对比 `num_layers=0/1/2` 对检索效果的影响；
- 对比 `vocab_size=500/2000/5000` 的精度与速度；
- 调整 `window` 大小，观察误检率变化。

### 阶段C：工程化（1~2周）
- 统一配置管理（YAML/argparse）替代脚本硬编码；
- 拆分“离线建库”和“在线检索”产物格式（元数据+版本）；
- 加入定量指标（Recall@K、PR曲线、回环召回/精确率）。

### 阶段D：算法升级（长期）
- 尝试增量 IDF 或滑动窗口 IDF（在线场景）；
- 引入几何一致性复核（RANSAC/PnP）过滤伪回环；
- 对接时序模型/位姿先验，构建多模态回环评分。

---

## 7. 快速排障清单

- `ModuleNotFoundError: lightglue`：先检查环境与安装源；
- 聚类慢：先减小 `VOCAB_SIZE` 和 `MAX_POOL_FEATURES`；
- 无回环候选：检查数据序列是否含真实回访段、以及 `window` 是否过大；
- 全部相似度偏低：先确认描述子是否做了 L2 归一化，词典是否与当前域匹配。

---

## 更新日志

### 20060423
- feat： 构建脚本、初步跑通5000聚类的回环检测；
- refactor: 支持大容量数据集；

### 20260409
- feat: initial commit
