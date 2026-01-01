## 基于 MovieLens-100k 的两阶段推荐系统（Recommender System）

### 一、项目简介

本项目模拟电商 / 视频平台的后台推荐系统
使用 **MovieLens 100k** 数据集（`u.user`, `u.item`, `u.data`）
从零实现工业界常见的 **“召回 + 排序”两阶段推荐架构**。

项目特点：
- **完全手写算法逻辑**，不依赖任何黑盒推荐 API  
- 使用 **Python + Pandas + NumPy + Scikit-learn**  
- 代码结构清晰、中文注释丰富，适合作为 **“课程设计 / 实验项目”**  
- 支持 **冷启动策略** 和 **离线评估 (Precision@5)**  

项目目录结构：

```text
Recommender System/
│
├── ml-100k/               # MovieLens 100k 数据集目录
│   ├── u.user            # 用户信息
│   ├── u.item            # 电影信息
│   └── u.data            # 用户-电影评分行为
│
├── data_loader.py          # 数据读取与清洗
├── feature_engineering.py  # 用户 / 电影特征构建
├── recall_cf.py            # Item-based 协同过滤召回
├── ranking_model.py        # LR 精排模型训练
├── recommender.py          # 推荐系统主类（含冷启动）
├── evaluation.py           # Precision@5 离线评估
├── main.py                 # 命令行 Demo 入口
└── README.md               # 项目说明（原理 + 使用）
```

> 注意：代码默认从 `ml-100k/` 目录读取数据文件。  
> 如果你的数据在其他目录，可以修改 `main.py` 中 `Recommender(data_dir="...")` 的参数。

---

### 二、系统架构说明

整体采用工业界常见的 **“召回 + 排序” 两阶段推荐架构**：

#### 1. 召回层（Recall）

- 算法：**Item-based Collaborative Filtering（基于物品的协同过滤）**
- 思路：
  - 用户喜欢的电影 X，与 X 行为相似的电影 Y，很可能也被该用户喜欢
  - 通过计算 **电影-电影之间的余弦相似度**，找到“相似电影”
- 作用：
  - 从上千部电影中快速筛选出约 **50 部候选电影**
  - 样本规模较小，便于后续排序模型做更复杂的计算

#### 2. 排序层（Ranking）

- 算法：**Logistic Regression（逻辑回归）**
- 样本构建：
  - 正样本：用户评分 ≥ 4 的电影
  - 负样本：用户未看过的电影中随机采样
- 特征：
  - 将 **用户画像特征** 和 **电影画像特征** 直接拼接：
    \[ feature = [user_features, item_features] \]
- 作用：
  - 为每个“用户-电影”对输出一个 0~1 概率，表示“喜欢 / 点击概率”
  - 按概率从高到低排序，得到最终推荐列表

#### 3. 冷启动策略

- 问题：**新用户没有历史行为**，协同过滤和排序模型无法正常工作
- 解决方案：
  - 在训练集上统计：
    - 每部电影的平均评分 `avg_rating`
    - 评分次数 `rating_count`
  - 只保留 `rating_count >= 10` 的电影
  - 按 `avg_rating` 从高到低排序，取 **Top-5 电影** 作为“热门兜底”
- 使用时机：
  - 当 `user_id` 不在训练集中（即用户画像表中不存在该用户）时
  - 或者召回 / 排序阶段无法给出有效候选时

#### 4. 离线评估（Precision@5）

- 将行为数据按时间排序：
  - **前 80%** 作为训练集
  - **后 20%** 作为测试集
- 对于每个用户：
  - 在测试集中，真实喜欢的电影（评分 ≥ 4）集合为 L_u
  - 模型给该用户的推荐 Top-5 为 R_u
  - Precision@5(u) = |R_u ∩ L_u| / 5
- 对所有用户取平均，得到整体 **Precision@5** 指标

---

### 三、各模块功能说明

#### 1. `data_loader.py` —— 数据读取与清洗

主要职责：

- 从 `ml-100k/` 目录读取 MovieLens 100k 的三个核心文件：
  - `u.user`：用户年龄、性别、职业等
  - `u.item`：电影标题、类型 one-hot 等
  - `u.data`：用户-电影评分及时间戳
- 输出三张 **原始 DataFrame**：
  - `user_raw_df`
  - `item_raw_df`
  - `interaction_raw_df`

文件顶部和函数内均有详细注释，便于向老师说明“每一列代表什么含义”。

#### 2. `feature_engineering.py` —— 特征工程

主要职责：

1. 构建 **用户画像表 `user_profile`**：
   - `user_id`
   - `age_normalized`（标准化后的年龄）
   - `gender_M`, `gender_F`（性别 0/1 编码）
   - `occ_*`（职业 one-hot 编码）

2. 构建 **电影画像表 `item_profile`**：
   - `movie_id`
   - 各类电影类型列（Action / Comedy / Drama / ...），直接使用 u.item 中的 0/1 列

3. 构建 **行为表 `interaction`**：
   - `user_id`, `movie_id`, `rating`, `timestamp`
   - 新增 `datetime` 列（时间戳转可读时间）
   - 按时间排序，为后续时间切分做准备

4. 提供 **按时间切分训练 / 测试集** 的函数：
   - `train_test_split_by_time(interaction, test_ratio=0.2)`

#### 3. `recall_cf.py` —— Item-based 协同过滤召回

主要职责：

- 实现 `ItemCFRecall` 类：
  - `fit(interaction_df)`：
    - 构建用户-电影评分矩阵（稀疏矩阵）
    - 计算电影-电影余弦相似度矩阵
  - `get_recall_items(user_id, top_k=50)`：
    - 为指定用户返回 50 个候选 `movie_id`
  - 代码中详细注释了 **余弦相似度公式** 与 **召回打分逻辑**

#### 4. `ranking_model.py` —— Logistic Regression 排序模型

主要职责：

- 实现 `LRRankingModel` 类：
  - `build_feature_dicts(user_profile, item_profile)`：
    - 建立 `user_id -> 用户特征向量`
    - 建立 `movie_id -> 电影特征向量`
  - `build_training_data(user_profile, item_profile, train_interaction)`：
    - 为每个用户构造正负样本
    - 特征为 `[用户特征 + 电影特征]`
  - `fit(X, y)`：
    - 使用 `sklearn.linear_model.LogisticRegression` 进行二分类训练
  - `predict_proba_for_pairs(user_id, movie_ids)`：
    - 对给定用户和一组候选电影输出“喜欢概率”

代码中对“为什么这么构造正负样本”和“为什么要拼接特征”都有清晰注释，适合课堂讲解。

#### 5. `recommender.py` —— 推荐系统主类（含冷启动）

主要职责：

- 实现 `Recommender` 类，将**数据处理 + 召回 + 排序 + 冷启动** 串联起来：
  - `prepare_data()`：
    - 使用 `DataLoader` 读取原始数据
    - 使用 `feature_engineering` 构建画像表 / 行为表
    - 按时间切分训练集与测试集
  - `train()`：
    - 训练 `ItemCFRecall` 召回模型
    - 构建排序训练数据并训练 `LRRankingModel`
    - 统计冷启动 Top-5 热门电影
  - `_is_new_user(user_id)`：
    - 判断用户是否在训练集中出现过，清晰标注“冷启动判断逻辑”
  - `recommend(user_id, top_n=5)`：
    - 新用户：直接返回冷启动热门 Top-5
    - 老用户：先召回 50 部候选，再用 LR 模型排序，输出 Top-5

#### 6. `evaluation.py` —— Precision@5 离线评估

主要职责：

- 提供 `precision_at_k(recommender, test_interaction, k=5)` 函数：
  - 基于测试集“未来 20% 的行为数据”
  - 计算每个用户的 Precision@5
  - 对所有用户求平均，得到系统整体的 Precision@5
- 在代码顶部和函数内详细解释了 **Precision@5 的业务含义**，非常适合写在实验报告中。

#### 7. `main.py` —— 命令行 Demo 入口

主要职责：

- 一键运行完整流程：
  1. 初始化 `Recommender`
  2. 调用 `prepare_data()`
  3. 调用 `train()`
  4. 调用 `precision_at_k()` 做一次离线评估
  5. 进入命令行交互，支持重复输入用户 ID 做推荐
- 用户体验：
  - 运行 `python main.py` 后，根据提示输入：
    - 用户 ID（1~943）
    - 输入 `q` 退出

---

### 四、如何运行项目

#### 1. 安装依赖

建议使用 Python 3.8+，并安装以下依赖：

```bash
pip install pandas numpy scikit-learn scipy
```

#### 2. 准备数据

将 MovieLens 100k 的三个文件：

- `u.user`
- `u.item`
- `u.data`

确保数据文件在 `ml-100k/` 目录下（项目已包含），目录结构如下：

```text
Recommender System/
├── ml-100k/
│   ├── u.user
│   ├── u.item
│   └── u.data
├── data_loader.py
├── feature_engineering.py
├── recall_cf.py
├── ranking_model.py
├── recommender.py
├── evaluation.py
├── main.py
└── README.md
```

#### 3. 运行命令行 Demo

在项目根目录执行：

```bash
python main.py
```

程序会依次完成：

1. 数据加载与特征工程  
2. 召回模型训练（ItemCF）  
3. 排序模型训练（Logistic Regression）  
4. 冷启动 Top-5 热门电影统计  
5. Precision@5 离线评估  
6. 进入命令行交互（输入用户 ID，输出 Top-5 推荐电影 ID 列表）  

#### 4. 快速检查是否运行成功

运行 `python main.py` 后，若看到大致如下输出，即表示流程正常：

```text
========== 数据加载与特征工程 ==========
用户画像表形状: ...
电影画像表形状: ...
训练集行为数: ...
...
========== 训练召回模型 (ItemCF) ==========
...
========== 训练排序模型 (Logistic Regression) ==========
...
========== 计算冷启动 Top-5 热门电影 ==========
冷启动 Top-5 电影 ID 列表: [...]

========== 离线评估：Precision@5 ==========
【Eval】有效用户数: ...
【Eval】Precision@5 = 0.0xxx
...
========== 进入命令行推荐 Demo ==========
请输入用户 ID（或输入 q 退出）:
```


