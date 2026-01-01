"""
recall_cf.py
--召回层（Recall）：基于物品的协同过滤（Item-based Collaborative Filtering）。

核心目标：
    对于每个用户，从所有电影中快速“召回”一小部分候选电影（例如 50 部），
    这些候选之后会交给排序层做精排。

核心思想（非常重要，适合课堂讲解）：
-------------------------------------------------
1. 用户 A 喜欢电影 X
2. 很多和 A 有相似品味的用户，也喜欢电影 X 和电影 Y
3. 如果“喜欢 X 的人通常也喜欢 Y”，说明 X 和 Y 在“行为上很相似”
4. 因此，当 A 喜欢 X 时，推荐 Y 也很有可能被 A 喜欢

数学上，我们为每部电影 i 构造一个“评分向量”（所有用户对它的评分），
然后使用 **余弦相似度** 来度量两部电影 i、j 的相似度：

    sim(i, j) = (R_i · R_j) / (||R_i|| * ||R_j||)

其中 R_i 表示电影 i 在不同用户上的评分向量。
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


class ItemCFRecall:
    """
    Item-based 协同过滤召回模型。
    """

    def __init__(self, top_k_similar: int = 50) -> None:
        """
        :param top_k_similar: 计算相似度时，每部电影最多保留多少个最相似邻居
                              （本示例简单实现，不做截断优化，只保留完整矩阵）
        """
        self.top_k_similar = top_k_similar

        # 用户 / 电影 ID 映射到矩阵行列索引
        self.user_ids: List[int] = []
        self.movie_ids: List[int] = []
        self.user_id_to_index: Dict[int, int] = {}
        self.movie_id_to_index: Dict[int, int] = {}
        self.movie_index_to_id: Dict[int, int] = {}

        # 用户-电影评分矩阵（稀疏）
        self.user_item_matrix: csr_matrix | None = None
        # 电影-电影相似度矩阵（稠密）
        self.item_sim_matrix: np.ndarray | None = None

    # ------------------------------------------------------------------
    # 构建用户-电影评分矩阵
    # ------------------------------------------------------------------
    def _build_user_item_matrix(self, interaction_df: pd.DataFrame) -> None:
        """
        根据行为表构建用户-电影评分矩阵 R（稀疏矩阵）。

        行为表格式：
            user_id, movie_id, rating, timestamp, datetime
        """
        # 固定排序，确保 id 与 index 的映射稳定
        self.user_ids = sorted(interaction_df["user_id"].unique())
        self.movie_ids = sorted(interaction_df["movie_id"].unique())

        self.user_id_to_index = {uid: idx for idx, uid in enumerate(self.user_ids)}
        self.movie_id_to_index = {mid: idx for idx, mid in enumerate(self.movie_ids)}
        self.movie_index_to_id = {idx: mid for mid, idx in self.movie_id_to_index.items()}

        # 构造 CSR 矩阵需要三部分：行索引、列索引、数据
        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []

        for _, row in interaction_df.iterrows():
            u = int(row["user_id"])
            m = int(row["movie_id"])
            r = float(row["rating"])

            rows.append(self.user_id_to_index[u])
            cols.append(self.movie_id_to_index[m])
            data.append(r)

        num_users = len(self.user_ids)
        num_movies = len(self.movie_ids)

        self.user_item_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(num_users, num_movies),
        )

    # ------------------------------------------------------------------
    # 训练（拟合）ItemCF 模型
    # ------------------------------------------------------------------
    def fit(self, interaction_df: pd.DataFrame) -> None:
        """
        训练 Item-based CF：
        1. 构建用户-电影评分矩阵 R
        2. 通过余弦相似度计算电影-电影相似度矩阵 S
        """
        print("【ItemCF】开始构建用户-电影评分矩阵...")
        self._build_user_item_matrix(interaction_df)

        print("【ItemCF】开始计算电影-电影余弦相似度矩阵...")
        assert self.user_item_matrix is not None

        # user_item_matrix 形状为 [num_users, num_movies]
        # 转置后 item_user_matrix 为 [num_movies, num_users]
        item_user_matrix = self.user_item_matrix.T

        # 使用 sklearn.metrics.pairwise.cosine_similarity 计算余弦相似度
        # 得到的 item_sim_matrix 形状为 [num_movies, num_movies]
        self.item_sim_matrix = cosine_similarity(item_user_matrix, dense_output=True)

        print("【ItemCF】相似度矩阵计算完成。")

    # ------------------------------------------------------------------
    # 为单个用户做召回
    # ------------------------------------------------------------------
    def get_recall_items(self, user_id: int, top_k: int = 50) -> List[int]:
        """
        为指定用户做召回，返回 top_k 个候选 movie_id。

        计算逻辑（关键公式，适合答辩时讲解）：
        -------------------------------------------------
        记 I(u) 为用户 u 已经评分过的电影集合。
        对于某个候选电影 j，我们定义：

            score(u, j) = sum_{i in I(u)} [ sim(i, j) * rating(u, i) ]

        即：用户对电影 j 的“兴趣分数”，等于他看过的每一部电影 i
            与 j 的相似度 * 他对 i 的评分 的加权和。

        最后，我们按 score(u, j) 从大到小排序，取前 top_k 个电影。
        """
        # 冷启动场景：用户没出现在训练数据中，召回阶段直接返回空列表
        if user_id not in self.user_id_to_index:
            return []

        if self.user_item_matrix is None or self.item_sim_matrix is None:
            raise ValueError("【ItemCF】模型尚未训练，请先调用 fit()。")

        user_index = self.user_id_to_index[user_id]

        # 取出该用户在所有电影上的评分向量：shape = [num_movies]
        user_ratings = self.user_item_matrix[user_index].toarray().flatten()

        # 用户评分过的电影的索引（评分 > 0）
        rated_movie_indices = np.where(user_ratings > 0)[0]
        if len(rated_movie_indices) == 0:
            # 没有历史行为，也算是一种冷启动，返回空列表
            return []

        # 初始化候选电影的得分数组
        num_movies = len(self.movie_ids)
        candidate_scores = np.zeros(num_movies, dtype=float)

        # 遍历用户评分过的每一部电影 i
        for i in rated_movie_indices:
            rating_ui = user_ratings[i]          # 用户对电影 i 的评分
            sim_vector = self.item_sim_matrix[i]  # 电影 i 与所有电影的相似度

            # 对所有候选电影 j，累加 sim(i, j) * rating(u, i)
            candidate_scores += sim_vector * rating_ui

        # 不推荐用户已经看过的电影：将这些索引位置的得分设为一个很小的值
        candidate_scores[rated_movie_indices] = -np.inf

        # 选取得分最高的 top_k 个电影索引
        top_indices = np.argsort(candidate_scores)[::-1][:top_k]

        # 映射回 movie_id，并过滤掉得分为 -inf 的无效项
        recall_movie_ids: List[int] = []
        for idx in top_indices:
            if candidate_scores[idx] == -np.inf:
                continue
            mid = self.movie_index_to_id[idx]
            recall_movie_ids.append(mid)

        return recall_movie_ids[:top_k]

    # ------------------------------------------------------------------
    # 批量召回（可选，用于评估）
    # ------------------------------------------------------------------
    def get_recall_items_batch(self, user_ids: List[int], top_k: int = 50) -> Dict[int, List[int]]:
        """
        为一组用户批量做召回，返回 {user_id: [movie_id1, movie_id2, ...]}。
        """
        result: Dict[int, List[int]] = {}
        for uid in user_ids:
            result[uid] = self.get_recall_items(uid, top_k=top_k)
        return result


