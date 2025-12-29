"""
ranking_model.py
----------------
排序层（Ranking）：使用 Logistic Regression（逻辑回归）模型，对
“用户-电影”对进行二分类预测，输出一个 0~1 之间的概率，表示：

    该用户喜欢 / 点击这部电影的概率。

训练样本构建规则：
--------------------------------
1. 正样本（label = 1）：
   - 用户对电影的评分 rating >= 4

2. 负样本（label = 0）：
   - 用户未看过（训练集中没有评分记录）的电影中，随机采样若干部，
     认为是“不喜欢 / 未点击”的近似样本。

3. 特征（Feature）：
   - 对于任意一个 (user_id, movie_id) 对：
       [用户特征向量 U, 电影特征向量 I] 直接拼接。
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class LRRankingModel:
    """
    使用 Logistic Regression 的排序模型。
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.model: LogisticRegression | None = None

        # 为了快速从 id 找到特征向量，提前构建两个字典
        self.user_feature_dict: Dict[int, np.ndarray] = {}
        self.item_feature_dict: Dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------
    # 特征字典构建
    # ------------------------------------------------------------------
    def build_feature_dicts(
        self,
        user_profile: pd.DataFrame,
        item_profile: pd.DataFrame,
    ) -> None:
        """
        将 user_profile 和 item_profile 转换为字典形式：

            user_feature_dict[user_id] = 用户特征向量 (ndarray)
            item_feature_dict[movie_id] = 电影特征向量 (ndarray)
        """
        # 用户特征：去掉 user_id 列，其余全部作为特征
        for _, row in user_profile.iterrows():
            uid = int(row["user_id"])
            feat = row.drop("user_id").values.astype(float)
            self.user_feature_dict[uid] = feat

        # 电影特征：去掉 movie_id 列，其余全部作为特征
        for _, row in item_profile.iterrows():
            mid = int(row["movie_id"])
            feat = row.drop("movie_id").values.astype(float)
            self.item_feature_dict[mid] = feat

    # ------------------------------------------------------------------
    # 为单个用户构造训练样本
    # ------------------------------------------------------------------
    def _build_samples_for_one_user(
        self,
        user_id: int,
        user_hist: pd.DataFrame,
        all_movie_ids: List[int],
        max_negative_per_user: int = 50,
    ) -> List[Tuple[np.ndarray, int]]:
        """
        为单个用户构造训练样本列表。

        :param user_id: 该用户的 ID
        :param user_hist: 该用户在训练集中的所有行为记录
        :param all_movie_ids: 所有在 item_profile 中出现的 movie_id
        :param max_negative_per_user: 每个用户最多采样多少个负样本
        :return: 列表 [(feature_vector, label), ...]
        """
        samples: List[Tuple[np.ndarray, int]] = []

        # 1. 正样本：评分 >= 4 的电影
        positive_movies = user_hist[user_hist["rating"] >= 4]["movie_id"].tolist()

        # 2. 用户看过的所有电影
        watched_movies = user_hist["movie_id"].unique().tolist()

        # 3. 负样本候选：所有电影 - 已看过的电影
        unwatched_movies = list(set(all_movie_ids) - set(watched_movies))

        # 如果没有正样本或没有未看过的电影，则无法为该用户构造有效样本
        if len(positive_movies) == 0 or len(unwatched_movies) == 0:
            return samples

        # 4. 随机采样负样本
        rng = np.random.default_rng(self.random_state + user_id)
        num_negative = min(max_negative_per_user, len(unwatched_movies))
        negative_movies = rng.choice(unwatched_movies, size=num_negative, replace=False)

        # 5. 构造正样本
        for mid in positive_movies:
            if user_id not in self.user_feature_dict:
                continue
            if mid not in self.item_feature_dict:
                continue
            u_feat = self.user_feature_dict[user_id]
            i_feat = self.item_feature_dict[mid]
            feature_vec = np.concatenate([u_feat, i_feat])
            samples.append((feature_vec, 1))

        # 6. 构造负样本
        for mid in negative_movies:
            if user_id not in self.user_feature_dict:
                continue
            if mid not in self.item_feature_dict:
                continue
            u_feat = self.user_feature_dict[user_id]
            i_feat = self.item_feature_dict[mid]
            feature_vec = np.concatenate([u_feat, i_feat])
            samples.append((feature_vec, 0))

        return samples

    # ------------------------------------------------------------------
    # 构建整体训练数据
    # ------------------------------------------------------------------
    def build_training_data(
        self,
        user_profile: pd.DataFrame,
        item_profile: pd.DataFrame,
        train_interaction: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        根据训练集行为数据构建排序模型的训练数据 (X, y)。

        :param user_profile: 用户画像表
        :param item_profile: 电影画像表
        :param train_interaction: 训练集行为表
        :return: (X, y)
        """
        print("【Ranking】开始构建训练样本...")

        # 先构建 id -> 特征向量 的映射字典
        self.build_feature_dicts(user_profile, item_profile)

        X_list: List[np.ndarray] = []
        y_list: List[int] = []

        all_movie_ids = item_profile["movie_id"].unique().tolist()

        # 按 user_id 分组，为每个用户构造样本
        for uid, user_hist in train_interaction.groupby("user_id"):
            user_id = int(uid)
            samples = self._build_samples_for_one_user(
                user_id=user_id,
                user_hist=user_hist,
                all_movie_ids=all_movie_ids,
                max_negative_per_user=50,
            )
            for feat_vec, label in samples:
                X_list.append(feat_vec)
                y_list.append(label)

        if not X_list:
            raise ValueError("【Ranking】没有构造出任何训练样本，请检查数据。")

        X = np.vstack(X_list)
        y = np.array(y_list, dtype=int)

        print(f"【Ranking】训练样本数: {len(y)}, 特征维度: {X.shape[1]}")
        return X, y

    # ------------------------------------------------------------------
    # 训练 LR 模型
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        训练 Logistic Regression 模型。
        """
        print("【Ranking】开始训练 Logistic Regression 模型...")
        self.model = LogisticRegression(
            max_iter=1000,
            random_state=self.random_state,
        )
        self.model.fit(X, y)
        print("【Ranking】模型训练完成。")

    # ------------------------------------------------------------------
    # 为给定用户和候选电影打分
    # ------------------------------------------------------------------
    def predict_proba_for_pairs(
        self,
        user_id: int,
        movie_ids: List[int],
    ) -> List[Tuple[int, float]]:
        """
        对指定用户与一组候选电影 (user_id, movie_id) 对，输出“喜欢概率”。

        返回格式：
            [(movie_id, prob), ...]
        其中 prob ∈ [0, 1]。
        """
        if self.model is None:
            raise ValueError("【Ranking】模型尚未训练，请先调用 fit()。")

        if user_id not in self.user_feature_dict:
            # 理论上对训练集中的用户不应发生
            return []

        user_feat = self.user_feature_dict[user_id]

        feature_vectors: List[np.ndarray] = []
        valid_movie_ids: List[int] = []

        for mid in movie_ids:
            if mid not in self.item_feature_dict:
                continue
            item_feat = self.item_feature_dict[mid]
            feat_vec = np.concatenate([user_feat, item_feat])
            feature_vectors.append(feat_vec)
            valid_movie_ids.append(mid)

        if not feature_vectors:
            return []

        X = np.vstack(feature_vectors)
        # predict_proba 返回 [N, 2]，第二列是 label=1 的概率
        probs = self.model.predict_proba(X)[:, 1]

        return list(zip(valid_movie_ids, probs))


