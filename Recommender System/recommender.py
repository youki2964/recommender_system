"""
recommender.py
--------------
推荐系统主类：把整个流程“串”起来。

包含：
1. 数据加载与特征工程
2. 召回层：基于物品协同过滤 (ItemCFRecall)
3. 排序层：Logistic Regression 排序模型 (LRRankingModel)
4. 冷启动策略：新用户推荐“平均评分最高的 Top-5 电影”
5. 对外接口：recommend(user_id, top_n=5)
"""

from typing import List, Tuple

import pandas as pd

from data_loader import DataLoader
from feature_engineering import (
    build_user_profile,
    build_item_profile,
    build_interaction,
    train_test_split_by_time,
)
from recall_cf import ItemCFRecall
from ranking_model import LRRankingModel


class Recommender:
    """
    推荐系统主类。

    使用示例：
        rec = Recommender(data_dir="data")
        rec.prepare_data()
        rec.train()
        rec.recommend(user_id=1, top_n=5)
    """

    def __init__(self, data_dir: str = "ml-100k") -> None:
        self.data_dir = data_dir

        # 原始数据
        self.user_raw_df: pd.DataFrame | None = None
        self.item_raw_df: pd.DataFrame | None = None
        self.interaction_raw_df: pd.DataFrame | None = None

        # 画像表 / 行为表
        self.user_profile: pd.DataFrame | None = None
        self.item_profile: pd.DataFrame | None = None
        self.interaction: pd.DataFrame | None = None
        self.train_interaction: pd.DataFrame | None = None
        self.test_interaction: pd.DataFrame | None = None

        # 模型
        self.recall_model = ItemCFRecall()
        self.ranking_model = LRRankingModel()

        # 冷启动用的热门电影列表（movie_id）
        self.cold_start_top_movies: List[int] = []

    # ------------------------------------------------------------------
    # 数据准备：加载 + 特征工程 + 时间切分
    # ------------------------------------------------------------------
    def prepare_data(self) -> None:
        """
        完成数据准备工作：
        1. 从 data/ 目录读取原始数据
        2. 构建 user_profile / item_profile / interaction
        3. 按时间 80% / 20% 切分 train / test
        """
        print("========== 数据加载与特征工程 ==========")
        loader = DataLoader(self.data_dir)
        self.user_raw_df, self.item_raw_df, self.interaction_raw_df = loader.load_all()

        # 构建画像表
        self.user_profile = build_user_profile(self.user_raw_df)
        self.item_profile = build_item_profile(self.item_raw_df)
        self.interaction = build_interaction(self.interaction_raw_df)

        # 时间切分训练 / 测试
        self.train_interaction, self.test_interaction = train_test_split_by_time(
            self.interaction, test_ratio=0.2
        )

        print(f"用户画像表形状: {self.user_profile.shape}")
        print(f"电影画像表形状: {self.item_profile.shape}")
        print(
            f"训练集行为数: {len(self.train_interaction)}, 测试集行为数: {len(self.test_interaction)}"
        )

    # ------------------------------------------------------------------
    # 训练召回 + 排序 + 冷启动
    # ------------------------------------------------------------------
    def train(self) -> None:
        """
        训练召回模型和排序模型，并计算冷启动使用的热门电影。
        """
        if self.user_profile is None or self.item_profile is None or self.train_interaction is None:
            raise ValueError("请先调用 prepare_data() 完成数据准备。")

        # 1. 训练召回模型
        print("\n========== 训练召回模型 (ItemCF) ==========")
        self.recall_model.fit(self.train_interaction)

        # 2. 构建排序层训练数据
        print("\n========== 构建排序模型训练数据 ==========")
        X, y = self.ranking_model.build_training_data(
            user_profile=self.user_profile,
            item_profile=self.item_profile,
            train_interaction=self.train_interaction,
        )

        # 3. 训练排序模型
        print("\n========== 训练排序模型 (Logistic Regression) ==========")
        self.ranking_model.fit(X, y)

        # 4. 计算冷启动用的热门电影
        print("\n========== 计算冷启动 Top-5 热门电影 ==========")
        self._build_cold_start_movies()
        print(f"冷启动 Top-5 电影 ID 列表: {self.cold_start_top_movies}")

    def _build_cold_start_movies(self, min_count: int = 10, top_n: int = 5) -> None:
        """
        冷启动策略中需要的“热门电影”计算逻辑：

        1. 在训练集 train_interaction 中，按 movie_id 分组
        2. 计算每部电影的：
           - 平均评分 avg_rating
           - 评分次数 rating_count
        3. 只保留 rating_count >= min_count 的电影，避免“极少数人打高分”的噪声
        4. 按 avg_rating 从高到低排序，取前 top_n 个 movie_id
        """
        if self.train_interaction is None:
            raise ValueError("train_interaction 为空，请确认已调用 prepare_data()。")

        movie_stats = (
            self.train_interaction.groupby("movie_id")["rating"]
            .agg(["mean", "count"])
            .reset_index()
        )
        movie_stats.rename(
            columns={"mean": "avg_rating", "count": "rating_count"}, inplace=True
        )

        # 过滤掉评分次数太少的
        movie_stats = movie_stats[movie_stats["rating_count"] >= min_count]
        movie_stats = movie_stats.sort_values("avg_rating", ascending=False)

        self.cold_start_top_movies = movie_stats["movie_id"].head(top_n).tolist()

    # ------------------------------------------------------------------
    # 冷启动判断逻辑（强可解释性）
    # ------------------------------------------------------------------
    def _is_new_user(self, user_id: int) -> bool:
        """
        判断 user_id 是否为“新用户”（训练集中从未出现过）。

        逻辑解释：
        --------------------------------
        - 如果 user_profile 中没有该 user_id 对应的行，
          说明这个用户没有任何历史行为（在训练阶段没见过）
        - 对于这样的用户：
          - 协同过滤算不出“相似用户 / 相似物品”
          - 排序模型也拿不到用户特征
        - 所以我们使用一个简单但有效的“热门电影兜底策略”
        """
        if self.user_profile is None:
            return True
        return user_id not in set(self.user_profile["user_id"].tolist())

    # ------------------------------------------------------------------
    # 对外推荐接口
    # ------------------------------------------------------------------
    def recommend(self, user_id: int, top_n: int = 5) -> List[int]:
        """
        推荐接口：为指定用户返回 Top-N 电影 ID 列表。

        完整业务流程：
        --------------------------------
        1. 先判断用户是不是“新用户”（冷启动判断逻辑）
           - 新用户：直接返回“热门 Top-N 电影”
        2. 老用户：
           - 召回层：使用 ItemCF 召回候选电影（默认 50 部）
           - 排序层：使用 LR 模型对候选电影打“喜欢概率分数”
           - 按概率从高到低排序，取前 Top-N
        """
        # 1. 冷启动判断
        if self._is_new_user(user_id):
            print(f"【Recommender】用户 {user_id} 是新用户，使用冷启动策略推荐。")
            return self.cold_start_top_movies[:top_n]

        # 2. 老用户 => 召回候选电影
        recall_movie_ids = self.recall_model.get_recall_items(user_id, top_k=50)

        if not recall_movie_ids:
            return self.cold_start_top_movies[:top_n]

        # 3. 排序层，使用 LR 模型打分
        movie_prob_pairs = self.ranking_model.predict_proba_for_pairs(
            user_id=user_id,
            movie_ids=recall_movie_ids,
        )
        if not movie_prob_pairs:
            print(
                f"【Recommender】用户 {user_id} 在排序阶段无有效候选，使用冷启动策略兜底。"
            )
            return self.cold_start_top_movies[:top_n]

        # 4. 按概率从大到小排序，取前 top_n
        movie_prob_pairs.sort(key=lambda x: x[1], reverse=True)
        top_movies = [mid for mid, _ in movie_prob_pairs[:top_n]]
        return top_movies

    # ------------------------------------------------------------------
    # 提供训练 / 测试行为表给评估模块使用
    # ------------------------------------------------------------------
    def get_train_test_interaction(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        返回训练集与测试集行为表，用于离线评估。
        """
        return self.train_interaction, self.test_interaction


