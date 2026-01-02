"""
feature_engineering.py
--特征工程模块：在这里把“原始表”转换为“画像表”和“行为表”。

目标产物：
1. user_profile（用户画像表）
   - user_id
   - age_normalized （标准化后的年龄）
   - gender_M, gender_F （二值编码）
   - occupation_* （职业 one-hot）

2. item_profile（电影画像表）
   - movie_id
   - 各类型 one-hot（直接使用 u.item 中的 0/1 列）

3. interaction（行为表）
   - user_id
   - movie_id
   - rating
   - timestamp
   - datetime （时间戳转为可读时间）

4. train / test 切分（按时间先后 80% / 20%）
"""

from typing import Tuple

import numpy as np
import pandas as pd


def build_user_profile(user_raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    构建用户画像表 user_profile。

    步骤说明：
    1. 性别 gender：转换为 gender_M / gender_F 两个 0/1 特征
    2. 职业 occupation：做 one-hot 编码（每种职业一个 0/1 列）
    3. 年龄 age：做标准化（减去均值 / 除以标准差）
    """
    df = user_raw_df.copy()

    # 1. 性别二值编码
    df["gender_M"] = (df["gender"] == "M").astype(int)
    df["gender_F"] = (df["gender"] == "F").astype(int)

    # 2. 职业 one-hot
    occupation_dummies = pd.get_dummies(
        df["occupation"],
        prefix="occ",
        dtype=int,
    )
    df = pd.concat([df, occupation_dummies], axis=1)

    # 3. 年龄标准化
    age_mean = df["age"].mean()
    age_std = df["age"].std()
    df["age_normalized"] = (df["age"] - age_mean) / age_std

    # 4. 选择输出列
    feature_cols = ["user_id", "age_normalized", "gender_M", "gender_F"] + [
        c for c in df.columns if c.startswith("occ_")
    ]
    user_profile = df[feature_cols].reset_index(drop=True)

    return user_profile


def compute_shrinkage_ratings(
    item_raw_df: pd.DataFrame,
    interaction_df: pd.DataFrame,
    shrinkage_factor: float = 5.0,
) -> pd.DataFrame:
    """
    计算每部电影的"收缩后平均评分"（Shrinkage Estimation）。

    核心思路（降噪方法）：
    --------------------------------
    当个别电影的评分数据不足时（例如只有1个人评分），这个评分可能是噪声。
    我们使用"同类电影的平均评分"来修正它，这就是"收缩估计"（Shrinkage Estimation）。

    公式：
        shrinkage_rating = α × movie_avg_rating + (1 - α) × genre_avg_rating

    其中：
        α = rating_count / (rating_count + shrinkage_factor)

    说明：
        - rating_count 越大，α 越接近 1，越信任该电影自身的平均评分
        - rating_count 越小，α 越小，越信任同类电影的平均评分
        - shrinkage_factor 是超参数，通常设为 5（可根据实际情况调整）

    例如：
        - 如果电影 A 只有1个人评分（rating_count=1），α ≈ 0.17
        - 如果电影 A 有10个人评分（rating_count=10），α ≈ 0.67
    """
    # 1. 计算每部电影的原始平均评分和评分次数
    movie_stats = (
        interaction_df.groupby("movie_id")["rating"]
        .agg(["mean", "count"])
        .reset_index()
    )
    movie_stats.rename(
        columns={"mean": "movie_avg_rating", "count": "rating_count"}, inplace=True
    )

    # 2. 获取类型列
    genre_cols = [
        c
        for c in item_raw_df.columns
        if c
        not in ["movie_id", "title", "release_date", "video_release_date", "imdb_url"]
    ]

    # 3. 计算全局平均评分（用于没有类型的电影）
    global_avg_rating = interaction_df["rating"].mean()

    # 4. 为每部电影计算"同类型电影的平均评分"
    shrinkage_ratings = []

    for _, movie_row in item_raw_df.iterrows():
        movie_id = movie_row["movie_id"]

        # 获取该电影的类型标签（1表示有该类型，0表示无）
        movie_genres = set(
            [col for col in genre_cols if movie_row[col] == 1]
        )  # 该电影所属的类型集合

        if not movie_genres:
            # 如果电影没有任何类型标签，使用全局平均评分作为类别平均
            genre_avg_rating = global_avg_rating
        else:
            # 找出所有与该电影有共同类型的电影（至少有一个共同类型）
            similar_movies = []
            for _, other_movie in item_raw_df.iterrows():
                if other_movie["movie_id"] == movie_id:
                    continue  # 跳过自己
                other_genres = set(
                    [col for col in genre_cols if other_movie[col] == 1]
                )
                if movie_genres & other_genres:  # 有交集
                    similar_movies.append(other_movie["movie_id"])

            if similar_movies:
                # 计算同类电影的平均评分（从评分数据中筛选）
                similar_ratings = interaction_df[
                    interaction_df["movie_id"].isin(similar_movies)
                ]["rating"]
                if len(similar_ratings) > 0:
                    genre_avg_rating = similar_ratings.mean()
                else:
                    genre_avg_rating = global_avg_rating
            else:
                # 如果没有同类电影，使用全局平均
                genre_avg_rating = global_avg_rating

        # 获取该电影的原始平均评分和评分次数
        movie_stat = movie_stats[movie_stats["movie_id"] == movie_id]
        if len(movie_stat) > 0:
            movie_avg_rating = movie_stat["movie_avg_rating"].values[0]
            rating_count = movie_stat["rating_count"].values[0]
        else:
            # 如果该电影没有任何评分，使用类别平均评分
            movie_avg_rating = genre_avg_rating
            rating_count = 0

        # 5. 计算收缩系数 α
        if rating_count > 0:
            alpha = rating_count / (rating_count + shrinkage_factor)
        else:
            alpha = 0.0  # 没有评分时，完全信任类别平均

        # 6. 应用收缩公式
        shrinkage_rating = alpha * movie_avg_rating + (1 - alpha) * genre_avg_rating

        shrinkage_ratings.append(
            {
                "movie_id": movie_id,
                "shrinkage_rating": shrinkage_rating,
            }
        )

    return pd.DataFrame(shrinkage_ratings)


def compute_shrinkage_ratings_vectorized(
    item_raw_df: pd.DataFrame,
    interaction_df: pd.DataFrame,
    shrinkage_factor: float = 5.0,
) -> pd.DataFrame:
    """
    计算每部电影的"收缩后平均评分"（Shrinkage Estimation）- 向量化优化版本。

    算法逻辑与原版本完全相同，但使用向量化操作大幅提升性能。

    核心思路（降噪方法）：
    --------------------------------
    当个别电影的评分数据不足时（例如只有1个人评分），这个评分可能是噪声。
    我们使用"同类电影的平均评分"来修正它，这就是"收缩估计"（Shrinkage Estimation）。

    公式：
        shrinkage_rating = α × movie_avg_rating + (1 - α) × genre_avg_rating

    其中：
        α = rating_count / (rating_count + shrinkage_factor)

    优化策略：
    --------------------------------
    1. 使用矩阵乘法快速找出同类电影（替代双重循环）
    2. 使用pandas向量化操作批量计算
    3. 预先构建索引和映射，避免重复计算
    """
    # 1. 计算每部电影的原始平均评分和评分次数
    movie_stats = (
        interaction_df.groupby("movie_id")["rating"]
        .agg(["mean", "count"])
        .reset_index()
    )
    movie_stats.rename(
        columns={"mean": "movie_avg_rating", "count": "rating_count"}, inplace=True
    )

    # 2. 获取类型列
    genre_cols = [
        c
        for c in item_raw_df.columns
        if c
        not in ["movie_id", "title", "release_date", "video_release_date", "imdb_url"]
    ]

    # 3. 计算全局平均评分（用于没有类型的电影）
    global_avg_rating = interaction_df["rating"].mean()

    # 4. 构建电影类型矩阵（向量化优化核心）
    # 按 movie_id 排序，确保顺序一致
    item_sorted = item_raw_df.sort_values("movie_id").reset_index(drop=True)
    movie_ids = item_sorted["movie_id"].values
    
    # 构建类型矩阵：shape = [num_movies, num_genres]
    genre_matrix = item_sorted[genre_cols].values.astype(float)
    
    # 5. 计算电影之间的类型重叠矩阵（向量化）
    # genre_overlap[i, j] = 电影i和电影j的共同类型数量
    # 如果 > 0，说明有共同类型
    genre_overlap = genre_matrix @ genre_matrix.T  # [num_movies, num_movies]
    
    # 6. 为每部电影计算"同类型电影的平均评分"（向量化）
    shrinkage_ratings = []
    
    # 预先构建 movie_id 到索引的映射（虽然这里没用到，但保留以备后用）
    # movie_id_to_idx = {mid: idx for idx, mid in enumerate(movie_ids)}
    
    # 预先构建映射（用于快速查找，避免重复的DataFrame查询）
    movie_id_to_avg_rating = dict(zip(movie_stats["movie_id"], movie_stats["movie_avg_rating"]))
    movie_id_to_rating_count = dict(zip(movie_stats["movie_id"], movie_stats["rating_count"]))
    
    # 预先为每部电影计算同类电影的平均评分（向量化优化）
    # 对于每部电影，找出所有同类电影，然后批量计算平均评分
    genre_avg_ratings = np.zeros(len(movie_ids))
    
    for i, movie_id in enumerate(movie_ids):
        # 找出所有有共同类型的电影（重叠数 > 0，且不是自己）
        similar_indices = np.where((genre_overlap[i] > 0) & (np.arange(len(movie_ids)) != i))[0]
        
        if len(similar_indices) == 0:
            # 如果没有同类电影，使用全局平均
            genre_avg_ratings[i] = global_avg_rating
        else:
            # 获取同类电影的 movie_id
            similar_movie_ids = movie_ids[similar_indices]
            
            # 批量计算同类电影的平均评分（向量化）
            # 从 interaction_df 中筛选出这些同类电影的评分
            similar_ratings = interaction_df[
                interaction_df["movie_id"].isin(similar_movie_ids)
            ]["rating"]
            
            if len(similar_ratings) > 0:
                genre_avg_ratings[i] = similar_ratings.mean()
            else:
                genre_avg_ratings[i] = global_avg_rating
    
    # 批量计算所有电影的收缩评分（向量化）
    shrinkage_ratings = []
    
    for i, movie_id in enumerate(movie_ids):
        genre_avg_rating = genre_avg_ratings[i]

        # 获取该电影的原始平均评分和评分次数（使用预先构建的映射，避免重复查询）
        if movie_id in movie_id_to_avg_rating:
            movie_avg_rating = movie_id_to_avg_rating[movie_id]
            rating_count = movie_id_to_rating_count[movie_id]
        else:
            # 如果该电影没有任何评分，使用类别平均评分
            movie_avg_rating = genre_avg_rating
            rating_count = 0

        # 7. 计算收缩系数 α
        if rating_count > 0:
            alpha = rating_count / (rating_count + shrinkage_factor)
        else:
            alpha = 0.0  # 没有评分时，完全信任类别平均

        # 8. 应用收缩公式
        shrinkage_rating = alpha * movie_avg_rating + (1 - alpha) * genre_avg_rating

        shrinkage_ratings.append(
            {
                "movie_id": movie_id,
                "shrinkage_rating": shrinkage_rating,
            }
        )

    return pd.DataFrame(shrinkage_ratings)


def build_item_profile(
    item_raw_df: pd.DataFrame,
    interaction_df: pd.DataFrame | None = None,
    shrinkage_factor: float = 5.0,
) -> pd.DataFrame:
    """
    构建电影画像表 item_profile。

    MovieLens 的 u.item 自带 19 个类型字段（unknown, Action, ..., Western），
    每一列已经是 0/1 编码，这里我们只需要挑出这些列即可。

    如果提供了 interaction_df，则会计算"收缩后平均评分"作为额外特征。
    """
    df = item_raw_df.copy()

    # 除去 movie_id, title 以及其他非类型列，剩下的就是类型 one-hot
    genre_cols = [
        c
        for c in df.columns
        if c
        not in ["movie_id", "title", "release_date", "video_release_date", "imdb_url"]
    ]

    item_profile = df[["movie_id"] + genre_cols].copy()

    # 如果提供了评分数据，计算收缩后平均评分并加入特征
    if interaction_df is not None:
        # 使用向量化优化版本（算法逻辑完全相同，但速度更快）
        shrinkage_df = compute_shrinkage_ratings_vectorized(
            item_raw_df, interaction_df, shrinkage_factor=shrinkage_factor
        )
        item_profile = item_profile.merge(shrinkage_df, on="movie_id", how="left")
        # 填充缺失值（理论上不应该有，但为了安全）
        item_profile["shrinkage_rating"] = item_profile["shrinkage_rating"].fillna(
            interaction_df["rating"].mean()
        )

    item_profile = item_profile.reset_index(drop=True)
    return item_profile


def build_interaction(interaction_raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    构建行为表 interaction。

    主要处理：
    1. 保留必要字段：user_id, movie_id, rating, timestamp
    2. 增加 datetime 字段，便于按时间排序与切分
    3. 按时间排序
    """
    df = interaction_raw_df.copy()

    # MovieLens 数据本身很干净，这里简单过滤 rating > 0 即可
    df = df[df["rating"] > 0]

    # 增加 datetime 字段
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")

    # 按时间排序，更贴近真实场景
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def train_test_split_by_time(
    interaction: pd.DataFrame,
    test_ratio: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    按时间先后顺序切分训练集 / 测试集。

    - 前 80% 时间段的数据作为训练集
    - 后 20% 时间段的数据作为测试集

    这样可以模拟“用过去的数据训练，用未来的数据评估”的业务场景。
    """
    num_total = len(interaction)
    split_idx = int(num_total * (1.0 - test_ratio))

    train_df = interaction.iloc[:split_idx].copy()
    test_df = interaction.iloc[split_idx:].copy()

    return train_df, test_df


